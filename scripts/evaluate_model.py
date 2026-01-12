#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for French Playing Cards Detector

This script evaluates YOLO models trained on French playing cards with robust error handling,
automatic head replacement detection, and comprehensive diagnostics.

Features:
- Automatic model loading with head replacement compatibility
- Robust PyTorch state_dict handling
- Comprehensive class validation and diagnostics
- Sample prediction testing with confusion analysis
- Fallback evaluation mechanisms
- Detailed architectural diagnostics
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback
import glob
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from src.head_replacer import YOLOHeadReplacer, replace_yolo_head_if_needed, validate_yolo_model_architecture


def safe_format(metric: Any, name: str) -> str:
    """
    Safely format metric values that might be scalars, arrays, or tensors.

    Args:
        metric: The metric value to format
        name: Name of the metric for error reporting

    Returns:
        Formatted string representation
    """
    try:
        # Handle None values
        if metric is None:
            return "N/A"

        # Handle scalar values
        if isinstance(metric, (int, float)):
            return f"{metric:.4f}"

        # Handle PyTorch tensors
        if hasattr(metric, 'item'):
            return f"{metric.item():.4f}"

        # Handle numpy arrays
        if hasattr(metric, 'mean'):
            if hasattr(metric, 'flatten'):
                flat_metric = metric.flatten()
                return f"{flat_metric.mean():.4f}" if len(flat_metric) > 0 else "N/A"
            else:
                return f"{metric.mean():.4f}"

        # Handle iterables (but not strings)
        if hasattr(metric, '__iter__') and not isinstance(metric, str):
            try:
                # Try to get mean/average
                if hasattr(metric, '__len__') and len(metric) > 0:
                    return f"{sum(metric) / len(metric):.4f}"
            except:
                pass

        # Fallback to string conversion
        try:
            return f"{float(metric):.4f}"
        except (ValueError, TypeError):
            return str(metric)

    except Exception as format_error:
        return f"FORMAT_ERROR({name})"


def find_latest_model() -> Optional[Path]:
    """
    Find the most recently modified model file in runs/ directory.

    Returns:
        Path to the latest model file, or None if no models found
    """
    runs_dir = Path("runs")

    if not runs_dir.exists():
        print("❌ No runs/ directory found")
        return None

    model_paths = []

    # Look for model files in runs directory and subdirectories
    for model_file in runs_dir.rglob("*.pt"):
        model_paths.append(model_file)

    if not model_paths:
        print("❌ No model files found in runs/ directory")
        return None

    # Sort by modification time (most recent first)
    latest_model = max(model_paths, key=lambda x: x.stat().st_mtime)

    print(f"📁 Found latest model: {latest_model}")
    print(f"📅 Modified: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")

    return latest_model


def load_dataset_config(config_path: str = "datasets/unified/data.yaml") -> Optional[Dict]:
    """
    Load dataset configuration from YAML file.

    Args:
        config_path: Path to the dataset configuration file

    Returns:
        Dataset configuration dictionary, or None if loading failed
    """
    if not os.path.exists(config_path):
        print(f"❌ Dataset config not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_keys = ['nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key '{key}' in dataset config")
                return None

        print(f"✅ Dataset config loaded: {config['nc']} classes")
        return config

    except Exception as e:
        print(f"❌ Failed to load dataset config: {e}")
        return None


def load_model_with_head_replacement(
    model_path: str,
    config: Dict,
    device: str,
    force_replacement: bool = False,
    disable_replacement: bool = False
) -> Tuple[Optional[YOLO], bool]:
    """
    Load YOLO model with automatic head replacement and validation.

    Args:
        model_path: Path to the model file
        config: Dataset configuration
        device: Device to load the model on
        force_replacement: Force head replacement even if classes match
        disable_replacement: Disable automatic head replacement

    Returns:
        Tuple of (loaded_model, was_head_replaced)
    """
    print(f"🚀 Loading model: {model_path}")
    print(f"🔧 Using device: {device}")

    try:
        # Determine if we need to use head replacement
        if model_path.endswith('.pt') and os.path.exists(model_path):
            # Check if this is a raw PyTorch state_dict or YOLO format
            try:
                checkpoint = torch.load(model_path, map_location=device)
                is_state_dict = all(isinstance(k, str) and k.startswith('model.') for k in checkpoint.keys())
            except:
                is_state_dict = False

            if is_state_dict:
                print("📋 Detected PyTorch state_dict format")
                return _load_pytorch_state_dict_model(model_path, config, device, force_replacement, disable_replacement)
            else:
                print("📋 Detected YOLO format checkpoint")
                return _load_yolo_format_model(model_path, config, device, force_replacement, disable_replacement)
        else:
            # Load base YOLO model
            print("📋 Loading base YOLO model")
            return _load_base_yolo_model(model_path, config, device, force_replacement, disable_replacement)

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return None, False


def _load_pytorch_state_dict_model(
    model_path: str,
    config: Dict,
    device: str,
    force_replacement: bool,
    disable_replacement: bool
) -> Tuple[Optional[YOLO], bool]:
    """Load model from PyTorch state_dict with head replacement."""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✅ Checkpoint loaded with {len(checkpoint)} keys")

        # Create base YOLO model
        yolo_model = YOLO('yolo11n.pt')
        model = yolo_model.model
        model.to(device)

        # Apply head replacement if needed
        was_replaced = False
        if not disable_replacement:
            if force_replacement:
                print("🔄 Forcing head replacement")
                replacer = YOLOHeadReplacer(model, 'datasets/unified/data.yaml', verbose=True)
                model = replacer.replace_classification_head()
                was_replaced = True
            else:
                model, was_replaced = replace_yolo_head_if_needed(model, 'datasets/unified/data.yaml', verbose=True)

        # Load state_dict weights
        print("⚡ Loading checkpoint weights...")
        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                state_dict[new_key] = value

        # Load with strict=False to handle potential mismatches
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"⚠️ Missing keys in model: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)}")

            print("✅ Weights loaded successfully")

        except Exception as load_error:
            print(f"❌ Failed to load weights: {load_error}")
            print("🔄 Continuing with base model weights")

        # Update model class information
        model.nc = config['nc']
        if hasattr(model, 'names'):
            model.names = {i: name for i, name in enumerate(config['names'])}

        print(f"🏷️ Updated model to {config['nc']} classes")

        # Move to device after all modifications
        yolo_model.model = model
        yolo_model.model.to(device)

        return yolo_model, was_replaced

    except Exception as e:
        print(f"❌ PyTorch state_dict loading failed: {e}")
        return None, False


def _load_yolo_format_model(
    model_path: str,
    config: Dict,
    device: str,
    force_replacement: bool,
    disable_replacement: bool
) -> Tuple[Optional[YOLO], bool]:
    """Load YOLO format model with potential head replacement."""
    try:
        # Load YOLO model directly
        yolo_model = YOLO(model_path)
        yolo_model.model.to(device)

        # Apply head replacement if needed
        was_replaced = False
        if not disable_replacement:
            if force_replacement:
                print("🔄 Forcing head replacement")
                replacer = YOLOHeadReplacer(yolo_model.model, 'datasets/unified/data.yaml', verbose=True)
                yolo_model.model = replacer.replace_classification_head()
                was_replaced = True
            else:
                yolo_model.model, was_replaced = replace_yolo_head_if_needed(
                    yolo_model.model, 'datasets/unified/data.yaml', verbose=True
                )

        # Update class information if needed
        if hasattr(yolo_model.model, 'nc'):
            yolo_model.model.nc = config['nc']
        if hasattr(yolo_model.model, 'names'):
            yolo_model.model.names = {i: name for i, name in enumerate(config['names'])}

        return yolo_model, was_replaced

    except Exception as e:
        print(f"❌ YOLO format loading failed: {e}")
        return None, False


def _load_base_yolo_model(
    model_path: str,
    config: Dict,
    device: str,
    force_replacement: bool,
    disable_replacement: bool
) -> Tuple[Optional[YOLO], bool]:
    """Load base YOLO model with head replacement."""
    try:
        # Load base model
        yolo_model = YOLO(model_path if model_path else 'yolo11n.pt')
        yolo_model.model.to(device)

        # Apply head replacement
        was_replaced = False
        if not disable_replacement:
            yolo_model.model, was_replaced = replace_yolo_head_if_needed(
                yolo_model.model, 'datasets/unified/data.yaml', verbose=True
            )

        # Update class information
        if hasattr(yolo_model.model, 'nc'):
            yolo_model.model.nc = config['nc']
        if hasattr(yolo_model.model, 'names'):
            yolo_model.model.names = {i: name for i, name in enumerate(config['names'])}

        return yolo_model, was_replaced

    except Exception as e:
        print(f"❌ Base model loading failed: {e}")
        return None, False


def validate_loaded_model(yolo_model: YOLO, config: Dict) -> bool:
    """
    Validate that the loaded model has correct architecture.

    Args:
        yolo_model: Loaded YOLO model
        config: Dataset configuration

    Returns:
        True if validation passes, False otherwise
    """
    print("🔍 Validating model architecture...")

    try:
        # Use our validation function from head_replacer
        validation_result = validate_yolo_model_architecture(
            yolo_model.model, config['nc'], verbose=True
        )

        if validation_result['is_valid']:
            print("✅ Model architecture validation passed")
            return True
        else:
            print("❌ Model architecture validation failed:")
            for issue in validation_result['issues']:
                print(f"   - {issue}")
            return False

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def run_sample_predictions(yolo_model: YOLO, config: Dict, max_samples: int = 5) -> None:
    """
    Run predictions on sample validation images to check model behavior.

    Args:
        yolo_model: Loaded YOLO model
        config: Dataset configuration
        max_samples: Maximum number of samples to test
    """
    print(f"\n🔬 Running sample predictions on up to {max_samples} validation images...")

    val_images_path = "datasets/unified/images/val"
    if not os.path.exists(val_images_path):
        print(f"⚠️ Validation path not found: {val_images_path}")
        return

    # Get sample images
    image_files = glob.glob(os.path.join(val_images_path, "*.jpg"))[:max_samples]
    if not image_files:
        image_files = glob.glob(os.path.join(val_images_path, "*.png"))[:max_samples]

    if not image_files:
        print("⚠️ No images found for sample prediction")
        return

    print(f"📁 Testing on {len(image_files)} sample images")

    total_detections = 0
    valid_class_detections = 0
    out_of_range_detections = 0

    for i, img_path in enumerate(image_files):
        print(f"\n🖼️ Sample {i+1}: {os.path.basename(img_path)}")

        try:
            # Run prediction with low confidence to see more results
            results = yolo_model.predict(img_path, verbose=False, conf=0.1)

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        print(f"   📦 Found {len(boxes)} detections:")

                        # Get detection details
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy().astype(int)

                        sample_detections = []
                        for j, (conf, cls_id) in enumerate(zip(confidences, class_ids)):
                            # Get class name
                            class_name = "UNKNOWN"
                            if hasattr(yolo_model.model, 'names'):
                                if isinstance(yolo_model.model.names, dict):
                                    if cls_id in yolo_model.model.names:
                                        class_name = yolo_model.model.names[cls_id]
                                    else:
                                        class_name = f"ID_{cls_id}_NO_MAPP"
                                else:
                                    try:
                                        if cls_id < len(yolo_model.model.names):
                                            class_name = yolo_model.model.names[cls_id]
                                        else:
                                            class_name = f"ID_{cls_id}_OUT_OF_RANGE"
                                    except:
                                        class_name = f"ID_{cls_id}_ERROR"

                            sample_detections.append((cls_id, class_name, conf))

                            # Check for issues
                            if cls_id >= config['nc']:
                                out_of_range_detections += 1
                                print(f"      ⚠️ {j+1}. {class_name} (conf: {conf:.3f}) - OUT OF RANGE")
                            else:
                                valid_class_detections += 1
                                print(f"      ✅ {j+1}. {class_name} (conf: {conf:.3f})")

                        total_detections += len(boxes)

                        # Check for common card confusions
                        predicted_names = [name for _, name, _ in sample_detections]
                        for pred_name in predicted_names:
                            if 'j' in pred_name.lower() and 'q' in pred_name.lower():
                                print(f"      🔍 Potential Jack/Queen similarity detected")
                            elif any(r in pred_name for r in ['10', '9', '8']) and any(f in pred_name for f in ['j', 'q', 'k']):
                                print(f"      🔍 Potential rank confusion detected")
                    else:
                        print("   ❌ No detections found")
                else:
                    print("   ❌ No bounding boxes found")
            else:
                print("   ❌ No prediction results")

        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")

    # Summary
    print(f"\n📊 Sample Prediction Summary:")
    print(f"   Total images tested: {len(image_files)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Valid class detections: {valid_class_detections}")
    print(f"   Out-of-range detections: {out_of_range_detections}")

    if out_of_range_detections > 0:
        print(f"   ⚠️ Warning: {out_of_range_detections} detections have invalid class IDs")
    else:
        print(f"   ✅ All detections have valid class IDs")


def evaluate_model_with_ultralytics(
    yolo_model: YOLO,
    config: Dict,
    device: str,
    save_plots: bool = True,
    save_json: bool = True
) -> bool:
    """
    Evaluate model using Ultralytics val method with comprehensive error handling.

    Args:
        yolo_model: Loaded YOLO model
        config: Dataset configuration
        device: Device for evaluation
        save_plots: Whether to save evaluation plots
        save_json: Whether to save JSON results

    Returns:
        True if evaluation successful, False otherwise
    """
    print(f"\n🚀 Starting comprehensive model evaluation...")

    try:
        # Prepare evaluation parameters
        eval_params = {
            'data': 'datasets/unified/data.yaml',
            'device': device,
            'split': 'val',
            'project': 'runs/evaluate',
            'name': f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'verbose': True,
        }

        # Add optional features
        if save_plots:
            eval_params['plots'] = True
        if save_json:
            eval_params['save_json'] = True
            eval_params['save_hybrid'] = True

        # Run evaluation
        print("📊 Running Ultralytics evaluation...")
        results = yolo_model.val(**eval_params)

        # Process and display results
        return _process_evaluation_results(results, config)

    except Exception as e:
        print(f"❌ Primary evaluation failed: {e}")

        # Try fallback evaluation with minimal settings
        print("🔄 Trying fallback evaluation with minimal settings...")
        try:
            fallback_results = yolo_model.val(
                data='datasets/unified/data.yaml',
                device=device,
                split='val',
                project='runs/evaluate',
                name=f'eval_fallback_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                verbose=False,
                plots=False,
                save_json=False
            )

            print("✅ Fallback evaluation completed successfully")
            return _process_evaluation_results(fallback_results, config)

        except Exception as fallback_error:
            print(f"❌ Fallback evaluation also failed: {fallback_error}")
            return False


def _process_evaluation_results(results, config: Dict) -> bool:
    """
    Process and display evaluation results with robust error handling.

    Args:
        results: Ultralytics evaluation results
        config: Dataset configuration

    Returns:
        True if processing successful, False otherwise
    """
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)

    try:
        if hasattr(results, 'box') and results.box:
            metrics = results.box

            # Helper function to safely extract metric values
            def extract_metric(metric_name: str, metric_obj: Any) -> str:
                try:
                    if hasattr(metric_obj, 'mean'):
                        return safe_format(metric_obj.mean(), metric_name)
                    elif hasattr(metric_obj, 'item'):
                        return safe_format(metric_obj.item(), metric_name)
                    elif hasattr(metric_obj, '__iter__') and not isinstance(metric_obj, str):
                        # Handle arrays - try to get a meaningful value
                        if hasattr(metric_obj, 'flatten'):
                            flat = metric_obj.flatten()
                            return safe_format(flat.mean() if len(flat) > 0 else 0, metric_name)
                        else:
                            return safe_format(list(metric_obj), metric_name)
                    else:
                        return safe_format(metric_obj, metric_name)
                except Exception as extract_error:
                    print(f"⚠️ Could not extract {metric_name}: {extract_error}")
                    return "N/A"

            # Core metrics
            map50 = extract_metric('mAP50', getattr(metrics, 'map50', None))
            map5095 = extract_metric('mAP50-95', getattr(metrics, 'map', None))

            print(f"🎯 mAP50: {map50}")
            print(f"🎯 mAP50-95: {map5095}")

            # Additional metrics
            precision = extract_metric('precision', getattr(metrics, 'precision', None))
            recall = extract_metric('recall', getattr(metrics, 'recall', None))
            f1_score = extract_metric('f1', getattr(metrics, 'f1', None))

            if precision != "N/A":
                print(f"📊 Precision: {precision}")
            if recall != "N/A":
                print(f"📊 Recall: {recall}")
            if f1_score != "N/A":
                print(f"📊 F1-score: {f1_score}")

            # Class information
            if hasattr(metrics, 'nc'):
                print(f"🏷️ Number of classes: {metrics.nc}")
                if metrics.nc != config['nc']:
                    print(f"⚠️ Expected {config['nc']} classes, found {metrics.nc}")

            # Class names sample
            if hasattr(metrics, 'names') and metrics.names:
                class_names = list(metrics.names.values())[:5]
                print(f"🏷️ Sample classes: {', '.join(class_names)}{'...' if len(metrics.names) > 5 else ''}")

                # Check for expected card classes
                expected_cards = ['10c', 'Ah', 'Js', 'Qd', 'Kh']
                found_expected = [name for name in expected_cards if name in class_names]
                if found_expected:
                    print(f"✅ Found expected card classes: {', '.join(found_expected)}")

            print(f"🔢 Results processed successfully")

        else:
            print("⚠️ No evaluation metrics found in results")
            print(f"Results type: {type(results)}")
            if hasattr(results, '__dict__'):
                print(f"Available attributes: {list(results.__dict__.keys())}")

        return True

    except Exception as results_error:
        print(f"❌ Error processing evaluation results: {results_error}")
        traceback.print_exc()
        return False


def main():
    """Main evaluation function."""
    print("🔍 French Playing Cards Model Evaluation")
    print("=" * 50)

    # Find latest model
    model_path = find_latest_model()
    if not model_path:
        return 1

    # Load dataset configuration
    config = load_dataset_config()
    if not config:
        return 1

    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")

    # Load model with head replacement
    yolo_model, was_head_replaced = load_model_with_head_replacement(
        str(model_path), config, device, force_replacement=False, disable_replacement=False
    )

    if not yolo_model:
        print("❌ Failed to load model")
        return 1

    # Validate model architecture
    if not validate_loaded_model(yolo_model, config):
        print("⚠️ Model validation failed, but continuing with evaluation")

    # Print model information
    print(f"\n📋 Model Information:")
    print(f"   Model path: {model_path}")
    print(f"   Dataset classes: {config['nc']}")
    print(f"   Head replacement: {'Yes' if was_head_replaced else 'No'}")

    # Run sample predictions
    run_sample_predictions(yolo_model, config, max_samples=5)

    # Run comprehensive evaluation
    evaluation_success = evaluate_model_with_ultralytics(yolo_model, config, device)

    if evaluation_success:
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: runs/evaluate/")
        return 0
    else:
        print("\n❌ Evaluation failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)