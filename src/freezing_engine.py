"""
YOLOv11 Layer Freezing Engine

This module provides utilities to freeze and unfreeze different regions
of YOLOv11 models for fine-tuning strategies.
"""

import torch
from typing import Dict, List


def freeze_model_regions(model, freeze_backbone=True, freeze_neck=True, freeze_head=False, verbose=False):
    """
    Freeze specific model regions for fine-tuning.

    Args:
        model: YOLOv11 model instance
        freeze_backbone: Whether to freeze backbone layers
        freeze_neck: Whether to freeze neck layers
        freeze_head: Whether to freeze head layers
        verbose: Whether to print detailed freezing information

    Returns:
        Dict with freezing statistics
    """
    freezing_stats = {
        'backbone': {'frozen': 0, 'trainable': 0},
        'neck': {'frozen': 0, 'trainable': 0},
        'head': {'frozen': 0, 'trainable': 0},
        'other': {'frozen': 0, 'trainable': 0}
    }

    total_frozen = 0
    total_trainable = 0

    # Get the internal model (YOLO wrapper structure)
    internal_model = model.model if hasattr(model, 'model') else model

    # Count layers in the internal model to determine regions
    total_layers = len(list(internal_model.children()))
    detect_layer_idx = total_layers - 1  # Last layer is usually Detect

    for name, param in model.named_parameters():
        param_count = param.numel()
        layer_name = name.lower()

        # Determine region based on layer index and naming
        should_freeze = False
        region = 'other'

        # Check if this is in the detection layer (last layer)
        if name.startswith(f'model.{detect_layer_idx}.') or 'detect' in layer_name:
            region = 'head'
            should_freeze = freeze_head

        # Neck layers are typically upsampling, concat, and C3k2 after SPPF
        elif any(key in layer_name for key in ['upsample', 'concat']) or \
             any(name.startswith(f'model.{i}.') for i in range(11, 23)):  # Neck region
            region = 'neck'
            should_freeze = freeze_neck

        # Backbone layers are the initial layers
        elif any(name.startswith(f'model.{i}.') for i in range(0, 11)):  # Backbone region
            region = 'backbone'
            should_freeze = freeze_backbone

        else:
            # Default to backbone setting
            region = 'backbone'
            should_freeze = freeze_backbone

        # Apply freezing
        param.requires_grad = not should_freeze

        # Update statistics
        if should_freeze:
            freezing_stats[region]['frozen'] += param_count
            total_frozen += param_count
        else:
            freezing_stats[region]['trainable'] += param_count
            total_trainable += param_count

    if verbose:
        print_freezing_report(freezing_stats, freeze_backbone, freeze_neck, freeze_head)

    return freezing_stats


def print_freezing_report(stats: Dict, freeze_backbone: bool, freeze_neck: bool, freeze_head: bool):
    """
    Print a detailed freezing report.

    Args:
        stats: Freezing statistics dictionary
        freeze_backbone: Whether backbone was frozen
        freeze_neck: Whether neck was frozen
        freeze_head: Whether head was frozen
    """
    print("\n" + "="*50)
    print("LAYER FREEZING REPORT")
    print("="*50)

    print(f"\n🔒 FREEZING CONFIGURATION:")
    print(f"  Backbone: {'FROZEN' if freeze_backbone else 'TRAINABLE'}")
    print(f"  Neck: {'FROZEN' if freeze_neck else 'TRAINABLE'}")
    print(f"  Head: {'FROZEN' if freeze_head else 'TRAINABLE'}")

    print(f"\n📊 PARAMETER COUNTS BY REGION:")
    for region, counts in stats.items():
        if region != 'other' or (counts['frozen'] > 0 or counts['trainable'] > 0):
            total = counts['frozen'] + counts['trainable']
            if total > 0:
                frozen_pct = (counts['frozen'] / total) * 100 if total > 0 else 0
                trainable_pct = (counts['trainable'] / total) * 100 if total > 0 else 0
                print(f"  {region.capitalize()}:")
                print(f"    Frozen: {counts['frozen']:,} ({frozen_pct:.1f}%)")
                print(f"    Trainable: {counts['trainable']:,} ({trainable_pct:.1f}%)")

    total_frozen = sum(stats[region]['frozen'] for region in stats)
    total_trainable = sum(stats[region]['trainable'] for region in stats)
    total_all = total_frozen + total_trainable

    if total_all > 0:
        frozen_overall = (total_frozen / total_all) * 100
        print(f"\n📈 OVERALL:")
        print(f"  Total Parameters: {total_all:,}")
        print(f"  Frozen: {total_frozen:,} ({frozen_overall:.1f}%)")
        print(f"  Trainable: {total_trainable:,} ({100-frozen_overall:.1f}%)")

    print("="*50)


def apply_phase_unfreezing(model, phase: str, verbose=True):
    """
    Apply freezing based on training phase.

    Args:
        model: YOLOv11 model instance
        phase: Training phase ('head-only', 'head+neck', or 'full')
        verbose: Whether to print information

    Returns:
        Freezing statistics
    """
    if phase == 'head-only':
        # Freeze backbone and neck, train only head
        stats = freeze_model_regions(
            model,
            freeze_backbone=True,
            freeze_neck=True,
            freeze_head=False,
            verbose=verbose
        )
        if verbose:
            print("\n🎯 PHASE: Head-only training")
            print("    Backbone: FROZEN")
            print("    Neck: FROZEN")
            print("    Head: TRAINABLE")

    elif phase == 'head+neck':
        # Freeze only backbone, train neck + head
        stats = freeze_model_regions(
            model,
            freeze_backbone=True,
            freeze_neck=False,
            freeze_head=False,
            verbose=verbose
        )
        if verbose:
            print("\n🎯 PHASE: Head + Neck training")
            print("    Backbone: FROZEN")
            print("    Neck: TRAINABLE")
            print("    Head: TRAINABLE")

    elif phase == 'full':
        # Train everything
        stats = freeze_model_regions(
            model,
            freeze_backbone=False,
            freeze_neck=False,
            freeze_head=False,
            verbose=verbose
        )
        if verbose:
            print("\n🎯 PHASE: Full model training")
            print("    Backbone: TRAINABLE")
            print("    Neck: TRAINABLE")
            print("    Head: TRAINABLE")

    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'head-only', 'head+neck', or 'full'")

    return stats


def get_model_info(model) -> Dict:
    """
    Get comprehensive information about the current model state.

    Args:
        model: YOLOv11 model instance

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Count parameters by region
    backbone_params = 0
    neck_params = 0
    head_params = 0

    backbone_trainable = 0
    neck_trainable = 0
    head_trainable = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        layer_name = name.lower()

        if any(key in layer_name for key in ['backbone', 'stem', 'dark']):
            backbone_params += param_count
            if param.requires_grad:
                backbone_trainable += param_count

        elif any(key in layer_name for key in ['neck', 'fpn', 'pan']):
            neck_params += param_count
            if param.requires_grad:
                neck_trainable += param_count

        elif any(key in layer_name for key in ['head', 'detect', 'cls']):
            head_params += param_count
            if param.requires_grad:
                head_trainable += param_count

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'regions': {
            'backbone': {
                'total': backbone_params,
                'trainable': backbone_trainable,
                'frozen': backbone_params - backbone_trainable
            },
            'neck': {
                'total': neck_params,
                'trainable': neck_trainable,
                'frozen': neck_params - neck_trainable
            },
            'head': {
                'total': head_params,
                'trainable': head_trainable,
                'frozen': head_params - head_trainable
            }
        }
    }


def validate_freezing_config(model, config: Dict) -> bool:
    """
    Validate that a freezing configuration is valid.

    Args:
        model: YOLOv11 model instance
        config: Freezing configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check that at least one region is trainable
        if all([
            config.get('freeze_backbone', True),
            config.get('freeze_neck', True),
            config.get('freeze_head', True)
        ]):
            print("Warning: All regions are frozen. No parameters will be trained.")
            return False

        # Check that the model has the expected structure
        model_info = get_model_info(model)
        if model_info['trainable_parameters'] == 0:
            print("Error: No trainable parameters found.")
            return False

        return True

    except Exception as e:
        print(f"Error validating freezing config: {e}")
        return False


if __name__ == "__main__":
    # Test the freezing engine
    try:
        from ultralytics import YOLO

        print("Testing YOLOv11 Freezing Engine...")

        # Load a YOLOv11 model
        model_wrapper = YOLO('yolo11n.pt')
        model = model_wrapper.model

        print(f"\nInitial model state:")
        initial_info = get_model_info(model)
        print(f"  Total parameters: {initial_info['total_parameters']:,}")
        print(f"  Trainable: {initial_info['trainable_parameters']:,}")

        # Test head-only freezing
        print(f"\n{'='*60}")
        print("Testing head-only freezing...")
        stats = apply_phase_unfreezing(model, 'head-only', verbose=True)

        # Test head+neck freezing
        print(f"\n{'='*60}")
        print("Testing head+neck freezing...")
        stats = apply_phase_unfreezing(model, 'head+neck', verbose=True)

        # Test full training (reset)
        print(f"\n{'='*60}")
        print("Testing full model training...")
        stats = apply_phase_unfreezing(model, 'full', verbose=True)

    except ImportError:
        print("YOLOv11 not available. Install ultralytics package to test.")
    except Exception as e:
        print(f"Error during testing: {e}")