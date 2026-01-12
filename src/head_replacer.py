"""
YOLO Head Replacement Module

This module provides automatic detection head replacement for YOLO models
when the number of classes in the pretrained model doesn't match the dataset.
"""

import torch
import torch.nn as nn
import yaml
import copy
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from ultralytics.nn.modules.head import Detect


class YOLOHeadReplacer:
    """
    Automatic YOLO classification head replacement for fine-tuning on different datasets.

    This class detects when a pretrained YOLO model has a different number of classes
    than the target dataset and automatically replaces the classification head with
    the correct output dimensions while preserving compatible weights.
    """

    def __init__(self, model, dataset_config_path: str, verbose: bool = True):
        """
        Initialize the YOLO Head Replacer.

        Args:
            model: The YOLO model to potentially modify
            dataset_config_path: Path to dataset YAML configuration file
            verbose: Whether to print detailed information
        """
        self.model = model
        self.verbose = verbose
        self.dataset_config = self._load_dataset_config(dataset_config_path)
        self.detect_layer = self._find_detect_layer()

        if self.detect_layer:
            self.original_nc = self.detect_layer.nc
            self.target_nc = self.dataset_config['nc']
        else:
            raise ValueError("No Detect layer found in YOLO model")

    def _load_dataset_config(self, config_path: str) -> Dict:
        """Load dataset configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            required_keys = ['nc', 'names']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in dataset config")

            return config
        except Exception as e:
            raise ValueError(f"Failed to load dataset config from {config_path}: {e}")

    def _find_detect_layer(self) -> Optional[Detect]:
        """Find the Detect layer in YOLO model"""
        detect_layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, Detect):
                detect_layers.append((name, module))

        if not detect_layers:
            return None

        # Return the last (typically final) Detect layer
        if len(detect_layers) == 1:
            if self.verbose:
                print(f"Found single Detect layer: {detect_layers[0][0]}")
            return detect_layers[0][1]
        else:
            # Multiple Detect layers found, use the last one
            last_layer = detect_layers[-1]
            if self.verbose:
                print(f"Found {len(detect_layers)} Detect layers, using: {last_layer[0]}")
            return last_layer[1]

    def needs_replacement(self) -> bool:
        """Check if head replacement is needed"""
        return self.original_nc != self.target_nc

    def analyze_head_mismatch(self) -> Dict:
        """Analyze current vs required class configuration for YOLOv11"""
        current_outputs = self.detect_layer.no  # Number of outputs
        # YOLOv11 formula: no = nc + reg_max * 4
        target_outputs = self.target_nc + self.detect_layer.reg_max * 4

        return {
            'original_nc': self.original_nc,
            'target_nc': self.target_nc,
            'current_outputs': current_outputs,
            'target_outputs': target_outputs,
            'difference': target_outputs - current_outputs,
            'num_scales': len(self.detect_layer.stride),
            'reg_max': self.detect_layer.reg_max,
            'regression_channels': self.detect_layer.reg_max * 4,  # DFL channels for bbox
            'classification_channels_original': self.original_nc,
            'classification_channels_target': self.target_nc,
            'channels_per_scale_original': self.original_nc + self.detect_layer.reg_max * 4,
            'channels_per_scale_target': self.target_nc + self.detect_layer.reg_max * 4
        }

    def replace_classification_head(self) -> nn.Module:
        """
        Replace final conv layers with correct output channels.

        Returns:
            The modified model with replaced classification head
        """
        if not self.needs_replacement():
            if self.verbose:
                print("✓ Model classes match dataset, no head replacement needed")
            return self.model

        analysis = self.analyze_head_mismatch()

        if self.verbose:
            print(f"🔄 Replacing YOLO classification head:")
            print(f"   Original: {analysis['original_nc']} classes → {analysis['current_outputs']} outputs")
            print(f"   Target:   {analysis['target_nc']} classes → {analysis['target_outputs']} outputs")
            print(f"   Change:   {analysis['difference']} output channels")

        # Store original head for weight transfer
        original_head = copy.deepcopy(self.detect_layer)

        # Create new Detect layer with correct number of classes
        new_detect = self._create_new_detect_layer()

        # Replace the detect layer in the model
        self._replace_detect_layer_in_model(new_detect)

        # Transfer compatible weights
        self._transfer_compatible_weights(original_head, new_detect)

        # Validate the replacement
        if self._validate_replacement():
            if self.verbose:
                print("✓ Classification head replacement successful")
        else:
            raise RuntimeError("Head replacement validation failed")

        return self.model

    def _create_new_detect_layer(self) -> Detect:
        """Create a new Detect layer with the correct number of classes"""
        # Copy the original detect layer configuration
        new_detect = copy.deepcopy(self.detect_layer)
        new_detect.nc = self.target_nc

        # YOLOv11 uses: no = nc + reg_max * 4
        new_detect.no = self.target_nc + new_detect.reg_max * 4

        # Update both cv2 (box regression) and cv3 (classification) branches
        # cv2: handles bbox regression (reg_max * 4 channels) - DON'T CHANGE
        # cv3: handles classification (nc channels) - NEEDS UPDATE

        if self.verbose:
            print(f"   📊 YOLOv11 Architecture:")
            print(f"      reg_max * 4 = {new_detect.reg_max} * 4 = {new_detect.reg_max * 4} (bbox channels)")
            print(f"      nc = {self.target_nc} (classification channels)")
            print(f"      Total no = {new_detect.no}")

        # Update cv3 classification branch (this handles class predictions)
        for i in range(len(new_detect.cv3)):
            scale_layer = new_detect.cv3[i]
            original_conv = scale_layer[2]  # The final Conv2d layer

            # Create new conv layer with correct number of class channels
            in_channels = original_conv.in_channels
            out_channels = self.target_nc

            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None
            )

            # Replace the conv layer in the scale
            scale_layer[2] = new_conv

            if self.verbose:
                print(f"      Scale {i+1} cv3: {original_conv.out_channels} → {out_channels} class channels")

        # cv2 box regression branch remains unchanged (reg_max * 4 channels)
        if self.verbose:
            for i, scale_layer in enumerate(new_detect.cv2):
                conv = scale_layer[2]
                print(f"      Scale {i+1} cv2: {conv.out_channels} bbox channels (unchanged)")

        return new_detect

    def _replace_detect_layer_in_model(self, new_detect: Detect) -> None:
        """Replace the Detect layer in the model with the new one"""
        # Find the detect layer path in the model
        for name, module in self.model.named_modules():
            if module is self.detect_layer:
                # Navigate to parent and replace the module
                parent_path = name.rsplit('.', 1)
                if len(parent_path) == 1:
                    # Top-level module
                    setattr(self.model, parent_path[0], new_detect)
                else:
                    # Nested module
                    parent = self.model.get_submodule(parent_path[0])
                    setattr(parent, parent_path[1], new_detect)
                break

        # Update reference to the new detect layer
        self.detect_layer = new_detect

        # **CRITICAL FIX**: Update the main model's nc attribute to match
        # The YOLO wrapper has its own model.nc attribute that needs to be updated
        if hasattr(self.model, 'nc'):
            self.model.nc = self.target_nc

        # Also update model.names if needed
        if hasattr(self.model, 'names') and self.dataset_config.get('names'):
            # Ensure names list has correct length
            names = self.dataset_config['names']
            if len(names) == self.target_nc:
                self.model.names = {i: name for i, name in enumerate(names)}

    def _transfer_compatible_weights(self, old_head: Detect, new_head: Detect) -> None:
        """
        Transfer weights from old head to new head for YOLOv11:
        - cv2 (bbox regression): Transfer all weights (unchanged architecture)
        - cv3 (classification): Transfer compatible class weights, reinitialize others
        """
        with torch.no_grad():
            # Transfer cv2 bbox regression weights (unchanged architecture)
            if self.verbose:
                print(f"   🔄 Transferring bbox regression weights (cv2):")

            for i, (old_scale, new_scale) in enumerate(zip(old_head.cv2, new_head.cv2)):
                old_conv = old_scale[2]  # Final Conv2d in each scale
                new_conv = new_scale[2]

                # cv2 architecture unchanged, can transfer all weights
                if old_conv.out_channels == new_conv.out_channels:
                    new_conv.weight.copy_(old_conv.weight)
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)

                    if self.verbose:
                        print(f"      Scale {i+1}: ✅ Transferred {old_conv.out_channels} bbox weights")
                else:
                    if self.verbose:
                        print(f"      Scale {i+1}: ⚠️ Bbox channel mismatch, reinitializing")

            # Transfer cv3 classification weights (architecture changed)
            if self.verbose:
                print(f"   🔄 Transferring classification weights (cv3):")

            for i, (old_scale, new_scale) in enumerate(zip(old_head.cv3, new_head.cv3)):
                old_conv = old_scale[2]  # Final Conv2d in each scale
                new_conv = new_scale[2]

                # Determine how many class weights to transfer
                min_classes = min(self.original_nc, self.target_nc)

                if self.verbose:
                    print(f"      Scale {i+1}: {old_conv.out_channels} → {new_conv.out_channels} class channels")
                    print(f"               Transferring {min_classes} compatible weights")

                # Transfer compatible class weights
                if min_classes > 0:
                    new_conv.weight[:min_classes] = old_conv.weight[:min_classes]
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias[:min_classes] = old_conv.bias[:min_classes]

                # Initialize new class weights (if target has more classes)
                if self.target_nc > self.original_nc:
                    new_classes_start = self.original_nc
                    nn.init.normal_(new_conv.weight[new_classes_start:], std=0.01)
                    if new_conv.bias is not None:
                        nn.init.constant_(new_conv.bias[new_classes_start:], 0.)

                    if self.verbose:
                        print(f"               Initialized {self.target_nc - self.original_nc} new class weights")

    def _validate_replacement(self) -> bool:
        """Validate that replacement was successful for YOLOv11"""
        try:
            # Check class count
            if self.detect_layer.nc != self.target_nc:
                if self.verbose:
                    print(f"❌ Class count mismatch: {self.detect_layer.nc} != {self.target_nc}")
                return False

            # Check output channels for YOLOv11: no = nc + reg_max * 4
            expected_outputs = self.target_nc + self.detect_layer.reg_max * 4
            if self.detect_layer.no != expected_outputs:
                if self.verbose:
                    print(f"❌ Output channels mismatch: {self.detect_layer.no} != {expected_outputs}")
                    print(f"   Expected formula: nc ({self.target_nc}) + reg_max*4 ({self.detect_layer.reg_max * 4})")
                return False

            # Check cv2 (bbox regression) channels - should be reg_max * 4
            expected_bbox_channels = self.detect_layer.reg_max * 4
            for i, scale_layer in enumerate(self.detect_layer.cv2):
                conv = scale_layer[2]
                if conv.out_channels != expected_bbox_channels:
                    if self.verbose:
                        print(f"❌ Scale {i+1} cv2 (bbox) outputs mismatch: {conv.out_channels} != {expected_bbox_channels}")
                    return False

            # Check cv3 (classification) channels - should be target_nc
            for i, scale_layer in enumerate(self.detect_layer.cv3):
                conv = scale_layer[2]
                if conv.out_channels != self.target_nc:
                    if self.verbose:
                        print(f"❌ Scale {i+1} cv3 (classification) outputs mismatch: {conv.out_channels} != {self.target_nc}")
                    return False

            if self.verbose:
                print(f"✓ YOLOv11 Validation passed:")
                print(f"   Classes: {self.target_nc}")
                print(f"   Total outputs: {self.detect_layer.no}")
                print(f"   Bbox channels per scale: {expected_bbox_channels}")
                print(f"   Classification channels per scale: {self.target_nc}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Validation error: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get comprehensive information about the model configuration"""
        return {
            'dataset_classes': self.target_nc,
            'dataset_names': self.dataset_config['names'][:10],  # First 10 names
            'model_classes': self.detect_layer.nc,
            'model_outputs': self.detect_layer.no,
            'num_scales': len(self.detect_layer.stride),
            'stride_values': list(self.detect_layer.stride),
            'needs_replacement': self.needs_replacement(),
            'replacement_analysis': self.analyze_head_mismatch() if self.needs_replacement() else None
        }


def replace_yolo_head_if_needed(model, dataset_config_path: str, verbose: bool = True) -> Tuple[nn.Module, bool]:
    """
    Convenience function to replace YOLO head if needed.

    Args:
        model: The YOLO model to potentially modify
        dataset_config_path: Path to dataset YAML configuration file
        verbose: Whether to print detailed information

    Returns:
        Tuple of (modified_model, was_replaced)
    """
    try:
        replacer = YOLOHeadReplacer(model, dataset_config_path, verbose)

        if replacer.needs_replacement():
            modified_model = replacer.replace_classification_head()
            return modified_model, True
        else:
            return model, False

    except Exception as e:
        if verbose:
            print(f"⚠️ Head replacement failed: {e}")
            print("   Continuing with original model (may cause class mismatch issues)")
        return model, False


def validate_yolo_model_architecture(model, expected_classes: int, verbose: bool = True) -> Dict:
    """
    Validate that a YOLO model has the correct architecture for expected classes.

    Args:
        model: The YOLO model to validate
        expected_classes: Expected number of classes
        verbose: Whether to print detailed information

    Returns:
        Validation results dictionary
    """
    validation_result = {
        'is_valid': False,
        'model_classes': None,
        'expected_classes': expected_classes,
        'output_channels': None,
        'expected_output_channels': None,
        'detect_layers_found': 0,
        'issues': []
    }

    # Find detect layers
    detect_layers = []
    for name, module in model.named_modules():
        if isinstance(module, Detect):
            detect_layers.append((name, module))

    validation_result['detect_layers_found'] = len(detect_layers)

    if not detect_layers:
        validation_result['issues'].append("No Detect layer found in model")
        return validation_result

    # Use the first detect layer for validation
    detect_layer = detect_layers[0][1]
    validation_result['model_classes'] = detect_layer.nc
    validation_result['output_channels'] = detect_layer.no
    # YOLOv11 formula: no = nc + reg_max * 4
    validation_result['expected_output_channels'] = expected_classes + detect_layer.reg_max * 4

    # Check class count
    if detect_layer.nc != expected_classes:
        validation_result['issues'].append(
            f"Class count mismatch: model has {detect_layer.nc}, expected {expected_classes}"
        )

    # Check output channels
    if detect_layer.no != validation_result['expected_output_channels']:
        validation_result['issues'].append(
            f"Output channels mismatch: model has {detect_layer.no}, expected {validation_result['expected_output_channels']}"
        )

    # Overall validity
    validation_result['is_valid'] = len(validation_result['issues']) == 0

    if verbose:
        if validation_result['is_valid']:
            print(f"✓ Model architecture is valid for {expected_classes} classes")
        else:
            print(f"❌ Model architecture validation failed:")
            for issue in validation_result['issues']:
                print(f"   - {issue}")

    return validation_result