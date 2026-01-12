"""
Optimizer Factory for YOLOv11 Fine-Tuning

This module provides utilities to create optimizers with conservative
learning rates for different model regions during fine-tuning.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple


def create_conservative_optimizer(model, head_lr: float = 1e-3, neck_lr: float = 1e-4,
                                  backbone_lr: Optional[float] = None, weight_decay: float = 0.0005):
    """
    Create optimizer with conservative learning rates for fine-tuning.

    Args:
        model: YOLOv11 model instance
        head_lr: Learning rate for head layers (classification)
        neck_lr: Learning rate for neck layers
        backbone_lr: Learning rate for backbone layers (None = keep frozen)
        weight_decay: Weight decay for all parameters

    Returns:
        Configured AdamW optimizer
    """
    param_groups = []
    head_params = []
    neck_params = []
    backbone_params = []

    # Get the internal model to determine layer structure
    internal_model = model.model if hasattr(model, 'model') else model
    total_layers = len(list(internal_model.children()))
    detect_layer_idx = total_layers - 1  # Last layer is usually Detect

    # Separate parameters by region
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is in the detection layer (last layer)
        if name.startswith(f'model.{detect_layer_idx}.') or 'detect' in name.lower():
            head_params.append(param)

        # Neck layers are typically upsampling, concat, and C3k2 after SPPF
        elif any(key in name.lower() for key in ['upsample', 'concat']) or \
             any(name.startswith(f'model.{i}.') for i in range(11, 23)):  # Neck region
            neck_params.append(param)

        # Backbone layers are the initial layers
        elif any(name.startswith(f'model.{i}.') for i in range(0, 11)):  # Backbone region
            if backbone_lr is not None:  # Only include if we want to train backbone
                backbone_params.append(param)

        else:
            # Default to backbone if training it
            if backbone_lr is not None:
                backbone_params.append(param)

    # Build parameter groups
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'head'
        })

    if neck_params:
        param_groups.append({
            'params': neck_params,
            'lr': neck_lr,
            'name': 'neck'
        })

    if backbone_params and backbone_lr is not None:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })

    # Create optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # Print optimizer information
    print_optimizer_info(param_groups)

    return optimizer


def print_optimizer_info(param_groups: List[Dict]):
    """
    Print information about the optimizer parameter groups.

    Args:
        param_groups: List of parameter group dictionaries
    """
    print("\n" + "="*50)
    print("OPTIMIZER CONFIGURATION")
    print("="*50)

    total_params = sum(len(group['params']) for group in param_groups)

    print(f"\n📊 PARAMETER GROUPS ({total_params} groups):")
    for group in param_groups:
        group_name = group.get('name', 'unknown')
        lr = group['lr']
        param_count = sum(p.numel() for p in group['params'])
        print(f"  {group_name.capitalize()}:")
        print(f"    Learning Rate: {lr}")
        print(f"    Parameters: {len(group['params'])} tensors ({param_count:,} elements)")

    print("="*50)


def create_differential_optimizer(model, lr_config: Dict[str, float], weight_decay: float = 0.0005):
    """
    Create optimizer with custom learning rate configuration.

    Args:
        model: YOLOv11 model instance
        lr_config: Dictionary with learning rates for different regions
        weight_decay: Weight decay for all parameters

    Returns:
        Configured optimizer
    """
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_name = name.lower()
        lr = lr_config.get('default', 1e-3)

        # Determine learning rate based on layer type
        if any(key in layer_name for key in ['head', 'detect', 'cls']):
            lr = lr_config.get('head', lr_config.get('default', 1e-3))
        elif any(key in layer_name for key in ['neck', 'fpn', 'pan']):
            lr = lr_config.get('neck', lr_config.get('default', 1e-4))
        elif any(key in layer_name for key in ['backbone', 'stem', 'dark']):
            lr = lr_config.get('backbone', lr_config.get('default', 1e-5))

        param_groups.append({
            'params': [param],
            'lr': lr,
            'name': name
        })

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def create_phase_optimizer(model, phase: str, head_lr: float = 1e-3, neck_lr: float = 1e-4,
                         weight_decay: float = 0.0005):
    """
    Create optimizer optimized for specific training phase.

    Args:
        model: YOLOv11 model instance
        phase: Training phase ('head-only', 'head+neck', 'full')
        head_lr: Learning rate for head layers
        neck_lr: Learning rate for neck layers
        weight_decay: Weight decay for all parameters

    Returns:
        Configured optimizer
    """
    if phase == 'head-only':
        # Only head layers should be trainable
        return create_conservative_optimizer(
            model, head_lr=head_lr, neck_lr=0, backbone_lr=None, weight_decay=weight_decay
        )

    elif phase == 'head+neck':
        # Head and neck layers trainable
        return create_conservative_optimizer(
            model, head_lr=head_lr, neck_lr=neck_lr, backbone_lr=None, weight_decay=weight_decay
        )

    elif phase == 'full':
        # All layers trainable with progressive learning rates
        return create_conservative_optimizer(
            model, head_lr=head_lr, neck_lr=neck_lr, backbone_lr=head_lr * 0.1, weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")


def create_cosine_scheduler(optimizer, T_max: int, eta_min: float = 1e-6, warmup_epochs: int = 0):
    """
    Create cosine annealing learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        warmup_epochs: Number of warmup epochs

    Returns:
        Configured scheduler
    """
    if warmup_epochs > 0:
        # Combined warmup + cosine scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min
                )
            ],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    return scheduler


def get_optimizer_lr(optimizer) -> Dict[str, float]:
    """
    Get current learning rates for different parameter groups.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Dictionary mapping group names to learning rates
    """
    lr_dict = {}
    for i, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{i}')
        lr_dict[group_name] = group['lr']

    return lr_dict


def count_trainable_parameters(model) -> Dict[str, int]:
    """
    Count trainable parameters by region.

    Args:
        model: YOLOv11 model instance

    Returns:
        Dictionary with parameter counts by region
    """
    head_params = 0
    neck_params = 0
    backbone_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_name = name.lower()

        if any(key in layer_name for key in ['head', 'detect', 'cls']):
            head_params += param.numel()
        elif any(key in layer_name for key in ['neck', 'fpn', 'pan']):
            neck_params += param.numel()
        elif any(key in layer_name for key in ['backbone', 'stem', 'dark']):
            backbone_params += param.numel()

    return {
        'head': head_params,
        'neck': neck_params,
        'backbone': backbone_params,
        'total': head_params + neck_params + backbone_params
    }


def validate_optimizer_config(model, head_lr: float, neck_lr: float) -> bool:
    """
    Validate optimizer configuration.

    Args:
        model: YOLOv11 model instance
        head_lr: Head learning rate
        neck_lr: Neck learning rate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check learning rates are positive
        if head_lr <= 0 or neck_lr <= 0:
            print("Error: Learning rates must be positive.")
            return False

        # Check that model has trainable parameters
        param_counts = count_trainable_parameters(model)
        if param_counts['total'] == 0:
            print("Error: No trainable parameters found in model.")
            return False

        # Check that head learning rate is reasonable (not too high)
        if head_lr > 1e-2:
            print("Warning: Head learning rate is very high (>1e-2).")

        if neck_lr > head_lr:
            print("Warning: Neck learning rate is higher than head learning rate.")

        return True

    except Exception as e:
        print(f"Error validating optimizer config: {e}")
        return False


if __name__ == "__main__":
    # Test the optimizer factory
    try:
        from ultralytics import YOLO

        print("Testing YOLOv11 Optimizer Factory...")

        # Load a YOLOv11 model
        model_wrapper = YOLO('yolo11n.pt')
        model = model_wrapper.model

        # Test conservative optimizer creation
        print(f"\n{'='*60}")
        print("Testing conservative optimizer...")
        optimizer = create_conservative_optimizer(model, head_lr=1e-3, neck_lr=1e-4)

        # Test phase-specific optimizers
        for phase in ['head-only', 'head+neck', 'full']:
            print(f"\n{'='*60}")
            print(f"Testing {phase} optimizer...")
            phase_optimizer = create_phase_optimizer(model, phase)

        # Test parameter counting
        print(f"\n{'='*60}")
        print("Testing parameter counting...")
        param_counts = count_trainable_parameters(model)
        print(f"Head parameters: {param_counts['head']:,}")
        print(f"Neck parameters: {param_counts['neck']:,}")
        print(f"Backbone parameters: {param_counts['backbone']:,}")
        print(f"Total trainable: {param_counts['total']:,}")

    except ImportError:
        print("YOLOv11 not available. Install ultralytics package to test.")
    except Exception as e:
        print(f"Error during testing: {e}")