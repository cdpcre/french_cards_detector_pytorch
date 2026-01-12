"""
Unit tests for YOLO Head Replacement System

This module tests the core functionality of the YOLO head replacement system,
including class mismatch detection, weight transfer, and validation.
"""

import pytest
import torch
import tempfile
import yaml
import os
from pathlib import Path

# Add src directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from ultralytics import YOLO
    from head_replacer import YOLOHeadReplacer, replace_yolo_head_if_needed, validate_yolo_model_architecture
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available - some tests will be skipped")


@pytest.fixture
def yolo_model():
    """Load YOLOv11-n model for testing."""
    if not ULTRALYTICS_AVAILABLE:
        pytest.skip("Ultralytics not available")

    model_wrapper = YOLO('yolo11n.pt')
    return model_wrapper.model


@pytest.fixture
def dataset_config():
    """Create temporary dataset config for testing."""
    config = {
        'nc': 53,
        'names': [f'class_{i}' for i in range(53)]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


class TestYOLOHeadReplacer:
    """Test the core YOLOHeadReplacer class."""

    def test_init_with_valid_model(self, yolo_model, dataset_config):
        """Test initialization with valid model and config."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)

        assert replacer.original_nc == 80  # YOLOv11-n has 80 COCO classes
        assert replacer.target_nc == 53
        assert replacer.needs_replacement() is True

    def test_init_with_compatible_classes(self, yolo_model, dataset_config):
        """Test initialization when model already has correct classes."""
        # Create config with 80 classes to match the model
        config_80 = {
            'nc': 80,
            'names': [f'class_{i}' for i in range(80)]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_80, f)
            config_path = f.name

        try:
            replacer = YOLOHeadReplacer(yolo_model, config_path, verbose=False)

            assert replacer.original_nc == 80
            assert replacer.target_nc == 80
            assert replacer.needs_replacement() is False
        finally:
            os.unlink(config_path)

    def test_analyze_head_mismatch(self, yolo_model, dataset_config):
        """Test head mismatch analysis."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)
        analysis = replacer.analyze_head_mismatch()

        assert analysis['original_nc'] == 80
        assert analysis['target_nc'] == 53
        assert analysis['current_outputs'] > analysis['target_outputs']
        assert analysis['difference'] < 0  # Should be negative (reducing outputs)
        assert analysis['num_scales'] == 3  # YOLOv11 typically has 3 scales

    def test_head_replacement(self, yolo_model, dataset_config):
        """Test actual head replacement."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)

        # Store original values
        original_nc = yolo_model.nc
        original_no = yolo_model.no if hasattr(yolo_model, 'no') else None

        # Perform replacement
        modified_model = replacer.replace_classification_head()

        # Verify changes
        assert modified_model.nc == 53
        if hasattr(modified_model, 'no'):
            expected_no = (53 + 5) * len(replacer.detect_layer.stride)
            assert modified_model.no == expected_no

    def test_weight_transfer(self, yolo_model, dataset_config):
        """Test weight transfer logic."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)

        # Get original cv2 (bbox) and cv3 (classification) weights before replacement
        original_cv2_weights = {}
        original_cv3_weights = {}
        for i, scale_layer in enumerate(replacer.detect_layer.cv2):
            conv = scale_layer[2]
            original_cv2_weights[i] = conv.weight.clone()
        for i, scale_layer in enumerate(replacer.detect_layer.cv3):
            conv = scale_layer[2]
            original_cv3_weights[i] = conv.weight.clone()

        # Replace head
        modified_model = replacer.replace_classification_head()

        # Check cv2 (bbox regression) weights are preserved unchanged
        for i, scale_layer in enumerate(replacer.detect_layer.cv2):
            conv = scale_layer[2]
            assert torch.allclose(original_cv2_weights[i], conv.weight, atol=1e-6), f"Scale {i} cv2 bbox weights should be preserved"

        # Check cv3 (classification) weights: first min(original_nc, target_nc) classes should be transferred
        min_classes = min(replacer.original_nc, replacer.target_nc)  # min(80, 53) = 53
        for i, scale_layer in enumerate(replacer.detect_layer.cv3):
            conv = scale_layer[2]
            # First min_classes weights should be transferred from original
            original_transferred = original_cv3_weights[i][:min_classes]
            new_transferred = conv.weight[:min_classes]
            assert torch.allclose(original_transferred, new_transferred, atol=1e-6), f"Scale {i} first {min_classes} class weights should be transferred"

    def test_validation(self, yolo_model, dataset_config):
        """Test replacement validation."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)

        # Before replacement, validation should fail
        assert not replacer._validate_replacement()

        # After replacement, validation should pass
        modified_model = replacer.replace_classification_head()
        assert replacer._validate_replacement()

    def test_get_model_info(self, yolo_model, dataset_config):
        """Test model information extraction."""
        replacer = YOLOHeadReplacer(yolo_model, dataset_config, verbose=False)
        info = replacer.get_model_info()

        assert info['dataset_classes'] == 53
        assert info['model_classes'] == 80
        assert info['needs_replacement'] is True
        assert 'replacement_analysis' in info
        assert len(info['dataset_names']) == 10  # First 10 names

    def test_invalid_dataset_config(self, yolo_model):
        """Test handling of invalid dataset config."""
        # Config without 'nc' key
        invalid_config = {'names': ['class1', 'class2']}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required key 'nc'"):
                YOLOHeadReplacer(yolo_model, config_path, verbose=False)
        finally:
            os.unlink(config_path)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_replace_yolo_head_if_needed(self, yolo_model, dataset_config):
        """Test the convenience function for head replacement."""
        # Should replace since 80 != 53
        modified_model, was_replaced = replace_yolo_head_if_needed(
            yolo_model, dataset_config, verbose=False
        )

        assert was_replaced is True
        assert modified_model.nc == 53

    def test_replace_yolo_head_if_not_needed(self, yolo_model):
        """Test convenience function when replacement not needed."""
        # Create config with 80 classes
        config_80 = {'nc': 80, 'names': [f'class_{i}' for i in range(80)]}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_80, f)
            config_path = f.name

        try:
            modified_model, was_replaced = replace_yolo_head_if_needed(
                yolo_model, config_path, verbose=False
            )

            assert was_replaced is False
            assert modified_model is yolo_model  # Should return same model
        finally:
            os.unlink(config_path)

    def test_validate_yolo_model_architecture(self, yolo_model):
        """Test model architecture validation."""
        # Test with wrong number of classes
        validation_result = validate_yolo_model_architecture(yolo_model, 53, verbose=False)

        assert validation_result['is_valid'] is False
        assert validation_result['model_classes'] == 80
        assert validation_result['expected_classes'] == 53
        assert len(validation_result['issues']) > 0
        assert validation_result['detect_layers_found'] == 1

    def test_validate_yolo_model_architecture_compatible(self, yolo_model):
        """Test validation with compatible model."""
        validation_result = validate_yolo_model_architecture(yolo_model, 80, verbose=False)

        assert validation_result['is_valid'] is True
        assert validation_result['model_classes'] == 80
        assert validation_result['expected_classes'] == 80
        assert len(validation_result['issues']) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_force_replacement_flag(self, yolo_model, dataset_config):
        """Test force replacement even when classes match."""
        # Create config with 80 classes (matches model)
        config_80 = {'nc': 80, 'names': [f'class_{i}' for i in range(80)]}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_80, f)
            config_path = f.name

        try:
            replacer = YOLOHeadReplacer(yolo_model, config_path, verbose=False)

            # Should not need replacement normally
            assert not replacer.needs_replacement()

            # But should work if forced
            # Note: This would require modifying the replacer to accept a force flag
            # which isn't implemented in the current version
        finally:
            os.unlink(config_path)

    def test_nonexistent_config_file(self, yolo_model):
        """Test handling of nonexistent config file."""
        with pytest.raises(ValueError, match="Failed to load dataset config"):
            YOLOHeadReplacer(yolo_model, "nonexistent_file.yaml", verbose=False)

    def test_malformed_yaml_file(self, yolo_model):
        """Test handling of malformed YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load dataset config"):
                YOLOHeadReplacer(yolo_model, config_path, verbose=False)
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])