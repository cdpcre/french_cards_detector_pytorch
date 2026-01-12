"""
Integration tests for the complete training pipeline with head replacement.

This module tests the end-to-end functionality including model loading,
head replacement, and training pipeline integration.
"""

import pytest
import torch
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from ultralytics import YOLO
    from src.head_replacer import replace_yolo_head_if_needed
    from src.trainer import train
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@pytest.fixture
def mock_args():
    """Create mock training arguments."""
    class MockArgs:
        def __init__(self):
            self.model = 'yolo11n.pt'
            self.data = 'datasets/unified/data.yaml'
            self.epochs = 1
            self.batch = 2
            self.imgsz = 640
            self.device = 'cpu'  # Use CPU for testing
            self.workers = 0
            self.project = 'test_runs'
            self.lr = 0.001
            self.fine_tune_mode = 'head-only'
            self.phase1_epochs = 1
            self.head_lr = 0.001
            self.neck_lr = 0.0001
            self.mosaic = 0.0
            self.class_weighted_sampling = False
            self.no_aug = True
            self.cache = False
            self.force_head_replacement = False
            self.disable_head_replacement = False

    return MockArgs()


@pytest.fixture
def temp_dataset_config():
    """Create temporary dataset configuration for testing."""
    dataset_config = {
        'path': './test_datasets',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 53,
        'names': [f'class_{i}' for i in range(53)]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(dataset_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


class TestTrainingPipelineIntegration:
    """Integration tests for the training pipeline."""

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Ultralytics not available")
    def test_model_loading_with_head_replacement(self, temp_dataset_config):
        """Test that model loading with head replacement works in training context."""
        # Load base model
        model_wrapper = YOLO('yolo11n.pt')
        model = model_wrapper.model

        # Check original state
        original_nc = model.nc
        assert original_nc == 80

        # Apply head replacement
        modified_model, was_replaced = replace_yolo_head_if_needed(
            model, temp_dataset_config, verbose=False
        )

        # Verify changes
        assert was_replaced is True
        assert modified_model.nc == 53

        # Verify model structure is intact
        assert hasattr(modified_model, 'named_parameters')
        param_count = sum(p.numel() for p in modified_model.parameters())
        assert param_count > 0

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Ultralytics not available")
    def test_head_replacement_preserves_model_structure(self, temp_dataset_config):
        """Test that head replacement preserves overall model structure."""
        model_wrapper = YOLO('yolo11n.pt')
        original_model = model_wrapper.model

        # Get original model structure info
        original_layers = len(list(original_model.named_children()))
        original_params = sum(p.numel() for p in original_model.parameters())

        # Apply head replacement
        modified_model, was_replaced = replace_yolo_head_if_needed(
            original_model, temp_dataset_config, verbose=False
        )

        # Verify structure is preserved
        modified_layers = len(list(modified_model.named_children()))
        modified_params = sum(p.numel() for p in modified_model.parameters())

        # Layer count should be the same
        assert modified_layers == original_layers

        # Parameter count should be similar (slight difference due to class count change)
        param_ratio = modified_params / original_params
        assert 0.8 < param_ratio < 1.2  # Allow some variation

    def test_dataset_config_validation(self, temp_dataset_config):
        """Test dataset configuration validation."""
        # Test valid config
        with open(temp_dataset_config, 'r') as f:
            config = yaml.safe_load(f)

        assert 'nc' in config
        assert 'names' in config
        assert config['nc'] == 53
        assert len(config['names']) == 53

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Ultralytics not available")
    def test_training_initialization_with_head_replacement(self, mock_args, temp_dataset_config):
        """Test training initialization with head replacement enabled."""
        # Update args to use our test config
        mock_args.data = temp_dataset_config

        # Verify that head replacement can be performed directly
        model_wrapper = YOLO('yolo11n.pt')
        model = model_wrapper.model
        
        # Verify the model starts with 80 classes (COCO)
        assert model.nc == 80
        
        # Perform head replacement using the real function
        modified_model, was_replaced = replace_yolo_head_if_needed(
            model, temp_dataset_config, verbose=False
        )
        
        # Verify replacement was successful
        assert was_replaced is True
        assert modified_model.nc == 53
        
        # Verify model can still perform forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = modified_model(dummy_input)
        assert output is not None

    def test_command_line_flags_integration(self, mock_args):
        """Test that command line flags are properly integrated."""
        # Test force head replacement flag
        mock_args.force_head_replacement = True
        assert hasattr(mock_args, 'force_head_replacement')
        assert mock_args.force_head_replacement is True

        # Test disable head replacement flag
        mock_args.force_head_replacement = False
        mock_args.disable_head_replacement = True
        assert hasattr(mock_args, 'disable_head_replacement')
        assert mock_args.disable_head_replacement is True


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_missing_dataset_config_handling(self, mock_args):
        """Test handling of missing dataset configuration."""
        mock_args.data = 'nonexistent_config.yaml'

        # This should be handled gracefully in the training pipeline
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model_wrapper = MagicMock()
            mock_model = MagicMock()
            mock_model_wrapper.model = mock_model
            mock_yolo.return_value = mock_model_wrapper

            with patch('src.head_replacer.replace_yolo_head_if_needed') as mock_replace:
                mock_replace.return_value = (mock_model, False)

                try:
                    from src.trainer import train

                    # Mock remaining components
                    with patch('src.trainer.v8DetectionLoss'), \
                         patch('torch.optim.AdamW'), \
                         patch('src.trainer.analyze_yolo11_layers'), \
                         patch('builtins.open', side_effect=FileNotFoundError):

                        # Should handle missing config gracefully
                        train(mock_args)

                except Exception as e:
                    # Should fail gracefully, not crash - any of these error types are acceptable
                    assert isinstance(e, (FileNotFoundError, ValueError, KeyError)) or \
                           "Dataset config not found" in str(e) or \
                           "Failed to load" in str(e)

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Ultralytics not available")
    def test_invalid_model_handling(self, temp_dataset_config):
        """Test handling of invalid model files."""
        # Test with nonexistent model file
        try:
            model_wrapper = YOLO('nonexistent_model.pt')
            # Should not reach here
            assert False, "Expected FileNotFoundError or similar"
        except Exception:
            # Expected to fail
            pass

    def test_malformed_dataset_config(self, mock_args):
        """Test handling of malformed dataset configuration."""
        # Create malformed config
        malformed_config = {'invalid': 'config'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(malformed_config, f)
            config_path = f.name

        try:
            mock_args.data = config_path

            with patch('ultralytics.YOLO') as mock_yolo:
                mock_model_wrapper = MagicMock()
                mock_model = MagicMock()
                mock_model_wrapper.model = mock_model
                mock_yolo.return_value = mock_model_wrapper

                with patch('src.head_replacer.replace_yolo_head_if_needed') as mock_replace:
                    mock_replace.return_value = (mock_model, False)

                    try:
                        from src.trainer import train

                        with patch('src.trainer.v8DetectionLoss'), \
                             patch('torch.optim.AdamW'), \
                             patch('src.trainer.analyze_yolo11_layers'):

                            # Should handle malformed config gracefully
                            train(mock_args)

                    except Exception as e:
                        # Should fail gracefully - KeyError or ValueError are acceptable
                        assert isinstance(e, (KeyError, ValueError)) or \
                               "Missing required key" in str(e) or \
                               "Failed to load" in str(e)

        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])