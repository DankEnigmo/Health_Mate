"""
Network Factory Module

This module provides factory functions for creating and loading neural network models
for pose estimation. It's a simplified version adapted for the fall detection system.
"""

import logging
import os
import sys
import pickle
import io
import torch

LOG = logging.getLogger(__name__)


class RenameUnpickler(pickle.Unpickler):
    """Custom unpickler to remap openpifpaf module names to local modules."""
    
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        # Setup the persistent_load for PyTorch tensors
        self.persistent_load = self._persistent_load
    
    def _persistent_load(self, pid):
        """Handle PyTorch's persistent tensors."""
        # This is called by pickle for PyTorch tensors
        # We need to let torch handle it
        raise pickle.UnpicklingError("Use torch.load instead")
    
    def find_class(self, module, name):
        """Remap openpifpaf imports to local vendorized modules."""
        # Remap openpifpaf module references to current package
        if module.startswith('openpifpaf.'):
            # Remove 'openpifpaf.' prefix to use local modules
            local_module = module.replace('openpifpaf.', '', 1)
            try:
                return super().find_class(local_module, name)
            except (ModuleNotFoundError, AttributeError):
                # If local module not found, try without any prefix
                try:
                    return super().find_class(name, name)
                except:
                    pass
        
        # Fall back to default behavior
        return super().find_class(module, name)


def renamed_load(f, map_location=None, weights_only=False):
    """Load a checkpoint with module name remapping."""
    if weights_only:
        return torch.load(f, map_location=map_location, weights_only=True)
    
    # For this vendorized version, we need to handle the openpifpaf -> local module mapping
    # The simplest solution: just use torch.load with weights_only=False after ensuring
    # the parent directory is importable as a package
    
    import sys
    
    # Get the Fall_Detection directory
    current_file = os.path.abspath(__file__)
    network_dir = os.path.dirname(current_file)
    fall_detection_dir = os.path.dirname(network_dir)
    parent_dir = os.path.dirname(fall_detection_dir)
    
    # Temporarily add parent to path so Fall_Detection can be imported as a package
    path_added = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        path_added = True
    
    # Save original modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key.startswith('openpifpaf'):
            saved_modules[key] = sys.modules.pop(key)
    
    try:
        # Create openpifpaf as an alias to Fall_Detection
        import Fall_Detection
        sys.modules['openpifpaf'] = Fall_Detection
        
        # Force import of submodules
        import Fall_Detection.network
        import Fall_Detection.network.basenetworks
        import Fall_Detection.network.nets
        import Fall_Detection.network.heads
        
        sys.modules['openpifpaf'] = Fall_Detection
        sys.modules['openpifpaf.network'] = Fall_Detection.network
        sys.modules['openpifpaf.network.nets'] = Fall_Detection.network.nets
        sys.modules['openpifpaf.network.heads'] = Fall_Detection.network.heads
        sys.modules['openpifpaf.network.basenetworks'] = Fall_Detection.network.basenetworks
        
        # Try to import encoder and decoder
        try:
            import Fall_Detection.encoder
            sys.modules['openpifpaf.encoder'] = Fall_Detection.encoder
        except Exception as e:
            LOG.debug(f"Could not import encoder: {e}")
        
        try:
            import Fall_Detection.decoder
            sys.modules['openpifpaf.decoder'] = Fall_Detection.decoder
        except Exception as e:
            LOG.debug(f"Could not import decoder: {e}")
        
        # Now load with torch
        checkpoint = torch.load(f, map_location=map_location, weights_only=False)
        return checkpoint
        
    finally:
        # Clean up
        for key in list(sys.modules.keys()):
            if key.startswith('openpifpaf'):
                sys.modules.pop(key, None)
        
        # Restore saved modules
        sys.modules.update(saved_modules)
        
        # Remove added path
        if path_added and parent_dir in sys.path:
            sys.path.remove(parent_dir)

# Model URLs for different checkpoints
MODEL_URLS = {
    'shufflenetv2k16': 'https://github.com/openpifpaf/torchhub/releases/download/v0.12.9/shufflenetv2k16-210224-123448-cocokp-o10s-d020d7f1.pkl',
    'shufflenetv2k30': 'https://github.com/openpifpaf/torchhub/releases/download/v0.12.9/shufflenetv2k30-210224-074128-cocokp-o10s-59ca2b89.pkl',
    'resnet50': 'https://github.com/openpifpaf/torchhub/releases/download/v0.12.9/resnet50-210224-202010-cocokp-o10s-627d901e.pkl',
}


def cli(parser):
    """Add command-line arguments for network configuration."""
    group = parser.add_argument_group('network configuration')
    group.add_argument('--checkpoint', default='shufflenetv2k16',
                       help='Load a model checkpoint. Use "resnet50", "shufflenetv2k16", etc.')
    group.add_argument('--basenet', default=None,
                       help='base network, e.g. resnet50')
    group.add_argument('--headnets', default=['cif', 'caf'], nargs='+',
                       help='head networks')
    group.add_argument('--no-pretrain', dest='pretrain', default=True, action='store_false',
                       help='create model without ImageNet pretraining')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--multi-scale-hflip', default=True, action='store_true',
                       help='[experimental]')
    group.add_argument('--no-multi-scale-hflip', dest='multi_scale_hflip',
                       default=True, action='store_false',
                       help='[experimental]')
    group.add_argument('--cross-talk', default=0.0, type=float,
                       help='[experimental]')


def configure(args):
    """Configure network settings from arguments."""
    # This is a placeholder for configuration
    pass


def local_checkpoint_path(checkpoint):
    """
    Get the local path for a checkpoint file.
    
    Args:
        checkpoint: Name of the checkpoint (e.g., 'shufflenetv2k16')
        
    Returns:
        Path to the local checkpoint file
    """
    if os.path.exists(checkpoint):
        return checkpoint
    
    # Check in common locations
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate filename from checkpoint name
    if checkpoint in MODEL_URLS:
        filename = MODEL_URLS[checkpoint].split('/')[-1]
        local_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(local_path):
            return local_path
    
    return None


def factory(*, checkpoint='shufflenetv2k16', download_progress=True):
    """
    Factory function to create and load a model.
    
    Args:
        checkpoint: Name or path of the checkpoint to load
        download_progress: Whether to show download progress
        
    Returns:
        Tuple of (model, epoch) where model is the loaded neural network
    """
    LOG.info('Loading checkpoint: %s', checkpoint)
    
    # First, try to load from local file if checkpoint is a path
    if os.path.exists(checkpoint):
        LOG.info('Loading from local checkpoint file: %s', checkpoint)
        try:
            checkpoint_data = renamed_load(checkpoint, map_location='cpu', weights_only=False)
            
            # Extract model from checkpoint
            if isinstance(checkpoint_data, dict):
                model = checkpoint_data.get('model', checkpoint_data)
                epoch = checkpoint_data.get('epoch', 0)
            else:
                model = checkpoint_data
                epoch = 0
            
            LOG.info('Model loaded successfully from local file')
            return model, epoch
        except Exception as e:
            LOG.error('Failed to load from checkpoint file: %s', e)
            raise RuntimeError(f'Could not load checkpoint {checkpoint}') from e
    
    # Try to load from cached checkpoint
    local_path = local_checkpoint_path(checkpoint)
    
    if local_path and os.path.exists(local_path):
        LOG.info('Loading from cached checkpoint: %s', local_path)
        try:
            checkpoint_data = renamed_load(local_path, map_location='cpu', weights_only=False)
            
            # Extract model from checkpoint
            if isinstance(checkpoint_data, dict):
                model = checkpoint_data.get('model', checkpoint_data)
                epoch = checkpoint_data.get('epoch', 0)
            else:
                model = checkpoint_data
                epoch = 0
            
            LOG.info('Model loaded successfully from cached file')
            return model, epoch
        except Exception as e:
            LOG.warning('Failed to load from cached checkpoint: %s', e)
    
    # If checkpoint is in our known models, try to download it
    if checkpoint in MODEL_URLS:
        url = MODEL_URLS[checkpoint]
        LOG.info('Downloading model from: %s', url)
        
        try:
            checkpoint_data = torch.hub.load_state_dict_from_url(
                url,
                progress=download_progress,
                map_location='cpu'
            )
            
            # Extract model from checkpoint
            if isinstance(checkpoint_data, dict):
                model = checkpoint_data.get('model', checkpoint_data)
                epoch = checkpoint_data.get('epoch', 0)
            else:
                model = checkpoint_data
                epoch = 0
            
            LOG.info('Model downloaded and loaded successfully')
            return model, epoch
            
        except Exception as download_error:
            LOG.warning('Failed to download model: %s', download_error)
            LOG.warning('Will try to build model from scratch...')
    
    # Last resort: try to build model from scratch using torchvision
    LOG.info('Attempting to build fresh model architecture: %s', checkpoint)
    try:
        from . import basenetworks
        
        # Map checkpoint names to model builders
        if checkpoint == 'shufflenetv2k16':
            LOG.info('Building ShuffleNetV2K16 from scratch...')
            model = basenetworks.ShuffleNetV2K(
                stages_repeats=[4, 8, 4], 
                stages_out_channels=[24, 348, 696, 1392, 1392]
            )
            LOG.warning('Created fresh model (untrained). For better results, download a pretrained checkpoint.')
            return model, 0
        elif checkpoint == 'shufflenetv2k30':
            model = basenetworks.ShuffleNetV2K(
                stages_repeats=[8, 16, 6],
                stages_out_channels=[32, 512, 1024, 2048, 2048]
            )
            return model, 0
        else:
            LOG.error(f'Unknown checkpoint type: {checkpoint}')
            raise RuntimeError(f'Could not build model for checkpoint: {checkpoint}')
    except Exception as e:
        LOG.error(f'Failed to build model from scratch: {e}')
        raise RuntimeError(f'Could not find, download, or build checkpoint: {checkpoint}') from e


def factory_from_args(args):
    """
    Create a model from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (model, epoch)
    """
    return factory(checkpoint=args.checkpoint)


# For backward compatibility
def load_checkpoint(checkpoint, **kwargs):
    """Load a checkpoint (alias for factory)."""
    return factory(checkpoint=checkpoint, **kwargs)
