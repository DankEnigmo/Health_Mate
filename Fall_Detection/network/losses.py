"""
Losses Module

Placeholder for loss functions. This is not used in the API server
but is imported by the network module.
"""

import logging
import torch

LOG = logging.getLogger(__name__)


class Loss(torch.nn.Module):
    """Base loss class."""
    
    def __init__(self):
        """Initialize loss."""
        super().__init__()
    
    def forward(self, *args, **kwargs):
        """Compute loss."""
        raise NotImplementedError("Loss computation not implemented")
