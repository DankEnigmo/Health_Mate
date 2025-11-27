"""
Configuration Manager for Fall Detection Backend

This module handles loading and validating configuration from environment variables
and provides default values for optional parameters.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Union, Optional
from dotenv import load_dotenv

LOG = logging.getLogger(__name__)


@dataclass
class FallDetectionConfig:
    """Configuration dataclass for fall detection system"""
    
    # Required fields
    supabase_url: str
    supabase_key: str
    
    # Video source configuration
    video_source: Union[int, str] = 0
    
    # Fall detection parameters
    fall_threshold: float = 0.6
    movement_threshold: float = 0.3
    fps_target: int = 30
    
    # Model configuration
    enable_cuda: bool = False
    model_checkpoint: str = "shufflenetv2k16"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    def __post_init__(self):
        """Convert video_source to appropriate type"""
        # If video_source is a string that represents an integer, convert it
        if isinstance(self.video_source, str):
            try:
                self.video_source = int(self.video_source)
            except ValueError:
                # It's a path or URL, keep as string
                pass


class ConfigManager:
    """Manager for loading and validating fall detection configuration"""
    
    @staticmethod
    def load_config(env_file: Optional[str] = None) -> FallDetectionConfig:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in fall-detection directory
            
        Returns:
            FallDetectionConfig object with loaded configuration
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Load environment variables from .env file
        if env_file is None:
            # Default to .env in the fall-detection directory
            base_path = os.path.dirname(os.path.abspath(__file__))
            env_file = os.path.join(base_path, '.env')
        
        if os.path.exists(env_file):
            load_dotenv(env_file)
            LOG.info(f"Loaded environment variables from {env_file}")
        else:
            LOG.warning(f"No .env file found at {env_file}, using system environment variables")
        
        # Load required fields
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url:
            raise ValueError("SUPABASE_URL is required but not set in environment variables")
        if not supabase_key:
            raise ValueError("SUPABASE_KEY is required but not set in environment variables")
        
        # Load optional fields with defaults
        video_source = os.getenv('VIDEO_SOURCE', '0')
        
        try:
            fall_threshold = float(os.getenv('FALL_THRESHOLD', '0.6'))
        except ValueError:
            raise ValueError(f"FALL_THRESHOLD must be a valid float, got: {os.getenv('FALL_THRESHOLD')}")
        
        try:
            movement_threshold = float(os.getenv('MOVEMENT_THRESHOLD', '0.3'))
        except ValueError:
            raise ValueError(f"MOVEMENT_THRESHOLD must be a valid float, got: {os.getenv('MOVEMENT_THRESHOLD')}")
        
        try:
            fps_target = int(os.getenv('FPS_TARGET', '30'))
        except ValueError:
            raise ValueError(f"FPS_TARGET must be a valid integer, got: {os.getenv('FPS_TARGET')}")
        
        # Parse boolean for CUDA
        enable_cuda_str = os.getenv('ENABLE_CUDA', 'false').lower()
        enable_cuda = enable_cuda_str in ('true', '1', 'yes', 'on')
        
        model_checkpoint = os.getenv('MODEL_CHECKPOINT', 'shufflenetv2k16')
        
        host = os.getenv('HOST', '0.0.0.0')
        
        try:
            port = int(os.getenv('PORT', '8000'))
        except ValueError:
            raise ValueError(f"PORT must be a valid integer, got: {os.getenv('PORT')}")
        
        # Create configuration object
        config = FallDetectionConfig(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            video_source=video_source,
            fall_threshold=fall_threshold,
            movement_threshold=movement_threshold,
            fps_target=fps_target,
            enable_cuda=enable_cuda,
            model_checkpoint=model_checkpoint,
            host=host,
            port=port
        )
        
        # Validate the configuration
        ConfigManager.validate_config(config)
        
        LOG.info("Configuration loaded successfully")
        LOG.debug(f"Video source: {config.video_source}")
        LOG.debug(f"Fall threshold: {config.fall_threshold}")
        LOG.debug(f"Movement threshold: {config.movement_threshold}")
        LOG.debug(f"FPS target: {config.fps_target}")
        LOG.debug(f"CUDA enabled: {config.enable_cuda}")
        LOG.debug(f"Model checkpoint: {config.model_checkpoint}")
        
        return config
    
    @staticmethod
    def validate_config(config: FallDetectionConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: FallDetectionConfig object to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid with descriptive error message
        """
        # Validate Supabase URL format
        if not config.supabase_url.startswith(('http://', 'https://')):
            raise ValueError(f"SUPABASE_URL must be a valid HTTP/HTTPS URL, got: {config.supabase_url}")
        
        # Validate thresholds are positive
        if config.fall_threshold <= 0:
            raise ValueError(f"FALL_THRESHOLD must be positive, got: {config.fall_threshold}")
        
        if config.movement_threshold <= 0:
            raise ValueError(f"MOVEMENT_THRESHOLD must be positive, got: {config.movement_threshold}")
        
        # Validate FPS target is reasonable
        if config.fps_target <= 0 or config.fps_target > 120:
            raise ValueError(f"FPS_TARGET must be between 1 and 120, got: {config.fps_target}")
        
        # Validate port is in valid range
        if config.port < 1 or config.port > 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got: {config.port}")
        
        # Validate video source
        if isinstance(config.video_source, int):
            if config.video_source < 0:
                raise ValueError(f"VIDEO_SOURCE as integer must be non-negative, got: {config.video_source}")
        elif isinstance(config.video_source, str):
            # If it's a file path, check if it exists (only for local files, not URLs)
            if not config.video_source.startswith(('rtsp://', 'http://', 'https://')):
                if not os.path.exists(config.video_source):
                    raise ValueError(f"VIDEO_SOURCE file does not exist: {config.video_source}")
        
        LOG.info("Configuration validation passed")
        return True
