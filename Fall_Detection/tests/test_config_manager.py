"""
Property-based tests for ConfigManager

**Feature: fall-detection-integration, Property 25: Invalid configuration handling**
Tests that invalid configuration causes the FastAPI Server to fail to start 
and log descriptive error messages.
"""

import os
import sys
import pytest
import tempfile
from hypothesis import given, strategies as st, settings

# Add parent directory to path to import config_manager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_manager import ConfigManager, FallDetectionConfig


# Test Property 25: Invalid configuration handling
# **Validates: Requirements 7.4**

@given(
    supabase_url=st.one_of(
        st.just(""),  # Empty string
        st.just("not-a-url"),  # Invalid URL format
        st.just("ftp://invalid.com"),  # Wrong protocol
        st.text(min_size=1, max_size=20).filter(lambda x: not x.startswith(('http://', 'https://'))),  # Random non-URL text
    )
)
@settings(max_examples=100)
def test_property_invalid_supabase_url_raises_error(supabase_url):
    """
    Property: For any invalid Supabase URL, configuration validation should raise ValueError
    with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url=supabase_url,
        supabase_key="valid_key_12345"
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "SUPABASE_URL" in error_message or "URL" in error_message


@given(
    fall_threshold=st.one_of(
        st.floats(max_value=0.0, exclude_max=True),  # Negative values
        st.just(0.0),  # Zero
        st.floats(min_value=-1000, max_value=0),  # More negative values
    )
)
@settings(max_examples=100)
def test_property_invalid_fall_threshold_raises_error(fall_threshold):
    """
    Property: For any non-positive fall threshold, configuration validation should raise ValueError
    with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        fall_threshold=fall_threshold
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "FALL_THRESHOLD" in error_message


@given(
    movement_threshold=st.one_of(
        st.floats(max_value=0.0, exclude_max=True),  # Negative values
        st.just(0.0),  # Zero
        st.floats(min_value=-1000, max_value=0),  # More negative values
    )
)
@settings(max_examples=100)
def test_property_invalid_movement_threshold_raises_error(movement_threshold):
    """
    Property: For any non-positive movement threshold, configuration validation should raise ValueError
    with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        movement_threshold=movement_threshold
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "MOVEMENT_THRESHOLD" in error_message


@given(
    fps_target=st.one_of(
        st.integers(max_value=0),  # Zero or negative
        st.integers(min_value=121, max_value=1000),  # Too high
    )
)
@settings(max_examples=100)
def test_property_invalid_fps_target_raises_error(fps_target):
    """
    Property: For any FPS target outside the valid range (1-120), configuration validation 
    should raise ValueError with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        fps_target=fps_target
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "FPS_TARGET" in error_message


@given(
    port=st.one_of(
        st.integers(max_value=0),  # Zero or negative
        st.integers(min_value=65536, max_value=100000),  # Too high
    )
)
@settings(max_examples=100)
def test_property_invalid_port_raises_error(port):
    """
    Property: For any port outside the valid range (1-65535), configuration validation 
    should raise ValueError with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        port=port
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "PORT" in error_message


@given(
    video_source=st.integers(min_value=-100, max_value=-1)  # Negative integers
)
@settings(max_examples=100)
def test_property_invalid_video_source_integer_raises_error(video_source):
    """
    Property: For any negative integer video source, configuration validation should raise 
    ValueError with a descriptive error message.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        video_source=video_source
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "VIDEO_SOURCE" in error_message


def test_property_invalid_video_source_file_path_raises_error():
    """
    Property: For any video source that is a non-existent file path (not a URL), 
    configuration validation should raise ValueError with a descriptive error message.
    """
    # Use a path that definitely doesn't exist
    non_existent_path = "/tmp/definitely_does_not_exist_video_file_12345.mp4"
    
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        video_source=non_existent_path
    )
    
    with pytest.raises(ValueError) as exc_info:
        ConfigManager.validate_config(config)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "VIDEO_SOURCE" in error_message or "does not exist" in error_message


def test_missing_required_supabase_url_raises_error():
    """
    Test that missing SUPABASE_URL raises ValueError during load_config.
    """
    # Create a temporary .env file without SUPABASE_URL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_KEY=test_key\n")
        temp_env_file = f.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "SUPABASE_URL" in error_message
    finally:
        os.unlink(temp_env_file)


def test_missing_required_supabase_key_raises_error():
    """
    Test that missing SUPABASE_KEY raises ValueError during load_config.
    """
    # Create a temporary .env file without SUPABASE_KEY
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        temp_env_file = f.name
    
    # Clear environment variables to ensure test isolation
    old_key = os.environ.pop('SUPABASE_KEY', None)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "SUPABASE_KEY" in error_message
    finally:
        os.unlink(temp_env_file)
        if old_key:
            os.environ['SUPABASE_KEY'] = old_key


def test_invalid_fall_threshold_string_raises_error():
    """
    Test that invalid FALL_THRESHOLD string raises ValueError during load_config.
    """
    # Create a temporary .env file with invalid FALL_THRESHOLD
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        f.write("SUPABASE_KEY=test_key\n")
        f.write("FALL_THRESHOLD=not_a_number\n")
        temp_env_file = f.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "FALL_THRESHOLD" in error_message
    finally:
        os.unlink(temp_env_file)


def test_invalid_movement_threshold_string_raises_error():
    """
    Test that invalid MOVEMENT_THRESHOLD string raises ValueError during load_config.
    """
    # Create a temporary .env file with invalid MOVEMENT_THRESHOLD
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        f.write("SUPABASE_KEY=test_key\n")
        f.write("FALL_THRESHOLD=0.6\n")  # Valid value
        f.write("MOVEMENT_THRESHOLD=invalid\n")
        temp_env_file = f.name
    
    # Clear environment variables to ensure test isolation
    old_vars = {}
    for var in ['FALL_THRESHOLD', 'MOVEMENT_THRESHOLD']:
        old_vars[var] = os.environ.pop(var, None)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "MOVEMENT_THRESHOLD" in error_message
    finally:
        os.unlink(temp_env_file)
        # Restore environment variables
        for var, value in old_vars.items():
            if value is not None:
                os.environ[var] = value


def test_invalid_fps_target_string_raises_error():
    """
    Test that invalid FPS_TARGET string raises ValueError during load_config.
    """
    # Create a temporary .env file with invalid FPS_TARGET
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        f.write("SUPABASE_KEY=test_key\n")
        f.write("FALL_THRESHOLD=0.6\n")  # Valid value
        f.write("MOVEMENT_THRESHOLD=0.3\n")  # Valid value
        f.write("FPS_TARGET=not_an_integer\n")
        temp_env_file = f.name
    
    # Clear environment variables to ensure test isolation
    old_vars = {}
    for var in ['FALL_THRESHOLD', 'MOVEMENT_THRESHOLD', 'FPS_TARGET']:
        old_vars[var] = os.environ.pop(var, None)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "FPS_TARGET" in error_message
    finally:
        os.unlink(temp_env_file)
        # Restore environment variables
        for var, value in old_vars.items():
            if value is not None:
                os.environ[var] = value


def test_invalid_port_string_raises_error():
    """
    Test that invalid PORT string raises ValueError during load_config.
    """
    # Create a temporary .env file with invalid PORT
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        f.write("SUPABASE_KEY=test_key\n")
        f.write("FALL_THRESHOLD=0.6\n")  # Valid value
        f.write("MOVEMENT_THRESHOLD=0.3\n")  # Valid value
        f.write("FPS_TARGET=30\n")  # Valid value
        f.write("PORT=not_a_port\n")
        temp_env_file = f.name
    
    # Clear environment variables to ensure test isolation
    old_vars = {}
    for var in ['FALL_THRESHOLD', 'MOVEMENT_THRESHOLD', 'FPS_TARGET', 'PORT']:
        old_vars[var] = os.environ.pop(var, None)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            ConfigManager.load_config(env_file=temp_env_file)
        
        error_message = str(exc_info.value)
        assert "PORT" in error_message
    finally:
        os.unlink(temp_env_file)
        # Restore environment variables
        for var, value in old_vars.items():
            if value is not None:
                os.environ[var] = value


def test_valid_configuration_passes():
    """
    Test that a valid configuration passes validation without errors.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        video_source=0,
        fall_threshold=0.6,
        movement_threshold=0.3,
        fps_target=30,
        enable_cuda=False,
        model_checkpoint="shufflenetv2k16",
        host="0.0.0.0",
        port=8000
    )
    
    # Should not raise any exception
    assert ConfigManager.validate_config(config) is True


def test_valid_configuration_with_rtsp_url():
    """
    Test that a valid configuration with RTSP URL passes validation.
    """
    config = FallDetectionConfig(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_key_12345",
        video_source="rtsp://camera.example.com/stream"
    )
    
    # Should not raise any exception
    assert ConfigManager.validate_config(config) is True


def test_load_config_with_defaults():
    """
    Test that load_config applies default values for optional parameters.
    """
    # Create a temporary .env file with only required fields
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SUPABASE_URL=https://test.supabase.co\n")
        f.write("SUPABASE_KEY=test_key_12345\n")
        temp_env_file = f.name
    
    # Clear optional environment variables to ensure defaults are used
    old_vars = {}
    for var in ['VIDEO_SOURCE', 'FALL_THRESHOLD', 'MOVEMENT_THRESHOLD', 'FPS_TARGET', 
                'ENABLE_CUDA', 'MODEL_CHECKPOINT', 'HOST', 'PORT']:
        old_vars[var] = os.environ.pop(var, None)
    
    try:
        config = ConfigManager.load_config(env_file=temp_env_file)
        
        # Verify defaults are applied
        assert config.video_source == 0
        assert config.fall_threshold == 0.6
        assert config.movement_threshold == 0.3
        assert config.fps_target == 30
        assert config.enable_cuda is False
        assert config.model_checkpoint == "shufflenetv2k16"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
    finally:
        os.unlink(temp_env_file)
        # Restore environment variables
        for var, value in old_vars.items():
            if value is not None:
                os.environ[var] = value
