"""
Property-based tests for VideoStreamManager

**Feature: fall-detection-integration, Property 23: Video source connection**
Tests that for any valid video source specified in configuration, 
the Fall Detection Backend successfully connects to that source.
"""

import os
import sys
import pytest
import tempfile
import cv2
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given, strategies as st, settings, assume

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stream_manager import VideoStreamManager


# Test Property 23: Video source connection
# **Validates: Requirements 7.2**

@given(
    webcam_id=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=50)
def test_property_valid_webcam_source_connects(webcam_id):
    """
    Property: For any valid webcam ID (non-negative integer), the VideoStreamManager 
    should successfully initialize the video capture.
    
    Note: This test mocks cv2.VideoCapture to avoid requiring actual hardware.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=webcam_id,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Verify connection was established
        assert manager.is_connected is True
        assert manager.capture is not None
        
        # Verify cv2.VideoCapture was called with the correct source
        mock_capture_class.assert_called_once_with(webcam_id)


@given(
    rtsp_url=st.text(min_size=10, max_size=100).map(lambda x: f"rtsp://camera.example.com/{x}")
)
@settings(max_examples=50)
def test_property_valid_rtsp_source_connects(rtsp_url):
    """
    Property: For any valid RTSP URL, the VideoStreamManager should successfully 
    initialize the video capture with CAP_FFMPEG backend.
    
    Note: This test mocks cv2.VideoCapture to avoid requiring actual RTSP streams.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=rtsp_url,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Verify connection was established
        assert manager.is_connected is True
        assert manager.capture is not None
        
        # Verify cv2.VideoCapture was called with RTSP URL and CAP_FFMPEG
        mock_capture_class.assert_called_once_with(rtsp_url, cv2.CAP_FFMPEG)


@given(
    http_url=st.text(min_size=10, max_size=100).map(lambda x: f"http://camera.example.com/{x}")
)
@settings(max_examples=50)
def test_property_valid_http_source_connects(http_url):
    """
    Property: For any valid HTTP URL, the VideoStreamManager should successfully 
    initialize the video capture.
    
    Note: This test mocks cv2.VideoCapture to avoid requiring actual HTTP streams.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=http_url,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Verify connection was established
        assert manager.is_connected is True
        assert manager.capture is not None
        
        # Verify cv2.VideoCapture was called with the HTTP URL
        mock_capture_class.assert_called_once_with(http_url)


def test_property_valid_file_path_connects():
    """
    Property: For any valid video file path, the VideoStreamManager should successfully 
    initialize the video capture.
    
    This test creates a temporary video file to test with.
    """
    # Create a temporary video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_video_path = f.name
    
    try:
        # Create a simple video file using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))
        
        # Write a few frames
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Mock the detector, processor, model, and device
        mock_detector = Mock()
        mock_detector.fall_count = 0
        mock_processor = Mock()
        mock_model = Mock()
        mock_device = Mock()
        
        # Create VideoStreamManager with the temp file
        manager = VideoStreamManager(
            source=temp_video_path,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Verify connection was established
        assert manager.is_connected is True
        assert manager.capture is not None
        assert manager.capture.isOpened() is True
        
        # Clean up
        manager.release()
        
    finally:
        # Remove temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)


def test_property_invalid_source_fails_to_connect():
    """
    Property: For any invalid video source, the VideoStreamManager should fail to connect
    and set is_connected to False.
    """
    # Use a non-existent file path
    invalid_source = "/tmp/definitely_does_not_exist_video_12345.mp4"
    
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Create VideoStreamManager with invalid source
    manager = VideoStreamManager(
        source=invalid_source,
        detector=mock_detector,
        processor=mock_processor,
        model=mock_model,
        device=mock_device
    )
    
    # Verify connection failed
    assert manager.is_connected is False


def test_get_stats_returns_correct_structure():
    """
    Test that get_stats() returns a dictionary with all required metrics.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 5
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=0,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Get stats
        stats = manager.get_stats()
        
        # Verify structure
        assert 'fps' in stats
        assert 'latency' in stats
        assert 'fall_count' in stats
        assert 'frame_count' in stats
        assert 'is_connected' in stats
        assert 'is_processing' in stats
        
        # Verify types
        assert isinstance(stats['fps'], (int, float))
        assert isinstance(stats['latency'], (int, float))
        assert isinstance(stats['fall_count'], int)
        assert isinstance(stats['frame_count'], int)
        assert isinstance(stats['is_connected'], bool)
        assert isinstance(stats['is_processing'], bool)
        
        # Verify values
        assert stats['fall_count'] == 5
        assert stats['is_connected'] is True


def test_reconnect_attempts_for_rtsp_stream():
    """
    Test that the VideoStreamManager attempts to reconnect for RTSP streams
    when connection is lost.
    """
    rtsp_url = "rtsp://camera.example.com/stream"
    
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        # First call succeeds, second call (reconnect) also succeeds
        mock_capture1 = MagicMock()
        mock_capture1.isOpened.return_value = True
        
        mock_capture2 = MagicMock()
        mock_capture2.isOpened.return_value = True
        
        mock_capture_class.side_effect = [mock_capture1, mock_capture2]
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=rtsp_url,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Verify initial connection
        assert manager.is_connected is True
        
        # Simulate reconnection
        result = manager._reconnect()
        
        # Verify reconnection succeeded
        assert result is True
        assert manager.is_connected is True
        
        # Verify cv2.VideoCapture was called twice
        assert mock_capture_class.call_count == 2


def test_release_closes_capture():
    """
    Test that release() properly closes the video capture.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=0,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Release
        manager.release()
        
        # Verify release was called
        mock_capture.release.assert_called_once()
        assert manager.is_connected is False


def test_fps_calculation_uses_moving_average():
    """
    Test that FPS calculation uses a moving average for stability.
    """
    # Mock the detector, processor, model, and device
    mock_detector = Mock()
    mock_detector.fall_count = 0
    mock_processor = Mock()
    mock_model = Mock()
    mock_device = Mock()
    
    # Mock cv2.VideoCapture
    with patch('cv2.VideoCapture') as mock_capture_class:
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        # Create VideoStreamManager
        manager = VideoStreamManager(
            source=0,
            detector=mock_detector,
            processor=mock_processor,
            model=mock_model,
            device=mock_device
        )
        
        # Simulate multiple frame times
        frame_times = [0.033, 0.034, 0.032, 0.035, 0.033]  # ~30 FPS
        
        for frame_time in frame_times:
            manager._calculate_fps(frame_time)
        
        # Verify FPS is calculated (should be around 30)
        assert 28 <= manager.fps <= 32
        
        # Verify fps_history has the correct length
        assert len(manager.fps_history) == len(frame_times)
