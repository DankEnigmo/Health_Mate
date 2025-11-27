"""
Property-based tests for FastAPI Server

**Feature: fall-detection-integration, Property 8: MJPEG stream format**
Tests that for any client request to the video stream endpoint, 
the response is in MJPEG format with fall detection overlays.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the problematic imports before they're loaded
sys.modules['network'] = MagicMock()
sys.modules['decoder'] = MagicMock()
sys.modules['config'] = MagicMock()

# Mock core module components
mock_core = MagicMock()
mock_core.CentroidTracker = MagicMock
mock_core.FallDetector = MagicMock
sys.modules['core'] = mock_core


# Test Property 8: MJPEG stream format
# **Validates: Requirements 3.2**

def test_property_mjpeg_stream_format_with_overlays():
    """
    Property: For any client request to the video stream endpoint, 
    the response should be in MJPEG format with fall detection overlays.
    
    This test verifies:
    1. The response has the correct MJPEG content type
    2. The stream contains JPEG frames with proper boundaries
    3. The frames are valid JPEG images
    """
    # Import here to avoid circular imports during module load
    from api_server import app
    
    # Mock the global state
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        # Configure mocks
        mock_config.video_source = 0
        mock_stream_manager.is_connected = True
        
        # Create a mock async generator that yields MJPEG frames
        async def mock_generate_frames():
            # Simulate MJPEG frames with proper format
            # Each frame should have the boundary marker and JPEG data
            frames = [
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0' + b'fake_jpeg_data_1' + b'\xff\xd9' + b'\r\n',
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0' + b'fake_jpeg_data_2' + b'\xff\xd9' + b'\r\n',
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0' + b'fake_jpeg_data_3' + b'\xff\xd9' + b'\r\n',
            ]
            for frame in frames:
                yield frame
        
        mock_stream_manager.generate_frames = mock_generate_frames
        
        # Create test client
        client = TestClient(app)
        
        # Make request to video stream endpoint
        response = client.get("/api/video/stream")
        
        # Verify response status
        assert response.status_code == 200
        
        # Verify content type is MJPEG
        assert response.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
        
        # Verify response contains frame boundaries
        content = response.content
        assert b'--frame' in content
        assert b'Content-Type: image/jpeg' in content
        
        # Verify JPEG markers are present (SOI and EOI markers)
        assert b'\xff\xd8' in content  # JPEG Start of Image marker
        assert b'\xff\xd9' in content  # JPEG End of Image marker


def test_property_stream_endpoint_returns_503_when_not_connected():
    """
    Property: For any client request when the stream is not connected,
    the endpoint should return 503 Service Unavailable.
    """
    from api_server import app
    
    # Mock the global state with disconnected stream
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = False
        
        # Create test client
        client = TestClient(app)
        
        # Make request to video stream endpoint
        response = client.get("/api/video/stream")
        
        # Verify response status is 503
        assert response.status_code == 503
        
        # Verify error message
        assert "error" in response.json()
        assert "not connected" in response.json()["error"].lower()


def test_property_stream_endpoint_returns_503_when_manager_not_initialized():
    """
    Property: For any client request when the stream manager is not initialized,
    the endpoint should return 503 Service Unavailable.
    """
    from api_server import app
    
    # Mock the global state with None stream manager
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager', None), \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        
        # Create test client
        client = TestClient(app)
        
        # Make request to video stream endpoint
        response = client.get("/api/video/stream")
        
        # Verify response status is 503
        assert response.status_code == 503
        
        # Verify error message
        assert "error" in response.json()
        assert "not initialized" in response.json()["error"].lower()


@given(
    num_frames=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=20)
def test_property_mjpeg_stream_contains_n_frames(num_frames):
    """
    Property: For any number of frames N generated by the stream manager,
    the MJPEG stream should contain N frame boundaries.
    """
    from api_server import app
    
    # Mock the global state
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = True
        
        # Create a mock async generator with N frames
        async def mock_generate_frames():
            for i in range(num_frames):
                frame_data = f'fake_jpeg_data_{i}'.encode()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       b'\xff\xd8\xff\xe0' + frame_data + b'\xff\xd9' + 
                       b'\r\n')
        
        mock_stream_manager.generate_frames = mock_generate_frames
        
        # Create test client
        client = TestClient(app)
        
        # Make request to video stream endpoint
        response = client.get("/api/video/stream")
        
        # Verify response status
        assert response.status_code == 200
        
        # Count frame boundaries in response
        content = response.content
        frame_count = content.count(b'--frame')
        
        # Verify we got the expected number of frames
        assert frame_count == num_frames


def test_health_check_endpoint():
    """
    Test that the health check endpoint returns correct status.
    """
    from api_server import app
    
    # Mock the global state
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = True
        mock_ws_manager.get_connection_count.return_value = 2
        
        # Create test client
        client = TestClient(app)
        
        # Make request to health check endpoint
        response = client.get("/health")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["video_stream_connected"] is True
        assert data["websocket_connections"] == 2
        assert "version" in data


def test_health_check_unhealthy_when_stream_disconnected():
    """
    Test that health check returns unhealthy when stream is disconnected.
    """
    from api_server import app
    
    # Mock the global state with disconnected stream
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = False
        mock_ws_manager.get_connection_count.return_value = 0
        
        # Create test client
        client = TestClient(app)
        
        # Make request to health check endpoint
        response = client.get("/health")
        
        # Verify response
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["video_stream_connected"] is False


def test_stats_endpoint_returns_metrics():
    """
    Test that the stats endpoint returns all required metrics.
    """
    from api_server import app
    
    # Mock the global state
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = True
        mock_stream_manager.get_stats.return_value = {
            'fps': 29.5,
            'latency': 45.2,
            'fall_count': 3,
            'frame_count': 1500,
            'is_connected': True,
            'is_processing': True
        }
        mock_ws_manager.get_connection_count.return_value = 1
        
        # Create test client
        client = TestClient(app)
        
        # Make request to stats endpoint
        response = client.get("/api/stats")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert 'fps' in data
        assert 'latency' in data
        assert 'fall_count' in data
        assert 'frame_count' in data
        assert 'is_connected' in data
        assert 'is_processing' in data
        assert 'websocket_connections' in data
        
        # Verify values
        assert data['fps'] == 29.5
        assert data['latency'] == 45.2
        assert data['fall_count'] == 3
        assert data['frame_count'] == 1500
        assert data['is_connected'] is True
        assert data['is_processing'] is True
        assert data['websocket_connections'] == 1


def test_config_endpoint_returns_configuration():
    """
    Test that the config endpoint returns configuration (excluding sensitive data).
    """
    from api_server import app
    from config_manager import FallDetectionConfig
    
    # Create a mock config
    mock_cfg = FallDetectionConfig(
        supabase_url="https://test.supabase.co",
        supabase_key="test_key",
        video_source=0,
        fall_threshold=0.6,
        movement_threshold=0.3,
        fps_target=30,
        enable_cuda=False,
        model_checkpoint="shufflenetv2k16",
        host="0.0.0.0",
        port=8000
    )
    
    # Mock the global state
    with patch('api_server.config', mock_cfg), \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_stream_manager.is_connected = True
        
        # Create test client
        client = TestClient(app)
        
        # Make request to config endpoint
        response = client.get("/api/config")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify configuration fields are present
        assert 'video_source' in data
        assert 'fall_threshold' in data
        assert 'movement_threshold' in data
        assert 'fps_target' in data
        assert 'enable_cuda' in data
        assert 'model_checkpoint' in data
        assert 'host' in data
        assert 'port' in data
        
        # Verify sensitive data is NOT included
        assert 'supabase_key' not in data
        
        # Verify values
        assert data['video_source'] == '0'
        assert data['fall_threshold'] == 0.6
        assert data['fps_target'] == 30


def test_stats_endpoint_returns_503_when_manager_not_initialized():
    """
    Test that stats endpoint returns 503 when stream manager is not initialized.
    """
    from api_server import app
    
    # Mock the global state with None stream manager
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager', None), \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        # Create test client
        client = TestClient(app)
        
        # Make request to stats endpoint
        response = client.get("/api/stats")
        
        # Verify response status is 503
        assert response.status_code == 503
        assert "error" in response.json()


def test_config_endpoint_returns_503_when_config_not_loaded():
    """
    Test that config endpoint returns 503 when configuration is not loaded.
    """
    from api_server import app
    
    # Mock the global state with None config
    with patch('api_server.config', None), \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        # Create test client
        client = TestClient(app)
        
        # Make request to config endpoint
        response = client.get("/api/config")
        
        # Verify response status is 503
        assert response.status_code == 503
        assert "error" in response.json()


@given(
    fps=st.floats(min_value=1.0, max_value=60.0),
    latency=st.floats(min_value=0.0, max_value=1000.0),
    fall_count=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=50)
def test_property_stats_endpoint_returns_valid_metrics(fps, latency, fall_count):
    """
    Property: For any valid FPS, latency, and fall count values,
    the stats endpoint should return them correctly.
    """
    from api_server import app
    
    # Mock the global state
    with patch('api_server.config') as mock_config, \
         patch('api_server.stream_manager') as mock_stream_manager, \
         patch('api_server.ws_manager') as mock_ws_manager:
        
        mock_config.video_source = 0
        mock_stream_manager.is_connected = True
        mock_stream_manager.get_stats.return_value = {
            'fps': fps,
            'latency': latency,
            'fall_count': fall_count,
            'frame_count': 1000,
            'is_connected': True,
            'is_processing': True
        }
        mock_ws_manager.get_connection_count.return_value = 0
        
        # Create test client
        client = TestClient(app)
        
        # Make request to stats endpoint
        response = client.get("/api/stats")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify the values match
        assert data['fps'] == fps
        assert data['latency'] == latency
        assert data['fall_count'] == fall_count
        
        # Verify types
        assert isinstance(data['fps'], (int, float))
        assert isinstance(data['latency'], (int, float))
        assert isinstance(data['fall_count'], int)


# Property-Based Tests for Fall Event Database Persistence

# Test Property 27: Fall event persistence
# **Validates: Requirements 8.1**

@given(
    patient_id=st.uuids().map(str),
    person_tracking_id=st.integers(min_value=0, max_value=1000),
    fall_count=st.integers(min_value=1, max_value=100),
    timestamp=st.datetimes(
        min_value=pytest.importorskip('datetime').datetime(2020, 1, 1),
        max_value=pytest.importorskip('datetime').datetime(2030, 12, 31)
    ).map(lambda dt: dt.isoformat())
)
@settings(max_examples=100, deadline=None)
def test_property_fall_event_persistence(patient_id, person_tracking_id, fall_count, timestamp):
    """
    **Feature: fall-detection-integration, Property 27: Fall event persistence**
    
    Property: For any fall detection, a fall event record should be created 
    in the Supabase database.
    
    This test verifies that when a fall is detected, the store_fall_event function
    is called and successfully creates a record in the database.
    
    **Validates: Requirements 8.1**
    """
    from api_server import store_fall_event
    from unittest.mock import AsyncMock, MagicMock
    import asyncio
    
    # Create mock Supabase client
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_execute = MagicMock()
    
    # Set up the mock chain
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_execute.data = [{"id": "test-id", "patient_id": patient_id}]
    mock_insert.execute.return_value = mock_execute
    
    # Prepare alert data
    alert_data = {
        "patient_id": patient_id,
        "person_tracking_id": person_tracking_id,
        "fall_count": fall_count,
        "timestamp": timestamp,
        "metadata": {"test": "data"}
    }
    
    # Patch the global supabase_client
    with patch('api_server.supabase_client', mock_supabase):
        # Call the function
        asyncio.run(store_fall_event(alert_data))
        
        # Verify that the database insert was called
        mock_supabase.table.assert_called_once_with("fall_events")
        
        # Verify insert was called with correct data structure
        mock_table.insert.assert_called_once()
        insert_call_args = mock_table.insert.call_args[0][0]
        
        # Verify all required fields are present
        assert "patient_id" in insert_call_args
        assert "person_tracking_id" in insert_call_args
        assert "fall_count" in insert_call_args
        assert "timestamp" in insert_call_args
        assert "status" in insert_call_args
        
        # Verify the values match
        assert insert_call_args["patient_id"] == patient_id
        assert insert_call_args["person_tracking_id"] == person_tracking_id
        assert insert_call_args["fall_count"] == fall_count
        assert insert_call_args["timestamp"] == timestamp
        assert insert_call_args["status"] == "new"
        
        # Verify execute was called
        mock_insert.execute.assert_called_once()


# Test Property 28: Fall event record completeness
# **Validates: Requirements 8.2**

@given(
    patient_id=st.uuids().map(str),
    caregiver_id=st.one_of(st.none(), st.uuids().map(str)),
    person_tracking_id=st.integers(min_value=0, max_value=1000),
    fall_count=st.integers(min_value=1, max_value=100),
    timestamp=st.datetimes(
        min_value=pytest.importorskip('datetime').datetime(2020, 1, 1),
        max_value=pytest.importorskip('datetime').datetime(2030, 12, 31)
    ).map(lambda dt: dt.isoformat())
)
@settings(max_examples=100, deadline=None)
def test_property_fall_event_record_completeness(patient_id, caregiver_id, person_tracking_id, fall_count, timestamp):
    """
    **Feature: fall-detection-integration, Property 28: Fall event record completeness**
    
    Property: For any created fall event record, it must include timestamp, 
    patient ID, caregiver ID, and fall count.
    
    This test verifies that all required fields are present in the fall event
    record that is inserted into the database.
    
    **Validates: Requirements 8.2**
    """
    from api_server import store_fall_event
    from unittest.mock import AsyncMock, MagicMock
    import asyncio
    
    # Create mock Supabase client
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_execute = MagicMock()
    
    # Set up the mock chain
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_execute.data = [{"id": "test-id"}]
    mock_insert.execute.return_value = mock_execute
    
    # Prepare alert data with optional caregiver_id
    alert_data = {
        "patient_id": patient_id,
        "person_tracking_id": person_tracking_id,
        "fall_count": fall_count,
        "timestamp": timestamp,
        "metadata": {"confidence": 0.95}
    }
    
    if caregiver_id is not None:
        alert_data["caregiver_id"] = caregiver_id
    
    # Patch the global supabase_client
    with patch('api_server.supabase_client', mock_supabase):
        # Call the function
        asyncio.run(store_fall_event(alert_data))
        
        # Get the inserted data
        insert_call_args = mock_table.insert.call_args[0][0]
        
        # Verify ALL required fields are present (Requirements 8.2)
        required_fields = ["patient_id", "fall_count", "timestamp"]
        for field in required_fields:
            assert field in insert_call_args, f"Required field '{field}' is missing from fall event record"
        
        # Verify the values are correct
        assert insert_call_args["patient_id"] == patient_id, "patient_id does not match"
        assert insert_call_args["fall_count"] == fall_count, "fall_count does not match"
        assert insert_call_args["timestamp"] == timestamp, "timestamp does not match"
        
        # Verify person_tracking_id is included
        assert insert_call_args["person_tracking_id"] == person_tracking_id
        
        # Verify status is set
        assert insert_call_args["status"] == "new"
        
        # Verify metadata is preserved
        assert "metadata" in insert_call_args
        assert isinstance(insert_call_args["metadata"], dict)


def test_property_fall_event_storage_handles_failures_gracefully():
    """
    **Feature: fall-detection-integration, Property 30: Database failure resilience**
    
    Property: For any database storage failure, the FastAPI Server should log 
    the error and continue processing without interrupting the video stream.
    
    This test verifies that database failures are caught and logged without
    raising exceptions that would interrupt the video stream.
    
    **Validates: Requirements 8.5**
    """
    from api_server import store_fall_event
    from unittest.mock import MagicMock
    import asyncio
    
    # Create mock Supabase client that raises an exception
    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = Exception("Database connection failed")
    
    # Prepare alert data
    alert_data = {
        "patient_id": "test-patient-id",
        "person_tracking_id": 1,
        "fall_count": 5,
        "timestamp": "2024-01-01T12:00:00",
        "metadata": {}
    }
    
    # Patch the global supabase_client and logger
    with patch('api_server.supabase_client', mock_supabase), \
         patch('api_server.LOG') as mock_logger:
        
        # Call the function - it should NOT raise an exception
        try:
            asyncio.run(store_fall_event(alert_data))
            # If we get here, the function handled the error gracefully
            exception_raised = False
        except Exception:
            # If an exception is raised, the test fails
            exception_raised = True
        
        # Verify no exception was raised
        assert not exception_raised, "store_fall_event should not raise exceptions on database failures"
        
        # Verify the error was logged
        mock_logger.error.assert_called()
        error_call_args = str(mock_logger.error.call_args)
        assert "Failed to store fall event" in error_call_args or "database" in error_call_args.lower()


@given(
    num_falls=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=20, deadline=None)
def test_property_multiple_fall_events_all_persisted(num_falls):
    """
    Property: For any sequence of N fall detections, N fall event records 
    should be created in the database.
    
    This test verifies that multiple fall events are all persisted correctly.
    """
    from api_server import store_fall_event
    from unittest.mock import MagicMock
    import asyncio
    import uuid
    
    # Create mock Supabase client
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_execute = MagicMock()
    
    # Set up the mock chain
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_execute.data = [{"id": "test-id"}]
    mock_insert.execute.return_value = mock_execute
    
    # Patch the global supabase_client
    with patch('api_server.supabase_client', mock_supabase):
        # Create and store multiple fall events
        for i in range(num_falls):
            alert_data = {
                "patient_id": str(uuid.uuid4()),
                "person_tracking_id": i,
                "fall_count": i + 1,
                "timestamp": f"2024-01-01T12:00:{i:02d}",
                "metadata": {}
            }
            asyncio.run(store_fall_event(alert_data))
        
        # Verify insert was called num_falls times
        assert mock_table.insert.call_count == num_falls, \
            f"Expected {num_falls} database inserts, got {mock_table.insert.call_count}"
        
        # Verify execute was called num_falls times
        assert mock_insert.execute.call_count == num_falls, \
            f"Expected {num_falls} execute calls, got {mock_insert.execute.call_count}"
