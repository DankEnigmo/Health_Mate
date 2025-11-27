"""
Property-based tests for WebSocketManager

**Feature: fall-detection-integration, Property 10: Fall event broadcast**
Tests that for any fall detection event, the event is broadcast to all connected WebSocket clients.
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from websocket_manager import WebSocketManager


# Test Property 10: Fall event broadcast
# **Validates: Requirements 3.4**

@given(
    num_connections=st.integers(min_value=1, max_value=10),
    patient_id=st.text(min_size=1, max_size=50),
    person_tracking_id=st.integers(min_value=0, max_value=100),
    fall_count=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=100)
def test_property_fall_event_broadcast_to_all_clients(
    num_connections,
    patient_id,
    person_tracking_id,
    fall_count
):
    """
    Property: For any fall detection event, the event should be broadcast to 
    all connected WebSocket clients.
    
    This test verifies that when a fall alert is broadcast, all connected clients
    receive the message with the correct data.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_connections = []
        for i in range(num_connections):
            mock_ws = AsyncMock()
            mock_ws.send_text = AsyncMock()
            mock_connections.append(mock_ws)
        
        # Add connections to manager
        for mock_ws in mock_connections:
            await manager.connect(mock_ws)
        
        # Verify all connections were added
        assert manager.get_connection_count() == num_connections
        
        # Create fall alert data
        alert_data = {
            "patient_id": patient_id,
            "person_tracking_id": person_tracking_id,
            "fall_count": fall_count,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "bounding_box": {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 50.0,
                    "height": 100.0
                },
                "fps": 30.0
            }
        }
        
        # Broadcast fall alert
        await manager.broadcast_fall_alert(alert_data)
        
        # Verify all connections received the message
        for mock_ws in mock_connections:
            mock_ws.send_text.assert_called_once()
            
            # Get the message that was sent
            call_args = mock_ws.send_text.call_args
            sent_message = call_args[0][0]
            
            # Parse the JSON message
            message_dict = json.loads(sent_message)
            
            # Verify message structure
            assert message_dict["type"] == "fall_detected"
            assert "data" in message_dict
            
            # Verify data fields
            data = message_dict["data"]
            assert data["patient_id"] == patient_id
            assert data["person_tracking_id"] == person_tracking_id
            assert data["fall_count"] == fall_count
            assert "timestamp" in data
            assert "metadata" in data
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


@given(
    patient_id=st.text(min_size=1, max_size=50),
    person_tracking_id=st.integers(min_value=0, max_value=100),
    fall_count=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=100)
def test_property_fall_alert_contains_required_fields(
    patient_id,
    person_tracking_id,
    fall_count
):
    """
    Property: For any fall alert broadcast, the message must contain all required fields:
    timestamp, person_tracking_id, fall_count, and patient_id.
    
    This validates Requirements 2.1 and 3.4.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create a mock WebSocket connection
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        # Add connection to manager
        await manager.connect(mock_ws)
        
        # Create fall alert data
        alert_data = {
            "patient_id": patient_id,
            "person_tracking_id": person_tracking_id,
            "fall_count": fall_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast fall alert
        await manager.broadcast_fall_alert(alert_data)
        
        # Verify message was sent
        mock_ws.send_text.assert_called_once()
        
        # Get the message that was sent
        call_args = mock_ws.send_text.call_args
        sent_message = call_args[0][0]
        
        # Parse the JSON message
        message_dict = json.loads(sent_message)
        
        # Verify all required fields are present
        assert "type" in message_dict
        assert message_dict["type"] == "fall_detected"
        
        data = message_dict["data"]
        assert "patient_id" in data
        assert "person_tracking_id" in data
        assert "fall_count" in data
        assert "timestamp" in data
        
        # Verify field values match
        assert data["patient_id"] == patient_id
        assert data["person_tracking_id"] == person_tracking_id
        assert data["fall_count"] == fall_count
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_broadcast_with_no_connections():
    """
    Test that broadcasting with no active connections doesn't raise an error.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Verify no connections
        assert manager.get_connection_count() == 0
        
        # Create fall alert data
        alert_data = {
            "patient_id": "test_patient",
            "person_tracking_id": 1,
            "fall_count": 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast should not raise an error
        await manager.broadcast_fall_alert(alert_data)
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_connection_management():
    """
    Test that connections are properly added and removed.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()
        
        # Add connections
        await manager.connect(mock_ws1)
        assert manager.get_connection_count() == 1
        
        await manager.connect(mock_ws2)
        assert manager.get_connection_count() == 2
        
        await manager.connect(mock_ws3)
        assert manager.get_connection_count() == 3
        
        # Remove a connection
        manager.disconnect(mock_ws2)
        assert manager.get_connection_count() == 2
        
        # Remove another connection
        manager.disconnect(mock_ws1)
        assert manager.get_connection_count() == 1
        
        # Remove last connection
        manager.disconnect(mock_ws3)
        assert manager.get_connection_count() == 0
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_send_personal_message():
    """
    Test that personal messages can be sent to specific clients.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        # Add connections
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        # Send personal message to first client
        test_message = "Hello, client 1!"
        await manager.send_personal_message(test_message, mock_ws1)
        
        # Verify only first client received the message
        mock_ws1.send_text.assert_called_once_with(test_message)
        mock_ws2.send_text.assert_not_called()
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_broadcast_generic_message():
    """
    Test that generic messages can be broadcast to all clients.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        # Add connections
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        # Broadcast generic message
        await manager.broadcast_message("system_status", {"status": "ok"})
        
        # Verify both clients received the message
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        
        # Verify message structure
        call_args = mock_ws1.send_text.call_args
        sent_message = call_args[0][0]
        message_dict = json.loads(sent_message)
        
        assert message_dict["type"] == "system_status"
        assert message_dict["data"]["status"] == "ok"
        assert "timestamp" in message_dict
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_failed_send_removes_connection():
    """
    Test that connections are removed when sending fails.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        
        mock_ws_good.send_text = AsyncMock()
        mock_ws_bad.send_text = AsyncMock(side_effect=Exception("Connection lost"))
        
        # Add connections
        await manager.connect(mock_ws_good)
        await manager.connect(mock_ws_bad)
        
        assert manager.get_connection_count() == 2
        
        # Broadcast message
        alert_data = {
            "patient_id": "test",
            "person_tracking_id": 1,
            "fall_count": 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.broadcast_fall_alert(alert_data)
        
        # Verify bad connection was removed
        assert manager.get_connection_count() == 1
        assert mock_ws_good in manager.active_connections
        assert mock_ws_bad not in manager.active_connections
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_heartbeat_mechanism():
    """
    Test that heartbeat messages are sent periodically.
    """
    async def run_test():
        # Create WebSocketManager with short heartbeat interval
        manager = WebSocketManager(heartbeat_interval=1)
        
        # Create mock WebSocket connection
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        # Add connection
        await manager.connect(mock_ws)
        
        # Wait for heartbeat to be sent
        await asyncio.sleep(1.5)
        
        # Verify heartbeat was sent
        assert mock_ws.send_text.call_count >= 1
        
        # Check that at least one call was a heartbeat
        heartbeat_found = False
        for call in mock_ws.send_text.call_args_list:
            message = call[0][0]
            message_dict = json.loads(message)
            if message_dict.get("type") == "heartbeat":
                heartbeat_found = True
                break
        
        assert heartbeat_found, "Heartbeat message was not sent"
        
        # Cleanup
        await manager.shutdown()
    
    # Run the async test
    asyncio.run(run_test())


def test_shutdown_closes_all_connections():
    """
    Test that shutdown properly closes all connections.
    """
    async def run_test():
        # Create WebSocketManager
        manager = WebSocketManager(heartbeat_interval=60)
        
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()
        
        # Add connections
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        await manager.connect(mock_ws3)
        
        assert manager.get_connection_count() == 3
        
        # Shutdown
        await manager.shutdown()
        
        # Verify all connections were closed
        mock_ws1.close.assert_called_once()
        mock_ws2.close.assert_called_once()
        mock_ws3.close.assert_called_once()
        
        # Verify connection count is zero
        assert manager.get_connection_count() == 0
    
    # Run the async test
    asyncio.run(run_test())
