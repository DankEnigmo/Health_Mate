"""
WebSocket Manager for Real-Time Fall Alerts

This module manages WebSocket connections and broadcasts fall detection alerts
to all connected clients in real-time.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

LOG = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts fall alerts to connected clients.
    Implements connection management, broadcast functionality, and heartbeat mechanism.
    """
    
    def __init__(self, heartbeat_interval: int = 30):
        """
        Initialize the WebSocketManager.
        
        Args:
            heartbeat_interval: Interval in seconds for sending heartbeat pings
        """
        self.active_connections: List[WebSocket] = []
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task = None
        LOG.info("WebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket):
        """
        Accept and store a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        LOG.info(f"New WebSocket connection established. Total connections: {len(self.active_connections)}")
        
        # Start heartbeat task if not already running
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from the active connections list.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            LOG.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
        
        # Stop heartbeat task if no connections remain
        if len(self.active_connections) == 0 and self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
    
    async def broadcast_fall_alert(self, alert_data: Dict[str, Any]):
        """
        Broadcast a fall alert to all connected WebSocket clients.
        
        Args:
            alert_data: Dictionary containing fall alert information
                Expected keys: patient_id, person_tracking_id, fall_count, timestamp, metadata
        """
        if not self.active_connections:
            LOG.debug("No active WebSocket connections to broadcast to")
            return
        
        # Create the alert message
        message = {
            "type": "fall_detected",
            "data": alert_data
        }
        
        # Convert to JSON
        message_json = json.dumps(message)
        
        LOG.info(f"Broadcasting fall alert to {len(self.active_connections)} clients")
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
                LOG.debug(f"Fall alert sent to client")
            except Exception as e:
                LOG.error(f"Error sending fall alert to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send a message to a specific WebSocket client.
        
        Args:
            message: The message to send
            websocket: The target WebSocket connection
        """
        try:
            await websocket.send_text(message)
            LOG.debug("Personal message sent to client")
        except Exception as e:
            LOG.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def _heartbeat_loop(self):
        """
        Internal method to send periodic heartbeat pings to all connected clients.
        This helps detect dead connections and keeps connections alive.
        """
        LOG.info("Heartbeat loop started")
        
        try:
            while self.active_connections:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.active_connections:
                    break
                
                LOG.debug(f"Sending heartbeat to {len(self.active_connections)} clients")
                
                # Create heartbeat message
                heartbeat_message = json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send heartbeat to all clients
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.send_text(heartbeat_message)
                    except Exception as e:
                        LOG.warning(f"Heartbeat failed for client: {e}")
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for connection in disconnected:
                    self.disconnect(connection)
        
        except asyncio.CancelledError:
            LOG.info("Heartbeat loop cancelled")
        except Exception as e:
            LOG.error(f"Error in heartbeat loop: {e}")
    
    def get_connection_count(self) -> int:
        """
        Get the number of active WebSocket connections.
        
        Returns:
            Number of active connections
        """
        return len(self.active_connections)
    
    async def broadcast_message(self, message_type: str, data: Dict[str, Any]):
        """
        Broadcast a generic message to all connected clients.
        
        Args:
            message_type: Type of message (e.g., 'fall_detected', 'system_status')
            data: Message data dictionary
        """
        if not self.active_connections:
            LOG.debug("No active WebSocket connections to broadcast to")
            return
        
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_json = json.dumps(message)
        
        LOG.info(f"Broadcasting {message_type} message to {len(self.active_connections)} clients")
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                LOG.error(f"Error broadcasting message to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def shutdown(self):
        """
        Gracefully shutdown all WebSocket connections.
        """
        LOG.info(f"Shutting down WebSocketManager with {len(self.active_connections)} active connections")
        
        # Cancel heartbeat task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection in self.active_connections[:]:  # Copy list to avoid modification during iteration
            try:
                await connection.close()
            except Exception as e:
                LOG.error(f"Error closing WebSocket connection: {e}")
            finally:
                self.disconnect(connection)
        
        LOG.info("WebSocketManager shutdown complete")
