import { useState, useEffect, useCallback, useRef } from 'react';

interface UseFallDetectionStreamReturn {
  streamUrl: string;
  isConnected: boolean;
  error: string | null;
  reconnect: () => void;
}

/**
 * Custom hook for managing fall detection video stream connection
 * Implements exponential backoff reconnection logic
 * 
 * @param patientId - The ID of the patient to monitor
 * @returns Stream URL, connection status, error state, and reconnect function
 */
export function useFallDetectionStream(
  patientId: string
): UseFallDetectionStreamReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  
  // Get backend URL from environment variables
  const backendUrl = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000';
  
  // Generate stream URL for the patient
  const streamUrl = `${backendUrl}/api/video/stream?patient_id=${patientId}`;
  
  /**
   * Calculate exponential backoff delay
   * Delays: 1s, 2s, 4s, 8s, 16s, max 30s
   */
  const getReconnectDelay = useCallback((attempt: number): number => {
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
    return delay;
  }, []);
  
  /**
   * Attempt to reconnect with exponential backoff
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    const delay = getReconnectDelay(reconnectAttempt);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      setReconnectAttempt(prev => prev + 1);
      setError(null);
      
      // Create a new image to test the connection
      const img = new Image();
      img.onload = () => {
        setIsConnected(true);
        setReconnectAttempt(0);
        setError(null);
      };
      img.onerror = () => {
        setIsConnected(false);
        setError('Failed to connect to video stream');
        scheduleReconnect();
      };
      img.src = streamUrl + `&t=${Date.now()}`;
      imageRef.current = img;
    }, delay);
  }, [reconnectAttempt, streamUrl, getReconnectDelay]);
  
  /**
   * Manual reconnect function
   */
  const reconnect = useCallback(() => {
    setReconnectAttempt(0);
    setError(null);
    setIsConnected(false);
    
    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Immediately attempt connection
    const img = new Image();
    img.onload = () => {
      setIsConnected(true);
      setError(null);
    };
    img.onerror = () => {
      setIsConnected(false);
      setError('Failed to connect to video stream');
      scheduleReconnect();
    };
    img.src = streamUrl + `&t=${Date.now()}`;
    imageRef.current = img;
  }, [streamUrl, scheduleReconnect]);
  
  /**
   * Initialize connection on mount
   */
  useEffect(() => {
    reconnect();
    
    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (imageRef.current) {
        imageRef.current.onload = null;
        imageRef.current.onerror = null;
      }
    };
  }, [patientId]); // Only reconnect when patientId changes
  
  return {
    streamUrl,
    isConnected,
    error,
    reconnect,
  };
}
