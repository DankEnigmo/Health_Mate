import { useState, useEffect, useCallback, useRef } from 'react';
import { supabase } from '@/integrations/supabase/client';

/**
 * Fall Event interface matching the database schema
 */
export interface FallEvent {
  id: string;
  patient_id: string;
  caregiver_id?: string;
  person_tracking_id: number;
  fall_count: number;
  timestamp: string;
  video_frame_url?: string;
  metadata?: {
    bounding_box?: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    confidence?: number;
    fps?: number;
  };
  status: 'new' | 'reviewed' | 'resolved';
  created_at: string;
  updated_at: string;
}

/**
 * WebSocket message format for fall alerts
 */
interface FallAlertMessage {
  type: 'fall_detected';
  data: {
    patient_id: string;
    person_tracking_id: number;
    fall_count: number;
    timestamp: string;
    metadata?: any;
  };
}

interface UseFallAlertsReturn {
  alerts: FallEvent[];
  isConnected: boolean;
  error: string | null;
  markAsReviewed: (alertId: string) => Promise<void>;
  markAsResolved: (alertId: string) => Promise<void>;
}

/**
 * Custom hook for managing fall alerts
 * 
 * Features:
 * - Connects to WebSocket for real-time fall alerts
 * - Maintains list of recent alerts
 * - Handles reconnection with exponential backoff
 * - Stores alerts in Supabase
 * - Provides functions to mark alerts as reviewed/resolved
 * 
 * @param patientId - The ID of the patient to monitor
 * @param maxAlerts - Maximum number of alerts to keep in memory (default: 50)
 * @returns Alerts list, connection status, error state, and alert management functions
 */
export function useFallAlerts(
  patientId: string,
  maxAlerts: number = 50
): UseFallAlertsReturn {
  const [alerts, setAlerts] = useState<FallEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isReconnectingRef = useRef(false);
  
  // Get backend WebSocket URL from environment variables
  const backendWsUrl = import.meta.env.VITE_BACKEND_WS_URL || 'ws://localhost:8000';
  
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
   * Fetch recent fall alerts from Supabase
   */
  const fetchRecentAlerts = useCallback(async () => {
    try {
      const { data, error: fetchError } = await supabase
        .from('fall_events')
        .select('*')
        .eq('patient_id', patientId)
        .order('timestamp', { ascending: false })
        .limit(maxAlerts);
      
      if (fetchError) {
        console.error('Error fetching fall alerts:', fetchError);
        setError('Failed to fetch fall alerts from database');
        return;
      }
      
      if (data) {
        setAlerts(data as FallEvent[]);
      }
    } catch (err) {
      console.error('Error fetching fall alerts:', err);
      setError('Failed to fetch fall alerts');
    }
  }, [patientId, maxAlerts]);
  
  /**
   * Store a new fall alert in Supabase
   */
  const storeAlert = useCallback(async (alertData: FallAlertMessage['data']) => {
    try {
      const { data, error: insertError } = await supabase
        .from('fall_events')
        .insert({
          patient_id: alertData.patient_id,
          person_tracking_id: alertData.person_tracking_id,
          fall_count: alertData.fall_count,
          timestamp: alertData.timestamp,
          metadata: alertData.metadata,
          status: 'new',
        })
        .select()
        .single();
      
      if (insertError) {
        console.error('Error storing fall alert:', insertError);
        return null;
      }
      
      return data as FallEvent;
    } catch (err) {
      console.error('Error storing fall alert:', err);
      return null;
    }
  }, []);
  
  /**
   * Connect to WebSocket
   */
  const connectWebSocket = useCallback(() => {
    if (isReconnectingRef.current) {
      return;
    }
    
    isReconnectingRef.current = true;
    
    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      const ws = new WebSocket(`${backendWsUrl}/api/ws/alerts?patient_id=${patientId}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        setReconnectAttempt(0);
        isReconnectingRef.current = false;
      };
      
      ws.onmessage = async (event) => {
        try {
          const message: FallAlertMessage = JSON.parse(event.data);
          
          if (message.type === 'fall_detected' && message.data.patient_id === patientId) {
            // Store the alert in Supabase
            const storedAlert = await storeAlert(message.data);
            
            if (storedAlert) {
              // Add to local state
              setAlerts(prev => {
                const newAlerts = [storedAlert, ...prev];
                // Keep only maxAlerts most recent
                return newAlerts.slice(0, maxAlerts);
              });
            }
          }
        } catch (err) {
          console.error('Error processing WebSocket message:', err);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
        isReconnectingRef.current = false;
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        isReconnectingRef.current = false;
        
        // Schedule reconnection with exponential backoff
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        
        const delay = getReconnectDelay(reconnectAttempt);
        reconnectTimeoutRef.current = setTimeout(() => {
          setReconnectAttempt(prev => prev + 1);
          connectWebSocket();
        }, delay);
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Error creating WebSocket:', err);
      setError('Failed to create WebSocket connection');
      isReconnectingRef.current = false;
    }
  }, [patientId, backendWsUrl, reconnectAttempt, getReconnectDelay, storeAlert, maxAlerts]);
  
  /**
   * Mark an alert as reviewed
   */
  const markAsReviewed = useCallback(async (alertId: string) => {
    try {
      const { error: updateError } = await supabase
        .from('fall_events')
        .update({ status: 'reviewed', updated_at: new Date().toISOString() })
        .eq('id', alertId);
      
      if (updateError) {
        console.error('Error marking alert as reviewed:', updateError);
        throw new Error('Failed to mark alert as reviewed');
      }
      
      // Update local state
      setAlerts(prev =>
        prev.map(alert =>
          alert.id === alertId
            ? { ...alert, status: 'reviewed' as const, updated_at: new Date().toISOString() }
            : alert
        )
      );
    } catch (err) {
      console.error('Error marking alert as reviewed:', err);
      throw err;
    }
  }, []);
  
  /**
   * Mark an alert as resolved
   */
  const markAsResolved = useCallback(async (alertId: string) => {
    try {
      const { error: updateError } = await supabase
        .from('fall_events')
        .update({ status: 'resolved', updated_at: new Date().toISOString() })
        .eq('id', alertId);
      
      if (updateError) {
        console.error('Error marking alert as resolved:', updateError);
        throw new Error('Failed to mark alert as resolved');
      }
      
      // Update local state
      setAlerts(prev =>
        prev.map(alert =>
          alert.id === alertId
            ? { ...alert, status: 'resolved' as const, updated_at: new Date().toISOString() }
            : alert
        )
      );
    } catch (err) {
      console.error('Error marking alert as resolved:', err);
      throw err;
    }
  }, []);
  
  /**
   * Initialize: Fetch recent alerts and connect to WebSocket
   */
  useEffect(() => {
    fetchRecentAlerts();
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [patientId]); // Only reconnect when patientId changes
  
  /**
   * Subscribe to Supabase realtime updates for fall_events
   * This ensures we get updates even if WebSocket is down
   */
  useEffect(() => {
    const channel = supabase
      .channel('fall_events_changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'fall_events',
          filter: `patient_id=eq.${patientId}`,
        },
        (payload) => {
          if (payload.eventType === 'INSERT') {
            const newAlert = payload.new as FallEvent;
            setAlerts(prev => {
              // Check if alert already exists (to avoid duplicates from WebSocket)
              if (prev.some(a => a.id === newAlert.id)) {
                return prev;
              }
              const newAlerts = [newAlert, ...prev];
              return newAlerts.slice(0, maxAlerts);
            });
          } else if (payload.eventType === 'UPDATE') {
            const updatedAlert = payload.new as FallEvent;
            setAlerts(prev =>
              prev.map(alert =>
                alert.id === updatedAlert.id ? updatedAlert : alert
              )
            );
          }
        }
      )
      .subscribe();
    
    return () => {
      supabase.removeChannel(channel);
    };
  }, [patientId, maxAlerts]);
  
  return {
    alerts,
    isConnected,
    error,
    markAsReviewed,
    markAsResolved,
  };
}
