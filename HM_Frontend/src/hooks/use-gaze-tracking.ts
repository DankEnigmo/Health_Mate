import { useState, useEffect, useCallback, useRef } from 'react';

// Types for gaze tracking
export interface GazeCoordinates {
  x: number;
  y: number;
  confidence: number;
  quality: string;
}

export interface IrisCenters {
  left: { x: number; y: number };
  right: { x: number; y: number };
  mid: { x: number; y: number };
}

export interface GazeData extends GazeCoordinates {
  success: boolean;
  error?: string;
  irisCenters?: IrisCenters;
}

export interface CalibrationPoint {
  x: number;
  y: number;
}

export interface CalibrationStatus {
  isCalibrating: boolean;
  currentPoint: CalibrationPoint | null;
  pointIndex: number;
  totalPoints: number;
  samplesCollected: number;
  samplesPerPoint: number;
  isCalibrated: boolean;
}

export interface GazeTrackingStats {
  framesProcessed: number;
  framesDropped: number;
  effectiveFps: number;
  calibrationPoints: number;
  isCalibrated: boolean;
}

export interface UseGazeTrackingOptions {
  backendUrl?: string;
  frameRate?: number;
  jpegQuality?: number;
  samplesPerCalibrationPoint?: number;
  autoConnect?: boolean;
}

export interface UseGazeTrackingReturn {
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;

  // Tracking state
  isTracking: boolean;
  gazeData: GazeData | null;

  // Calibration state
  calibrationStatus: CalibrationStatus;

  // Camera state
  isCameraActive: boolean;
  cameraError: string | null;
  videoRef: React.RefObject<HTMLVideoElement>;

  // Actions
  connect: () => Promise<void>;
  disconnect: () => void;
  startTracking: () => void;
  stopTracking: () => void;
  startCalibration: () => void;
  cancelCalibration: () => void;
  collectCalibrationSample: (targetX: number, targetY: number) => void;
  resetCalibration: () => void;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  getStats: () => void;
  stats: GazeTrackingStats | null;
}

// Default calibration points (9-point grid)
const DEFAULT_CALIBRATION_POINTS: CalibrationPoint[] = [
  { x: 0.1, y: 0.1 },   // Top-left
  { x: 0.5, y: 0.1 },   // Top-center
  { x: 0.9, y: 0.1 },   // Top-right
  { x: 0.1, y: 0.5 },   // Middle-left
  { x: 0.5, y: 0.5 },   // Center
  { x: 0.9, y: 0.5 },   // Middle-right
  { x: 0.1, y: 0.9 },   // Bottom-left
  { x: 0.5, y: 0.9 },   // Bottom-center
  { x: 0.9, y: 0.9 },   // Bottom-right
];

const DEFAULT_SAMPLES_PER_POINT = 10;

export function useGazeTracking(options: UseGazeTrackingOptions = {}): UseGazeTrackingReturn {
  const {
    backendUrl = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000',
    frameRate = 15,
    jpegQuality = 0.8,
    samplesPerCalibrationPoint = DEFAULT_SAMPLES_PER_POINT,
    autoConnect = false,
  } = options;

  // WebSocket URL
  const wsUrl = backendUrl.replace(/^http/, 'ws') + '/ws/gaze-tracking';

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trackingIntervalRef = useRef<number | null>(null);
  const calibrationSamplesRef = useRef<number>(0);

  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Tracking state
  const [isTracking, setIsTracking] = useState(false);
  const [gazeData, setGazeData] = useState<GazeData | null>(null);

  // Camera state
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Calibration state
  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>({
    isCalibrating: false,
    currentPoint: null,
    pointIndex: 0,
    totalPoints: DEFAULT_CALIBRATION_POINTS.length,
    samplesCollected: 0,
    samplesPerPoint: samplesPerCalibrationPoint,
    isCalibrated: false,
  });

  // Stats
  const [stats, setStats] = useState<GazeTrackingStats | null>(null);

  // Initialize canvas for frame capture
  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
  }, []);

  // Capture frame from video
  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) {
      return null;
    }

    // Set canvas size to video size (or scaled down for performance)
    const targetWidth = 640;
    const targetHeight = 480;
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Convert to base64 JPEG
    return canvas.toDataURL('image/jpeg', jpegQuality);
  }, [jpegQuality]);

  // Handle WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'gaze_data':
          if (data.success) {
            setGazeData({
              success: true,
              x: data.x,
              y: data.y,
              confidence: data.confidence,
              quality: data.quality,
              irisCenters: data.iris_centers ? {
                left: data.iris_centers.left,
                right: data.iris_centers.right,
                mid: data.iris_centers.mid,
              } : undefined,
            });
          } else {
            setGazeData({
              success: false,
              x: 0,
              y: 0,
              confidence: 0,
              quality: 'none',
              error: data.error,
            });
          }
          break;

        case 'calibration_response':
          if (data.success) {
            calibrationSamplesRef.current = data.samples_collected;

            setCalibrationStatus(prev => ({
              ...prev,
              samplesCollected: data.samples_collected,
              isCalibrated: data.is_calibrated,
            }));

            // Check if we collected enough samples for current point
            if (data.samples_collected >= samplesPerCalibrationPoint) {
              // Move to next point
              setCalibrationStatus(prev => {
                const nextIndex = prev.pointIndex + 1;
                if (nextIndex >= DEFAULT_CALIBRATION_POINTS.length) {
                  // Calibration complete
                  return {
                    ...prev,
                    isCalibrating: false,
                    currentPoint: null,
                    isCalibrated: true,
                  };
                }
                // Move to next point
                calibrationSamplesRef.current = 0;
                return {
                  ...prev,
                  pointIndex: nextIndex,
                  currentPoint: DEFAULT_CALIBRATION_POINTS[nextIndex],
                  samplesCollected: 0,
                };
              });
            }
          }
          break;

        case 'calibration_reset':
          setCalibrationStatus(prev => ({
            ...prev,
            isCalibrated: false,
            samplesCollected: 0,
            pointIndex: 0,
          }));
          calibrationSamplesRef.current = 0;
          break;

        case 'stats':
          setStats({
            framesProcessed: data.data.frames_processed,
            framesDropped: data.data.frames_dropped,
            effectiveFps: data.data.effective_fps,
            calibrationPoints: data.data.calibration_points,
            isCalibrated: data.data.is_calibrated,
          });
          break;

        case 'heartbeat':
          // Connection is alive
          break;

        case 'error':
          console.error('Gaze tracking error:', data.message);
          break;
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [samplesPerCalibrationPoint]);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);
    setConnectionError(null);

    return new Promise<void>((resolve, reject) => {
      try {
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionError(null);
          wsRef.current = ws;
          resolve();
        };

        ws.onclose = () => {
          setIsConnected(false);
          setIsTracking(false);
          wsRef.current = null;
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnectionError('Failed to connect to gaze tracking server');
          setIsConnecting(false);
          reject(new Error('WebSocket connection failed'));
        };

        ws.onmessage = handleMessage;

      } catch (error) {
        setConnectionError('Failed to create WebSocket connection');
        setIsConnecting(false);
        reject(error);
      }
    });
  }, [wsUrl, handleMessage]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (trackingIntervalRef.current) {
      clearInterval(trackingIntervalRef.current);
      trackingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsTracking(false);
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
    } catch (error) {
      console.error('Camera error:', error);
      setCameraError(
        error instanceof Error
          ? error.message
          : 'Failed to access camera. Please check permissions.'
      );
      setIsCameraActive(false);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsCameraActive(false);
  }, []);

  // Send frame for tracking
  const sendTrackingFrame = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const frame = captureFrame();
    if (!frame) return;

    wsRef.current.send(JSON.stringify({
      action: 'track',
      frame: frame,
    }));
  }, [captureFrame]);

  // Start tracking
  const startTracking = useCallback(() => {
    if (!isConnected || !isCameraActive) {
      console.warn('Cannot start tracking: not connected or camera not active');
      return;
    }

    setIsTracking(true);

    // Start sending frames at specified frame rate
    const interval = 1000 / frameRate;
    trackingIntervalRef.current = window.setInterval(sendTrackingFrame, interval);
  }, [isConnected, isCameraActive, frameRate, sendTrackingFrame]);

  // Stop tracking
  const stopTracking = useCallback(() => {
    if (trackingIntervalRef.current) {
      clearInterval(trackingIntervalRef.current);
      trackingIntervalRef.current = null;
    }
    setIsTracking(false);
    setGazeData(null);
  }, []);

  // Start calibration
  const startCalibration = useCallback(() => {
    if (!isConnected || !isCameraActive) {
      console.warn('Cannot start calibration: not connected or camera not active');
      return;
    }

    calibrationSamplesRef.current = 0;

    setCalibrationStatus({
      isCalibrating: true,
      currentPoint: DEFAULT_CALIBRATION_POINTS[0],
      pointIndex: 0,
      totalPoints: DEFAULT_CALIBRATION_POINTS.length,
      samplesCollected: 0,
      samplesPerPoint: samplesPerCalibrationPoint,
      isCalibrated: false,
    });
  }, [isConnected, isCameraActive, samplesPerCalibrationPoint]);

  // Cancel calibration
  const cancelCalibration = useCallback(() => {
    setCalibrationStatus(prev => ({
      ...prev,
      isCalibrating: false,
      currentPoint: null,
    }));
  }, []);

  // Collect calibration sample
  const collectCalibrationSample = useCallback((targetX: number, targetY: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const frame = captureFrame();
    if (!frame) return;

    wsRef.current.send(JSON.stringify({
      action: 'calibrate',
      frame: frame,
      target_x: targetX,
      target_y: targetY,
    }));
  }, [captureFrame]);

  // Reset calibration
  const resetCalibration = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    wsRef.current.send(JSON.stringify({
      action: 'reset_calibration',
    }));

    calibrationSamplesRef.current = 0;

    setCalibrationStatus({
      isCalibrating: false,
      currentPoint: null,
      pointIndex: 0,
      totalPoints: DEFAULT_CALIBRATION_POINTS.length,
      samplesCollected: 0,
      samplesPerPoint: samplesPerCalibrationPoint,
      isCalibrated: false,
    });
  }, [samplesPerCalibrationPoint]);

  // Get stats
  const getStats = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    wsRef.current.send(JSON.stringify({
      action: 'get_stats',
    }));
  }, []);

  // Auto connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
      stopCamera();
    };
  }, [autoConnect]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // Connection state
    isConnected,
    isConnecting,
    connectionError,

    // Tracking state
    isTracking,
    gazeData,

    // Calibration state
    calibrationStatus,

    // Camera state
    isCameraActive,
    cameraError,
    videoRef,

    // Actions
    connect,
    disconnect,
    startTracking,
    stopTracking,
    startCalibration,
    cancelCalibration,
    collectCalibrationSample,
    resetCalibration,
    startCamera,
    stopCamera,
    getStats,
    stats,
  };
}

export default useGazeTracking;
