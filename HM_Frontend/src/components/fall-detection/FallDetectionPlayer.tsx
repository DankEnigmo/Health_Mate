import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useFallDetectionStream } from '@/hooks/use-fall-detection-stream';
import { 
  Video, 
  VideoOff, 
  RefreshCw, 
  Activity, 
  Clock, 
  AlertTriangle,
  Eye,
  EyeOff
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface FallEvent {
  id: string;
  patient_id: string;
  person_tracking_id: number;
  fall_count: number;
  timestamp: string;
  metadata?: any;
}

interface StreamMetrics {
  fps: number;
  latency: number;
  fallCount: number;
  isProcessing: boolean;
}

interface FallDetectionPlayerProps {
  patientId: string;
  onFallDetected?: (fallData: FallEvent) => void;
  showMetrics?: boolean;
}

/**
 * FallDetectionPlayer Component
 * 
 * Displays live MJPEG video stream from the fall detection backend
 * with performance metrics and error handling.
 * 
 * Features:
 * - Real-time video streaming with fall detection overlays
 * - Performance metrics display (FPS, latency, fall count)
 * - Automatic reconnection with exponential backoff
 * - Loading and error states
 * - Manual reconnection control
 * 
 * @param patientId - The ID of the patient to monitor
 * @param onFallDetected - Optional callback when a fall is detected
 * @param showMetrics - Whether to show performance metrics (default: true)
 */
export const FallDetectionPlayer: React.FC<FallDetectionPlayerProps> = ({
  patientId,
  onFallDetected,
  showMetrics = true,
}) => {
  const { streamUrl, isConnected, error, reconnect } = useFallDetectionStream(patientId);
  
  const [isLoading, setIsLoading] = useState(true);
  const [metrics, setMetrics] = useState<StreamMetrics>({
    fps: 0,
    latency: 0,
    fallCount: 0,
    isProcessing: false,
  });
  const [metricsVisible, setMetricsVisible] = useState(showMetrics);
  const [imageError, setImageError] = useState(false);
  
  const imgRef = useRef<HTMLImageElement>(null);
  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Get backend URL for stats endpoint
  const backendUrl = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000';
  
  /**
   * Fetch performance metrics from backend
   */
  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/stats`);
      if (response.ok) {
        const data = await response.json();
        setMetrics({
          fps: data.fps || 0,
          latency: data.latency || 0,
          fallCount: data.fall_count || 0,
          isProcessing: data.is_processing || false,
        });
      }
    } catch (err) {
      // Silently fail - metrics are not critical
      console.error('Failed to fetch metrics:', err);
    }
  };
  
  /**
   * Handle image load event
   */
  const handleImageLoad = () => {
    setIsLoading(false);
    setImageError(false);
  };
  
  /**
   * Handle image error event
   */
  const handleImageError = () => {
    setIsLoading(false);
    setImageError(true);
  };
  
  /**
   * Handle manual reconnection
   */
  const handleReconnect = () => {
    setIsLoading(true);
    setImageError(false);
    reconnect();
  };
  
  /**
   * Toggle metrics visibility
   */
  const toggleMetrics = () => {
    setMetricsVisible(prev => !prev);
  };
  
  /**
   * Set up metrics polling
   */
  useEffect(() => {
    if (isConnected && !error) {
      // Fetch metrics immediately
      fetchMetrics();
      
      // Poll metrics every 2 seconds
      statsIntervalRef.current = setInterval(fetchMetrics, 2000);
    }
    
    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
      }
    };
  }, [isConnected, error, backendUrl]);
  
  /**
   * Check for performance warnings
   */
  const hasPerformanceWarning = metrics.fps > 0 && metrics.fps < 10;
  const hasHighLatency = metrics.latency > 200;
  
  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-xl font-semibold flex items-center gap-2">
          <Video className="h-5 w-5" />
          Live Fall Detection Monitor
        </CardTitle>
        <div className="flex items-center gap-2">
          {showMetrics && (
            <Button
              variant="outline"
              size="sm"
              onClick={toggleMetrics}
              className="gap-2"
            >
              {metricsVisible ? (
                <>
                  <EyeOff className="h-4 w-4" />
                  Hide Metrics
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4" />
                  Show Metrics
                </>
              )}
            </Button>
          )}
          <Badge
            variant={isConnected ? 'default' : 'destructive'}
            className="gap-1"
          >
            <div className={cn(
              'h-2 w-2 rounded-full',
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            )} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReconnect}
                className="ml-4"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        )}
        
        {/* Performance Warning */}
        {(hasPerformanceWarning || hasHighLatency) && !error && (
          <Alert variant="default" className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <AlertDescription className="text-yellow-800 dark:text-yellow-200">
              {hasPerformanceWarning && `Low FPS detected (${metrics.fps.toFixed(1)} FPS). `}
              {hasHighLatency && `High latency detected (${metrics.latency.toFixed(0)}ms). `}
              Video quality may be degraded.
            </AlertDescription>
          </Alert>
        )}
        
        {/* Video Stream Container */}
        <div className="relative w-full bg-black rounded-lg overflow-hidden aspect-video">
          {isLoading && !imageError && (
            <div className="absolute inset-0 flex items-center justify-center bg-muted">
              <div className="text-center space-y-2">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Loading video stream...</p>
              </div>
            </div>
          )}
          
          {imageError && (
            <div className="absolute inset-0 flex items-center justify-center bg-muted">
              <div className="text-center space-y-4">
                <VideoOff className="h-12 w-12 mx-auto text-muted-foreground" />
                <div className="space-y-2">
                  <p className="text-sm font-medium">Video stream unavailable</p>
                  <p className="text-xs text-muted-foreground">
                    Unable to load the video feed
                  </p>
                </div>
                <Button onClick={handleReconnect} variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reconnect
                </Button>
              </div>
            </div>
          )}
          
          <img
            ref={imgRef}
            src={streamUrl}
            alt="Fall Detection Video Stream"
            className={cn(
              'w-full h-full object-contain',
              (isLoading || imageError) && 'hidden'
            )}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
          
          {/* Metrics Overlay */}
          {metricsVisible && isConnected && !isLoading && !imageError && (
            <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-3 space-y-2 text-white">
              <div className="flex items-center gap-2 text-sm">
                <Activity className="h-4 w-4" />
                <span className="font-medium">FPS:</span>
                <span className={cn(
                  'font-mono',
                  metrics.fps < 10 && 'text-yellow-400',
                  metrics.fps >= 10 && 'text-green-400'
                )}>
                  {metrics.fps.toFixed(1)}
                </span>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4" />
                <span className="font-medium">Latency:</span>
                <span className={cn(
                  'font-mono',
                  metrics.latency > 200 && 'text-yellow-400',
                  metrics.latency <= 200 && 'text-green-400'
                )}>
                  {metrics.latency.toFixed(0)}ms
                </span>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="h-4 w-4" />
                <span className="font-medium">Falls:</span>
                <span className={cn(
                  'font-mono font-bold',
                  metrics.fallCount > 0 ? 'text-red-400' : 'text-green-400'
                )}>
                  {metrics.fallCount}
                </span>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <div className={cn(
                  'h-2 w-2 rounded-full',
                  metrics.isProcessing ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
                )} />
                <span className="text-xs">
                  {metrics.isProcessing ? 'Processing' : 'Idle'}
                </span>
              </div>
            </div>
          )}
        </div>
        
        {/* Connection Info */}
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <span>Patient ID: {patientId}</span>
          </div>
          {!isConnected && !error && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReconnect}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Reconnect
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FallDetectionPlayer;
