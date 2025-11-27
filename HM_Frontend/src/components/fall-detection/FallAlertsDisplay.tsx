import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useFallAlerts, FallEvent } from '@/hooks/use-fall-alerts';
import {
  AlertTriangle,
  BellRing,
  CheckCircle,
  Clock,
  User,
  RefreshCw,
  AlertCircle,
  Eye,
  CheckCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';

interface FallAlertsDisplayProps {
  patientId: string;
  maxAlerts?: number;
  showHistory?: boolean;
  className?: string;
}

/**
 * FallAlertsDisplay Component
 * 
 * Displays real-time fall alerts for a patient with the ability to mark them as reviewed or resolved.
 * 
 * Features:
 * - Real-time WebSocket connection for instant alerts
 * - Display list of recent fall alerts
 * - Show alert details (timestamp, fall count, person ID)
 * - Allow marking alerts as reviewed or resolved
 * - Connection status indicator
 * - Error handling and display
 * - Chronological ordering of alerts
 * 
 * @param patientId - The ID of the patient to monitor
 * @param maxAlerts - Maximum number of alerts to display (default: 10)
 * @param showHistory - Whether to show all alerts or just new ones (default: true)
 * @param className - Additional CSS classes
 */
export const FallAlertsDisplay: React.FC<FallAlertsDisplayProps> = ({
  patientId,
  maxAlerts = 10,
  showHistory = true,
  className,
}) => {
  const { alerts, isConnected, error, markAsReviewed, markAsResolved } = useFallAlerts(
    patientId,
    maxAlerts
  );
  
  const [loadingAlertId, setLoadingAlertId] = useState<string | null>(null);
  
  /**
   * Handle marking an alert as reviewed
   */
  const handleMarkAsReviewed = async (alertId: string) => {
    setLoadingAlertId(alertId);
    try {
      await markAsReviewed(alertId);
    } catch (err) {
      console.error('Failed to mark alert as reviewed:', err);
    } finally {
      setLoadingAlertId(null);
    }
  };
  
  /**
   * Handle marking an alert as resolved
   */
  const handleMarkAsResolved = async (alertId: string) => {
    setLoadingAlertId(alertId);
    try {
      await markAsResolved(alertId);
    } catch (err) {
      console.error('Failed to mark alert as resolved:', err);
    } finally {
      setLoadingAlertId(null);
    }
  };
  
  /**
   * Filter alerts based on showHistory prop
   */
  const displayedAlerts = showHistory
    ? alerts
    : alerts.filter(alert => alert.status === 'new');
  
  /**
   * Get status badge variant
   */
  const getStatusBadgeVariant = (status: FallEvent['status']) => {
    switch (status) {
      case 'new':
        return 'destructive';
      case 'reviewed':
        return 'default';
      case 'resolved':
        return 'secondary';
      default:
        return 'default';
    }
  };
  
  /**
   * Get status icon
   */
  const getStatusIcon = (status: FallEvent['status']) => {
    switch (status) {
      case 'new':
        return <AlertCircle className="h-4 w-4" />;
      case 'reviewed':
        return <Eye className="h-4 w-4" />;
      case 'resolved':
        return <CheckCircle className="h-4 w-4" />;
      default:
        return <AlertCircle className="h-4 w-4" />;
    }
  };
  
  /**
   * Format timestamp for display
   */
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return {
        relative: formatDistanceToNow(date, { addSuffix: true }),
        absolute: date.toLocaleString(),
      };
    } catch (err) {
      return {
        relative: 'Unknown time',
        absolute: timestamp,
      };
    }
  };
  
  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-xl font-semibold flex items-center gap-2">
          <BellRing className="h-5 w-5" />
          Fall Alerts
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge
            variant={isConnected ? 'default' : 'destructive'}
            className="gap-1"
          >
            <div
              className={cn(
                'h-2 w-2 rounded-full',
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              )}
            />
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {/* Alerts List */}
        {displayedAlerts.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <CheckCircle className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No fall alerts to display</p>
            {!showHistory && (
              <p className="text-xs mt-1">All alerts have been reviewed</p>
            )}
          </div>
        ) : (
          <ScrollArea className="h-[500px] pr-4">
            <div className="space-y-3">
              {displayedAlerts.map((alert, index) => {
                const timestamp = formatTimestamp(alert.timestamp);
                const isLoading = loadingAlertId === alert.id;
                
                return (
                  <React.Fragment key={alert.id}>
                    <div
                      className={cn(
                        'p-4 rounded-lg border transition-all',
                        alert.status === 'new' &&
                          'border-destructive bg-destructive/5 ring-1 ring-destructive/20',
                        alert.status === 'reviewed' &&
                          'border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20',
                        alert.status === 'resolved' &&
                          'border-green-500 bg-green-50 dark:bg-green-950/20 opacity-75'
                      )}
                    >
                      {/* Alert Header */}
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <AlertTriangle
                            className={cn(
                              'h-5 w-5',
                              alert.status === 'new' && 'text-destructive',
                              alert.status === 'reviewed' && 'text-yellow-600',
                              alert.status === 'resolved' && 'text-green-600'
                            )}
                          />
                          <span className="font-semibold">Fall Detected</span>
                        </div>
                        <Badge
                          variant={getStatusBadgeVariant(alert.status)}
                          className="gap-1"
                        >
                          {getStatusIcon(alert.status)}
                          {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                        </Badge>
                      </div>
                      
                      {/* Alert Details */}
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <Clock className="h-4 w-4" />
                          <span title={timestamp.absolute}>{timestamp.relative}</span>
                        </div>
                        
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <User className="h-4 w-4" />
                          <span>Person ID: {alert.person_tracking_id}</span>
                        </div>
                        
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <AlertTriangle className="h-4 w-4" />
                          <span>Total Falls: {alert.fall_count}</span>
                        </div>
                        
                        {alert.metadata?.fps && (
                          <div className="flex items-center gap-2 text-muted-foreground text-xs">
                            <span>FPS: {alert.metadata.fps.toFixed(1)}</span>
                          </div>
                        )}
                      </div>
                      
                      {/* Action Buttons */}
                      {alert.status !== 'resolved' && (
                        <div className="flex gap-2 mt-4">
                          {alert.status === 'new' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleMarkAsReviewed(alert.id)}
                              disabled={isLoading}
                              className="gap-2"
                            >
                              {isLoading ? (
                                <RefreshCw className="h-4 w-4 animate-spin" />
                              ) : (
                                <Eye className="h-4 w-4" />
                              )}
                              Mark as Reviewed
                            </Button>
                          )}
                          <Button
                            variant={alert.status === 'new' ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => handleMarkAsResolved(alert.id)}
                            disabled={isLoading}
                            className="gap-2"
                          >
                            {isLoading ? (
                              <RefreshCw className="h-4 w-4 animate-spin" />
                            ) : (
                              <CheckCheck className="h-4 w-4" />
                            )}
                            Mark as Resolved
                          </Button>
                        </div>
                      )}
                    </div>
                    
                    {index < displayedAlerts.length - 1 && (
                      <Separator className="my-2" />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </ScrollArea>
        )}
        
        {/* Footer Info */}
        <div className="text-xs text-muted-foreground text-center pt-2 border-t">
          {displayedAlerts.length > 0 && (
            <p>
              Showing {displayedAlerts.length} of {alerts.length} alert
              {alerts.length !== 1 ? 's' : ''}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FallAlertsDisplay;
