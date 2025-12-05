"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  MessageSquareText,
  Eye,
  EyeOff,
  Video,
  VideoOff,
  Settings2,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import OnScreenKeyboard from "@/components/patient/OnScreenKeyboard";
import GazeCalibration from "@/components/patient/GazeCalibration";
import GazeCursor from "@/components/patient/GazeCursor";
import { useGazeTracking } from "@/hooks/use-gaze-tracking";
import { cn } from "@/lib/utils";

const PatientMessageInput: React.FC = () => {
  const [keyboardInput, setKeyboardInput] = useState<string>("");
  const [gazeTrackingEnabled, setGazeTrackingEnabled] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [dwellTime, setDwellTime] = useState(800);

  // Initialize gaze tracking hook
  const {
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
  } = useGazeTracking({
    frameRate: 15,
    jpegQuality: 0.8,
  });

  const handleKeyboardKeyPress = useCallback((value: string) => {
    setKeyboardInput(value);
  }, []);

  // Handle enabling/disabling gaze tracking
  const handleToggleGazeTracking = useCallback(async () => {
    if (gazeTrackingEnabled) {
      // Disable gaze tracking
      stopTracking();
      setGazeTrackingEnabled(false);
    } else {
      // Enable gaze tracking
      try {
        // Connect if not connected
        if (!isConnected) {
          await connect();
        }

        // Start camera if not active
        if (!isCameraActive) {
          await startCamera();
        }

        // If calibrated, start tracking immediately
        if (calibrationStatus.isCalibrated) {
          startTracking();
          setGazeTrackingEnabled(true);
        } else {
          // Need to calibrate first
          startCalibration();
        }
      } catch (error) {
        console.error("Failed to enable gaze tracking:", error);
      }
    }
  }, [
    gazeTrackingEnabled,
    isConnected,
    isCameraActive,
    calibrationStatus.isCalibrated,
    connect,
    startCamera,
    startTracking,
    stopTracking,
    startCalibration,
  ]);

  // Handle calibration complete
  const handleCalibrationComplete = useCallback(() => {
    startTracking();
    setGazeTrackingEnabled(true);
  }, [startTracking]);

  // Handle recalibration
  const handleRecalibrate = useCallback(() => {
    stopTracking();
    resetCalibration();
    startCalibration();
  }, [stopTracking, resetCalibration, startCalibration]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopTracking();
      stopCamera();
      disconnect();
    };
  }, [stopTracking, stopCamera, disconnect]);

  // Get connection status badge
  const getConnectionStatus = () => {
    if (isConnecting) {
      return (
        <Badge variant="secondary" className="gap-1">
          <RefreshCw className="h-3 w-3 animate-spin" />
          Connecting...
        </Badge>
      );
    }
    if (isConnected) {
      return (
        <Badge variant="default" className="gap-1 bg-green-600">
          <Wifi className="h-3 w-3" />
          Connected
        </Badge>
      );
    }
    return (
      <Badge variant="destructive" className="gap-1">
        <WifiOff className="h-3 w-3" />
        Disconnected
      </Badge>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-primary sansation-bold">
            Your Message Pad
          </h1>
          <p className="text-lg text-muted-foreground">
            Use the on-screen keyboard below to type messages or notes.
          </p>
        </div>

        {/* Gaze Tracking Controls */}
        <div className="flex items-center gap-2">
          {isConnected && getConnectionStatus()}

          <Button
            variant={gazeTrackingEnabled ? "default" : "outline"}
            size="sm"
            onClick={handleToggleGazeTracking}
            disabled={isConnecting}
            className="gap-2"
          >
            {gazeTrackingEnabled ? (
              <>
                <Eye className="h-4 w-4" />
                Gaze On
              </>
            ) : (
              <>
                <EyeOff className="h-4 w-4" />
                Gaze Off
              </>
            )}
          </Button>

          {isConnected && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Error Messages */}
      {connectionError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Connection Error:</strong> {connectionError}
            <Button
              variant="link"
              className="ml-2 h-auto p-0"
              onClick={connect}
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {cameraError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Camera Error:</strong> {cameraError}
            <Button
              variant="link"
              className="ml-2 h-auto p-0"
              onClick={startCamera}
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Settings Panel */}
      {showSettings && isConnected && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Gaze Tracking Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Camera Preview */}
            <div className="flex gap-4">
              <div className="flex-1">
                <h4 className="text-sm font-medium mb-2">Camera Preview</h4>
                <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    autoPlay
                    playsInline
                    muted
                  />
                  {!isCameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <VideoOff className="h-8 w-8 text-muted-foreground" />
                    </div>
                  )}
                </div>
                <div className="flex gap-2 mt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={isCameraActive ? stopCamera : startCamera}
                    className="gap-1"
                  >
                    {isCameraActive ? (
                      <>
                        <VideoOff className="h-4 w-4" />
                        Stop Camera
                      </>
                    ) : (
                      <>
                        <Video className="h-4 w-4" />
                        Start Camera
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Calibration Status */}
              <div className="flex-1">
                <h4 className="text-sm font-medium mb-2">Calibration</h4>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    {calibrationStatus.isCalibrated ? (
                      <>
                        <CheckCircle2 className="h-5 w-5 text-green-600" />
                        <span className="text-sm text-green-600">
                          Calibrated
                        </span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-5 w-5 text-amber-600" />
                        <span className="text-sm text-amber-600">
                          Not Calibrated
                        </span>
                      </>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRecalibrate}
                    disabled={!isCameraActive || !isConnected}
                    className="gap-1"
                  >
                    <RefreshCw className="h-4 w-4" />
                    {calibrationStatus.isCalibrated
                      ? "Recalibrate"
                      : "Calibrate Now"}
                  </Button>
                </div>

                {/* Dwell Time Setting */}
                <div className="mt-4">
                  <h4 className="text-sm font-medium mb-2">
                    Dwell Time: {dwellTime}ms
                  </h4>
                  <input
                    type="range"
                    min="400"
                    max="1500"
                    step="100"
                    value={dwellTime}
                    onChange={(e) => setDwellTime(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Faster (400ms)</span>
                    <span>Slower (1500ms)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Tracking Status */}
            {isTracking && gazeData && (
              <div className="text-xs text-muted-foreground bg-muted p-2 rounded">
                <strong>Tracking Status:</strong>{" "}
                {gazeData.success ? (
                  <>
                    Position: ({(gazeData.x * 100).toFixed(1)}%,{" "}
                    {(gazeData.y * 100).toFixed(1)}%) | Confidence:{" "}
                    {(gazeData.confidence * 100).toFixed(0)}% | Quality:{" "}
                    {gazeData.quality}
                  </>
                ) : (
                  <span className="text-amber-600">
                    Face tracking lost - please look at the camera
                  </span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Hidden video element for gaze tracking (when settings not shown) */}
      {!showSettings && (
        <video ref={videoRef} className="hidden" autoPlay playsInline muted />
      )}

      {/* Message Input Card */}
      <Card className="mb-4">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xl font-medium flex items-center gap-2">
            <MessageSquareText className="h-6 w-6" /> Your Message
          </CardTitle>
          {gazeTrackingEnabled && isTracking && (
            <Badge
              variant="outline"
              className={cn(
                "gap-1",
                gazeData?.success
                  ? "border-green-500 text-green-600"
                  : "border-amber-500 text-amber-600",
              )}
            >
              <Eye
                className={cn("h-3 w-3", gazeData?.success && "animate-pulse")}
              />
              {gazeData?.success ? "Tracking" : "No Face"}
            </Badge>
          )}
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Type your message here using the on-screen keyboard..."
            className="min-h-[150px] text-lg"
            value={keyboardInput}
            onChange={(e) => setKeyboardInput(e.target.value)}
          />
          <p className="text-sm text-muted-foreground mt-2">
            {gazeTrackingEnabled
              ? "Look at the keys below to type using your eyes."
              : "This is your primary communication area."}
          </p>
        </CardContent>
      </Card>

      {/* On-Screen Keyboard with Gaze Support */}
      <OnScreenKeyboard
        onKeyPress={handleKeyboardKeyPress}
        currentInput={keyboardInput}
        gazeData={gazeData}
        gazeEnabled={gazeTrackingEnabled && isTracking}
        dwellTime={dwellTime}
      />

      {/* Gaze Cursor Overlay */}
      {gazeTrackingEnabled && isTracking && (
        <GazeCursor
          gazeData={gazeData}
          isVisible={gazeData?.success ?? false}
          size={40}
          showTrail={true}
        />
      )}

      {/* Calibration Overlay */}
      <GazeCalibration
        calibrationStatus={calibrationStatus}
        onCollectSample={collectCalibrationSample}
        onCancel={cancelCalibration}
        onComplete={handleCalibrationComplete}
      />
    </div>
  );
};

export default PatientMessageInput;
