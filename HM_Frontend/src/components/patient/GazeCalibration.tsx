"use client";

import React, { useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { X, Eye, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { CalibrationStatus } from "@/hooks/use-gaze-tracking";

interface GazeCalibrationProps {
  calibrationStatus: CalibrationStatus;
  onCollectSample: (targetX: number, targetY: number) => void;
  onCancel: () => void;
  onComplete?: () => void;
}

// Calibration point positions (9-point grid)
const CALIBRATION_POINTS = [
  { x: 0.1, y: 0.1, label: "Top Left" },
  { x: 0.5, y: 0.1, label: "Top Center" },
  { x: 0.9, y: 0.1, label: "Top Right" },
  { x: 0.1, y: 0.5, label: "Middle Left" },
  { x: 0.5, y: 0.5, label: "Center" },
  { x: 0.9, y: 0.5, label: "Middle Right" },
  { x: 0.1, y: 0.9, label: "Bottom Left" },
  { x: 0.5, y: 0.9, label: "Bottom Center" },
  { x: 0.9, y: 0.9, label: "Bottom Right" },
];

const SAMPLE_COLLECTION_INTERVAL = 100; // ms between sample collections

const GazeCalibration: React.FC<GazeCalibrationProps> = ({
  calibrationStatus,
  onCollectSample,
  onCancel,
  onComplete,
}) => {
  const [isCollecting, setIsCollecting] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);

  const {
    isCalibrating,
    currentPoint,
    pointIndex,
    totalPoints,
    samplesCollected,
    samplesPerPoint,
    isCalibrated,
  } = calibrationStatus;

  // Calculate overall progress
  const overallProgress = ((pointIndex * samplesPerPoint + samplesCollected) / (totalPoints * samplesPerPoint)) * 100;
  const pointProgress = (samplesCollected / samplesPerPoint) * 100;

  // Auto-collect samples when user is looking at calibration point
  useEffect(() => {
    if (!isCalibrating || !currentPoint || !isCollecting) {
      return;
    }

    const interval = setInterval(() => {
      if (samplesCollected < samplesPerPoint) {
        onCollectSample(currentPoint.x, currentPoint.y);
      }
    }, SAMPLE_COLLECTION_INTERVAL);

    return () => clearInterval(interval);
  }, [isCalibrating, currentPoint, isCollecting, samplesCollected, samplesPerPoint, onCollectSample]);

  // Handle calibration complete
  useEffect(() => {
    if (isCalibrated && onComplete) {
      onComplete();
    }
  }, [isCalibrated, onComplete]);

  // Start collecting samples when user clicks/focuses on calibration point
  const handleStartCollecting = useCallback(() => {
    setIsCollecting(true);
    setShowInstructions(false);
  }, []);

  // Stop collecting samples
  const handleStopCollecting = useCallback(() => {
    setIsCollecting(false);
  }, []);

  // If calibration is complete, show success message
  if (isCalibrated && !isCalibrating) {
    return (
      <div className="fixed inset-0 z-50 bg-background/95 flex items-center justify-center">
        <Card className="w-full max-w-md mx-4">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-600">
              <CheckCircle2 className="h-6 w-6" />
              Calibration Complete!
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              Gaze tracking has been calibrated successfully. You can now use eye tracking to type on the keyboard.
            </p>
            <Button onClick={onComplete} className="w-full">
              Start Using Gaze Tracking
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // If not calibrating, don't render
  if (!isCalibrating) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-background">
      {/* Cancel button */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-4 right-4 z-10"
        onClick={onCancel}
      >
        <X className="h-6 w-6" />
      </Button>

      {/* Instructions overlay */}
      {showInstructions && (
        <div className="absolute inset-0 z-20 bg-background/95 flex items-center justify-center">
          <Card className="w-full max-w-lg mx-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-6 w-6" />
                Gaze Calibration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-semibold">Instructions:</h3>
                <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                  <li>Keep your head still and at a comfortable distance from the screen</li>
                  <li>Look at the glowing circle that appears on the screen</li>
                  <li>Click and hold (or press Space) while looking at each point</li>
                  <li>Hold until the progress circle fills completely</li>
                  <li>Repeat for all 9 calibration points</li>
                </ol>
              </div>
              <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950 rounded-lg">
                <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0" />
                <p className="text-sm text-amber-700 dark:text-amber-300">
                  For best results, ensure good lighting and position your face clearly in front of the camera.
                </p>
              </div>
              <Button onClick={() => setShowInstructions(false)} className="w-full">
                Begin Calibration
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Progress bar at top */}
      <div className="absolute top-0 left-0 right-0 p-4 bg-background/80 backdrop-blur">
        <div className="max-w-2xl mx-auto space-y-2">
          <div className="flex justify-between text-sm">
            <span>Point {pointIndex + 1} of {totalPoints}</span>
            <span>{Math.round(overallProgress)}% Complete</span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>
      </div>

      {/* Calibration points grid */}
      <div className="absolute inset-0 p-16">
        {CALIBRATION_POINTS.map((point, index) => {
          const isCurrentPoint = index === pointIndex;
          const isCompleted = index < pointIndex;
          const isFuture = index > pointIndex;

          return (
            <div
              key={index}
              className="absolute transform -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${point.x * 100}%`,
                top: `${point.y * 100}%`,
              }}
            >
              {/* Calibration point */}
              <button
                className={cn(
                  "relative rounded-full transition-all duration-300 focus:outline-none",
                  isCurrentPoint && "w-20 h-20",
                  isCompleted && "w-8 h-8",
                  isFuture && "w-6 h-6 opacity-30"
                )}
                onMouseDown={isCurrentPoint ? handleStartCollecting : undefined}
                onMouseUp={isCurrentPoint ? handleStopCollecting : undefined}
                onMouseLeave={isCurrentPoint ? handleStopCollecting : undefined}
                onTouchStart={isCurrentPoint ? handleStartCollecting : undefined}
                onTouchEnd={isCurrentPoint ? handleStopCollecting : undefined}
                onKeyDown={(e) => {
                  if (isCurrentPoint && e.key === ' ') {
                    e.preventDefault();
                    handleStartCollecting();
                  }
                }}
                onKeyUp={(e) => {
                  if (isCurrentPoint && e.key === ' ') {
                    e.preventDefault();
                    handleStopCollecting();
                  }
                }}
                disabled={!isCurrentPoint}
                aria-label={`Calibration point ${index + 1}: ${point.label}`}
              >
                {/* Completed point */}
                {isCompleted && (
                  <div className="w-full h-full rounded-full bg-green-500 flex items-center justify-center">
                    <CheckCircle2 className="h-5 w-5 text-white" />
                  </div>
                )}

                {/* Future point */}
                {isFuture && (
                  <div className="w-full h-full rounded-full bg-muted border-2 border-muted-foreground/20" />
                )}

                {/* Current point */}
                {isCurrentPoint && (
                  <>
                    {/* Outer pulse animation */}
                    <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />

                    {/* Progress ring */}
                    <svg className="absolute inset-0 w-full h-full -rotate-90">
                      <circle
                        cx="50%"
                        cy="50%"
                        r="45%"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="4"
                        className="text-muted"
                      />
                      <circle
                        cx="50%"
                        cy="50%"
                        r="45%"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="4"
                        strokeDasharray={`${pointProgress * 2.83} 283`}
                        className="text-primary transition-all duration-100"
                      />
                    </svg>

                    {/* Inner circle */}
                    <div
                      className={cn(
                        "absolute inset-4 rounded-full transition-all duration-150",
                        isCollecting
                          ? "bg-primary scale-110"
                          : "bg-primary/70"
                      )}
                    />

                    {/* Center dot */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div
                        className={cn(
                          "w-3 h-3 rounded-full bg-white transition-transform",
                          isCollecting && "scale-150"
                        )}
                      />
                    </div>
                  </>
                )}
              </button>

              {/* Label for current point */}
              {isCurrentPoint && (
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-4 text-center whitespace-nowrap">
                  <p className="text-sm font-medium">{point.label}</p>
                  <p className="text-xs text-muted-foreground">
                    {isCollecting ? "Keep looking..." : "Click and hold while looking here"}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Bottom status bar */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-background/80 backdrop-blur">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className={cn(
              "h-5 w-5 transition-colors",
              isCollecting ? "text-primary" : "text-muted-foreground"
            )} />
            <span className="text-sm">
              {isCollecting
                ? `Collecting samples... (${samplesCollected}/${samplesPerPoint})`
                : "Look at the highlighted point and click to collect samples"
              }
            </span>
          </div>
          <Button variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
};

export default GazeCalibration;
