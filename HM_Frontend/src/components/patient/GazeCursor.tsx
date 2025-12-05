"use client";

import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { GazeData } from "@/hooks/use-gaze-tracking";

interface GazeCursorProps {
  gazeData: GazeData | null;
  isVisible?: boolean;
  size?: number;
  showTrail?: boolean;
  trailLength?: number;
  className?: string;
}

interface TrailPoint {
  x: number;
  y: number;
  timestamp: number;
}

const GazeCursor: React.FC<GazeCursorProps> = ({
  gazeData,
  isVisible = true,
  size = 40,
  showTrail = true,
  trailLength = 5,
  className,
}) => {
  const [trail, setTrail] = useState<TrailPoint[]>([]);
  const [smoothedPosition, setSmoothedPosition] = useState<{ x: number; y: number } | null>(null);

  // Smoothing factor for cursor movement (0-1, higher = more responsive, lower = smoother)
  const smoothingFactor = 0.3;

  // Update position with smoothing
  useEffect(() => {
    if (!gazeData?.success || gazeData.x === undefined || gazeData.y === undefined) {
      return;
    }

    const targetX = gazeData.x * window.innerWidth;
    const targetY = gazeData.y * window.innerHeight;

    setSmoothedPosition((prev) => {
      if (!prev) {
        return { x: targetX, y: targetY };
      }

      // Apply exponential smoothing
      return {
        x: prev.x + (targetX - prev.x) * smoothingFactor,
        y: prev.y + (targetY - prev.y) * smoothingFactor,
      };
    });
  }, [gazeData]);

  // Update trail
  useEffect(() => {
    if (!smoothedPosition || !showTrail) {
      return;
    }

    setTrail((prev) => {
      const now = Date.now();
      const newTrail = [
        ...prev,
        { x: smoothedPosition.x, y: smoothedPosition.y, timestamp: now },
      ];

      // Keep only recent trail points
      return newTrail.slice(-trailLength);
    });
  }, [smoothedPosition, showTrail, trailLength]);

  // Don't render if not visible or no valid position
  if (!isVisible || !smoothedPosition) {
    return null;
  }

  const { x, y } = smoothedPosition;
  const confidence = gazeData?.confidence ?? 0;
  const quality = gazeData?.quality ?? "none";

  // Determine cursor color based on tracking quality
  const getQualityColor = () => {
    switch (quality) {
      case "high":
        return "bg-green-500 border-green-400";
      case "medium":
        return "bg-yellow-500 border-yellow-400";
      case "low":
        return "bg-orange-500 border-orange-400";
      default:
        return "bg-red-500 border-red-400";
    }
  };

  // Calculate opacity based on confidence
  const cursorOpacity = Math.max(0.4, Math.min(1, confidence));

  return (
    <div className="fixed inset-0 pointer-events-none z-[100]" aria-hidden="true">
      {/* Trail effect */}
      {showTrail && trail.map((point, index) => {
        const trailOpacity = ((index + 1) / trail.length) * 0.3;
        const trailSize = size * 0.5 * ((index + 1) / trail.length);

        return (
          <div
            key={point.timestamp}
            className={cn(
              "absolute rounded-full",
              getQualityColor()
            )}
            style={{
              left: point.x - trailSize / 2,
              top: point.y - trailSize / 2,
              width: trailSize,
              height: trailSize,
              opacity: trailOpacity,
              transition: "opacity 0.1s ease-out",
            }}
          />
        );
      })}

      {/* Main cursor */}
      <div
        className={cn(
          "absolute transform -translate-x-1/2 -translate-y-1/2 transition-transform duration-75",
          className
        )}
        style={{
          left: x,
          top: y,
          opacity: cursorOpacity,
        }}
      >
        {/* Outer ring */}
        <div
          className={cn(
            "absolute rounded-full border-2 animate-pulse",
            getQualityColor()
          )}
          style={{
            width: size,
            height: size,
            left: -size / 2,
            top: -size / 2,
            opacity: 0.5,
          }}
        />

        {/* Inner circle */}
        <div
          className={cn(
            "absolute rounded-full",
            getQualityColor()
          )}
          style={{
            width: size * 0.4,
            height: size * 0.4,
            left: -(size * 0.4) / 2,
            top: -(size * 0.4) / 2,
          }}
        />

        {/* Center dot */}
        <div
          className="absolute bg-white rounded-full"
          style={{
            width: size * 0.15,
            height: size * 0.15,
            left: -(size * 0.15) / 2,
            top: -(size * 0.15) / 2,
          }}
        />
      </div>

      {/* Debug info (optional - can be enabled for development) */}
      {process.env.NODE_ENV === "development" && false && (
        <div className="fixed bottom-20 left-4 bg-black/80 text-white text-xs p-2 rounded font-mono">
          <div>X: {x.toFixed(0)}px ({(gazeData?.x ?? 0).toFixed(3)})</div>
          <div>Y: {y.toFixed(0)}px ({(gazeData?.y ?? 0).toFixed(3)})</div>
          <div>Confidence: {(confidence * 100).toFixed(1)}%</div>
          <div>Quality: {quality}</div>
        </div>
      )}
    </div>
  );
};

export default GazeCursor;
