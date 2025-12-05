"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUp, Delete, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GazeData } from "@/hooks/use-gaze-tracking";

interface OnScreenKeyboardProps {
  onKeyPress: (value: string) => void;
  currentInput: string;
  gazeData?: GazeData | null;
  gazeEnabled?: boolean;
  dwellTime?: number; // Time in ms to trigger key selection
  onGazeSelect?: (key: string) => void;
}

interface KeyPosition {
  key: string;
  rect: DOMRect | null;
}

const keyboardLayout = {
  default: [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["z", "x", "c", "v", "b", "n", "m"],
  ],
  shift: [
    ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
  ],
};

const DEFAULT_DWELL_TIME = 800; // ms

const OnScreenKeyboard: React.FC<OnScreenKeyboardProps> = ({
  onKeyPress,
  currentInput,
  gazeData,
  gazeEnabled = false,
  dwellTime = DEFAULT_DWELL_TIME,
  onGazeSelect,
}) => {
  const [shiftActive, setShiftActive] = useState(false);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [dwellProgress, setDwellProgress] = useState(0);
  const [gazeKey, setGazeKey] = useState<string | null>(null);

  const keyRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const dwellTimerRef = useRef<number | null>(null);
  const dwellStartTimeRef = useRef<number | null>(null);
  const lastGazeKeyRef = useRef<string | null>(null);
  const keyboardRef = useRef<HTMLDivElement>(null);

  const activeLayout = shiftActive
    ? keyboardLayout.shift
    : keyboardLayout.default;

  // Register key ref
  const setKeyRef = useCallback(
    (key: string, element: HTMLButtonElement | null) => {
      if (element) {
        keyRefs.current.set(key, element);
      } else {
        keyRefs.current.delete(key);
      }
    },
    [],
  );

  // Find which key the gaze is on
  const findGazedKey = useCallback(
    (gazeX: number, gazeY: number): string | null => {
      if (!keyboardRef.current) return null;

      const screenX = gazeX * window.innerWidth;
      const screenY = gazeY * window.innerHeight;

      // Check each key's bounding rect
      for (const [key, element] of keyRefs.current.entries()) {
        const rect = element.getBoundingClientRect();

        // Add some padding for easier selection
        const padding = 5;
        if (
          screenX >= rect.left - padding &&
          screenX <= rect.right + padding &&
          screenY >= rect.top - padding &&
          screenY <= rect.bottom + padding
        ) {
          return key;
        }
      }

      return null;
    },
    [],
  );

  // Handle key click (manual or gaze-triggered)
  const handleKeyClick = useCallback(
    (key: string) => {
      let newValue = currentInput;
      if (key === "Shift") {
        setShiftActive((prev) => !prev);
        return;
      } else if (key === "Backspace") {
        newValue = newValue.slice(0, -1);
      } else if (key === "Space") {
        newValue += " ";
      } else {
        newValue += key;
      }
      onKeyPress(newValue);

      // Notify gaze select callback if provided
      if (gazeEnabled && onGazeSelect) {
        onGazeSelect(key);
      }
    },
    [currentInput, onKeyPress, gazeEnabled, onGazeSelect],
  );

  // Handle gaze-based dwell selection
  useEffect(() => {
    if (!gazeEnabled || !gazeData?.success) {
      // Reset when gaze tracking is disabled or lost
      setGazeKey(null);
      setDwellProgress(0);
      if (dwellTimerRef.current) {
        cancelAnimationFrame(dwellTimerRef.current);
        dwellTimerRef.current = null;
      }
      dwellStartTimeRef.current = null;
      lastGazeKeyRef.current = null;
      return;
    }

    const currentGazedKey = findGazedKey(gazeData.x, gazeData.y);
    setGazeKey(currentGazedKey);

    // If gaze moved to a different key, reset dwell timer
    if (currentGazedKey !== lastGazeKeyRef.current) {
      lastGazeKeyRef.current = currentGazedKey;
      dwellStartTimeRef.current = currentGazedKey ? Date.now() : null;
      setDwellProgress(0);

      if (dwellTimerRef.current) {
        cancelAnimationFrame(dwellTimerRef.current);
        dwellTimerRef.current = null;
      }
    }

    // If gazing at a key, start/continue dwell timer
    if (currentGazedKey && dwellStartTimeRef.current) {
      const updateDwell = () => {
        if (
          !dwellStartTimeRef.current ||
          lastGazeKeyRef.current !== currentGazedKey
        ) {
          return;
        }

        const elapsed = Date.now() - dwellStartTimeRef.current;
        const progress = Math.min(100, (elapsed / dwellTime) * 100);
        setDwellProgress(progress);

        if (progress >= 100) {
          // Trigger key press
          handleKeyClick(currentGazedKey);

          // Reset for next selection
          dwellStartTimeRef.current = Date.now() + 200; // Small cooldown
          setDwellProgress(0);
        } else {
          dwellTimerRef.current = requestAnimationFrame(updateDwell);
        }
      };

      dwellTimerRef.current = requestAnimationFrame(updateDwell);
    }

    return () => {
      if (dwellTimerRef.current) {
        cancelAnimationFrame(dwellTimerRef.current);
      }
    };
  }, [gazeEnabled, gazeData, dwellTime, findGazedKey, handleKeyClick]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (dwellTimerRef.current) {
        cancelAnimationFrame(dwellTimerRef.current);
      }
    };
  }, []);

  // Render a key button with gaze support
  const renderKey = (key: string, isSpecialKey: boolean = false) => {
    const isGazeTarget = gazeEnabled && gazeKey === key;
    const isHovered = hoveredKey === key;

    return (
      <Button
        key={key}
        ref={(el) => setKeyRef(key, el)}
        variant="secondary"
        className={cn(
          "relative flex-1 h-16 text-2xl font-bold p-2 sm:h-20 sm:text-3xl transition-all duration-150",
          isSpecialKey &&
            key === "Shift" &&
            shiftActive &&
            "bg-primary text-primary-foreground",
          isSpecialKey && key === "Space" && "flex-grow-[3]",
          isGazeTarget && "ring-4 ring-primary ring-offset-2 scale-105",
          isHovered && !isGazeTarget && "bg-secondary/80",
        )}
        onClick={() => handleKeyClick(key)}
        onMouseEnter={() => setHoveredKey(key)}
        onMouseLeave={() => setHoveredKey(null)}
      >
        {/* Dwell progress indicator */}
        {isGazeTarget && dwellProgress > 0 && (
          <div
            className="absolute inset-0 bg-primary/30 rounded-md transition-all duration-75"
            style={{
              clipPath: `inset(${100 - dwellProgress}% 0 0 0)`,
            }}
          />
        )}

        {/* Circular progress for gaze */}
        {isGazeTarget && dwellProgress > 0 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox="0 0 100 100"
          >
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeDasharray={`${dwellProgress * 2.83} 283`}
              className="text-primary -rotate-90 origin-center"
              style={{ transition: "stroke-dasharray 50ms linear" }}
            />
          </svg>
        )}

        {/* Key content */}
        <span className="relative z-10">
          {key === "Shift" ? (
            <ArrowUp className="h-8 w-8" />
          ) : key === "Backspace" ? (
            <Delete className="h-8 w-8" />
          ) : (
            key
          )}
        </span>
      </Button>
    );
  };

  return (
    <div ref={keyboardRef} className="w-full bg-card border-t p-4 shadow-lg">
      {/* Gaze tracking indicator */}
      {gazeEnabled && (
        <div className="flex items-center justify-center gap-2 mb-3 text-sm text-muted-foreground">
          <Eye className="h-4 w-4 text-primary animate-pulse" />
          <span>Gaze tracking active - Look at a key to select it</span>
        </div>
      )}

      <div className="flex flex-col gap-2 max-w-4xl mx-auto">
        {/* Letter/number rows */}
        {activeLayout.map((row, rowIndex) => (
          <div key={rowIndex} className="flex justify-center gap-2">
            {row.map((key) => renderKey(key))}
          </div>
        ))}

        {/* Bottom row with special keys */}
        <div className="flex justify-center gap-2 mt-2">
          {renderKey("Shift", true)}
          {renderKey("Space", true)}
          {renderKey("Backspace", true)}
        </div>
      </div>

      {/* Debug info for development */}
      {process.env.NODE_ENV === "development" &&
        gazeEnabled &&
        gazeData?.success && (
          <div className="mt-4 text-xs text-center text-muted-foreground">
            Gaze: ({(gazeData.x * 100).toFixed(1)}%,{" "}
            {(gazeData.y * 100).toFixed(1)}%)
            {gazeKey &&
              ` | Key: ${gazeKey} | Dwell: ${dwellProgress.toFixed(0)}%`}
          </div>
        )}
    </div>
  );
};

export default OnScreenKeyboard;
