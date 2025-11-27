"use client";

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUp, Delete } from "lucide-react";
import { cn } from "@/lib/utils";

interface OnScreenKeyboardProps {
  onKeyPress: (value: string) => void;
  currentInput: string;
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

const OnScreenKeyboard: React.FC<OnScreenKeyboardProps> = ({ onKeyPress, currentInput }) => {
  const [shiftActive, setShiftActive] = useState(false);
  const activeLayout = shiftActive ? keyboardLayout.shift : keyboardLayout.default;

  const handleKeyClick = useCallback((key: string) => {
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
  }, [currentInput, onKeyPress]);

  return (
    <div className="w-full bg-card border-t p-4 shadow-lg"> {/* Removed fixed positioning and z-index */}
      <div className="flex flex-col gap-2 max-w-4xl mx-auto">
        {activeLayout.map((row, rowIndex) => (
          <div key={rowIndex} className="flex justify-center gap-2">
            {row.map((key) => (
              <Button
                key={key}
                variant="secondary"
                className="flex-1 h-16 text-2xl font-bold p-2 sm:h-20 sm:text-3xl"
                onClick={() => handleKeyClick(key)}
              >
                {key}
              </Button>
            ))}
          </div>
        ))}
        <div className="flex justify-center gap-2 mt-2">
          <Button
            variant="secondary"
            className={cn(
              "flex-1 h-16 text-2xl font-bold p-2 sm:h-20 sm:text-3xl",
              shiftActive && "bg-primary text-primary-foreground"
            )}
            onClick={() => handleKeyClick("Shift")}
          >
            <ArrowUp className="h-8 w-8" />
          </Button>
          <Button
            variant="secondary"
            className="flex-grow-[3] h-16 text-2xl font-bold p-2 sm:h-20 sm:text-3xl"
            onClick={() => handleKeyClick("Space")}
          >
            Space
          </Button>
          <Button
            variant="secondary"
            className="flex-1 h-16 text-2xl font-bold p-2 sm:h-20 sm:text-3xl"
            onClick={() => handleKeyClick("Backspace")}
          >
            <Delete className="h-8 w-8" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default OnScreenKeyboard;