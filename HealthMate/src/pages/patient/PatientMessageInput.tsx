"use client";

import React, { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { MessageSquareText } from "lucide-react";
import OnScreenKeyboard from "@/components/patient/OnScreenKeyboard";

const PatientMessageInput: React.FC = () => {
  const [keyboardInput, setKeyboardInput] = useState<string>("");

  const handleKeyboardKeyPress = useCallback((value: string) => {
    setKeyboardInput(value);
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary sansation-bold">Your Message Pad</h1>
      <p className="text-lg text-muted-foreground">
        Use the on-screen keyboard below to type messages or notes.
      </p>

      <Card className="mb-4"> {/* Added mb-4 for spacing */}
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xl font-medium flex items-center gap-2">
            <MessageSquareText className="h-6 w-6" /> Your Message
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Type your message here using the on-screen keyboard..."
            className="min-h-[150px] text-lg"
            value={keyboardInput}
            onChange={(e) => setKeyboardInput(e.target.value)}
          />
          <p className="text-sm text-muted-foreground mt-2">
            This is your primary communication area.
          </p>
        </CardContent>
      </Card>

      {/* On-Screen Keyboard rendered only on this page */}
      <OnScreenKeyboard onKeyPress={handleKeyboardKeyPress} currentInput={keyboardInput} />
    </div>
  );
};

export default PatientMessageInput;