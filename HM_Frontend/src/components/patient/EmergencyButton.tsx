"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";
import { showSuccess, showError } from "@/utils/toast";
import { useSupabase } from "@/components/SessionContextProvider"; // Re-added useSupabase import

const EmergencyButton: React.FC = () => {
  const { session } = useSupabase(); // Re-added session check

  const handleEmergencyClick = async () => {
    if (!session) {
      showError("You must be logged in to send an emergency alert.");
      return;
    }
    console.log("EmergencyButton: Sending alert with access token:", session.access_token ? "present" : "missing");
    try {
      const response = await fetch('https://ruqdkozfcnevgxuaitkk.supabase.co/functions/v1/send-emergency-alert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`, // Re-added Authorization header
        },
      });

      const result = await response.json();

      if (response.ok) {
        showSuccess(result.message || "Emergency alert sent!");
      } else {
        showError(result.error || "Failed to send emergency alert.");
      }
    } catch (error) {
      console.error("Error sending emergency alert:", error);
      showError("An unexpected error occurred while sending the alert.");
    }
  };

  return (
    <Button
      variant="destructive"
      size="icon"
      className="fixed bottom-4 right-4 h-24 w-24 rounded-full shadow-lg flex items-center justify-center text-xl font-bold z-50 animate-pulse"
      onClick={handleEmergencyClick}
      aria-label="Emergency SOS Button"
    >
      <AlertTriangle className="h-12 w-12" />
      <span className="sr-only">SOS</span>
    </Button>
  );
};

export default EmergencyButton;