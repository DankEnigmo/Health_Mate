"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";
import { FallDetectionPlayer } from "@/components/fall-detection/FallDetectionPlayer";
import { FallAlertsDisplay } from "@/components/fall-detection/FallAlertsDisplay";

const CaregiverFallDetection: React.FC = () => {
  const { patientId, loading: patientLoading, error: patientError } = useCaregiverPatient();
  const [patientName, setPatientName] = useState<string | null>(null);

  useEffect(() => {
    const fetchPatientProfile = async () => {
      if (patientId) {
        const { data, error } = await supabase
          .from('profiles')
          .select('first_name, last_name')
          .eq('id', patientId)
          .single();

        if (error) {
          console.error("Error fetching patient profile:", error);
          showError("Failed to fetch patient profile.");
          setPatientName(null);
        } else if (data) {
          setPatientName(`${data.first_name || ''} ${data.last_name || ''}`.trim());
        }
      } else {
        setPatientName(null);
      }
    };

    if (!patientLoading && !patientError) {
      fetchPatientProfile();
    }
  }, [patientId, patientLoading, patientError]);

  if (patientLoading) {
    return <p className="text-muted-foreground p-4">Loading patient information...</p>;
  }

  if (patientError) {
    return <p className="text-destructive p-4">Error: {patientError}</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-primary sansation-bold">
          Fall Detection Monitor for {patientName || "Your Patient"}
        </h1>
        <p className="text-lg text-muted-foreground mt-2">
          Monitor live video feed with real-time fall detection and alerts for {patientName || "your patient"}.
        </p>
      </div>

      {!patientId ? (
        <Card className="p-4 border rounded-lg bg-card">
          <CardContent>
            <p className="text-destructive">
              No patient assigned. Please assign a patient to monitor fall detection.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Video Player - Takes up 2 columns on large screens */}
          <div className="lg:col-span-2">
            <FallDetectionPlayer 
              patientId={patientId}
              showMetrics={true}
            />
          </div>
          
          {/* Fall Alerts - Takes up 1 column on large screens */}
          <div className="lg:col-span-1">
            <FallAlertsDisplay 
              patientId={patientId}
              maxAlerts={10}
              showHistory={true}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default CaregiverFallDetection;