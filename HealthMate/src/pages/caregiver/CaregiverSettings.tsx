"use client";

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";

const CaregiverSettings = () => {
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
    return <p className="text-muted-foreground p-4">Loading settings...</p>;
  }

  if (patientError) {
    return <p className="text-destructive p-4">Error: {patientError}</p>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Settings for {patientName || "Your Patient"}</h1>
      <p className="text-lg text-muted-foreground">
        Configure your preferences and application settings for {patientName || "your patient"}'s care.
      </p>
      {!patientId ? (
        <Card className="p-4 border rounded-lg bg-card">
          <p className="text-destructive">No patient assigned. Please assign a patient to manage their settings.</p>
        </Card>
      ) : (
        <div className="p-4 border rounded-lg bg-card">
          <p className="text-muted-foreground">
            User settings and application configurations for {patientName || "your patient"} will be available here.
          </p>
        </div>
      )}
    </div>
  );
};

export default CaregiverSettings;