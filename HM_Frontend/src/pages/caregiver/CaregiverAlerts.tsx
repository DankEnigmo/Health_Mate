"use client";

import React, { useEffect, useState } from "react";
import EmergencyAlertsDisplay from "@/components/caregiver/EmergencyAlertsDisplay";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";
import { Card } from "@/components/ui/card";

const CaregiverAlerts = () => {
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
    return <p className="text-muted-foreground p-4">Loading alerts...</p>;
  }

  if (patientError) {
    return <p className="text-destructive p-4">Error: {patientError}</p>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Alerts for {patientName || "Your Patient"}</h1>
      <p className="text-lg text-muted-foreground">
        Review real-time fall detection alerts and other important notifications for {patientName || "your patient"}.
      </p>
      {!patientId ? (
        <Card className="p-4 border rounded-lg bg-card">
          <p className="text-destructive">No patient assigned. Please assign a patient to view their alerts.</p>
        </Card>
      ) : (
        <EmergencyAlertsDisplay />
      )}
    </div>
  );
};

export default CaregiverAlerts;