"use client";

import React from "react";
import { MedicineScheduleManager } from "@/components/medicine/MedicineScheduleManager";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { useSupabase } from "@/components/SessionContextProvider";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Pill, AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const CaregiverMedications: React.FC = () => {
  const { session } = useSupabase();
  const { patientId, loading: patientLoading, error: patientError } = useCaregiverPatient();
  const caregiverId = session?.user?.id;

  if (patientLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-primary sansation-bold">Medication Management</h1>
          <p className="text-lg text-muted-foreground mt-2">
            Loading patient information...
          </p>
        </div>
      </div>
    );
  }

  if (patientError) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-primary sansation-bold">Medication Management</h1>
          <p className="text-lg text-muted-foreground mt-2">
            Manage medication schedules for your patient
          </p>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            Failed to load patient information: {patientError}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!patientId) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-primary sansation-bold">Medication Management</h1>
          <p className="text-lg text-muted-foreground mt-2">
            Manage medication schedules for your patient
          </p>
        </div>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Pill className="h-5 w-5" />
              No Patient Assigned
            </CardTitle>
            <CardDescription>
              You currently do not have a patient assigned. Please ensure a patient is linked to your caregiver account.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Once a patient is assigned, you'll be able to create and manage their medication schedules here.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!caregiverId) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-primary sansation-bold">Medication Management</h1>
          <p className="text-lg text-muted-foreground mt-2">
            Manage medication schedules for your patient
          </p>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Authentication Error</AlertTitle>
          <AlertDescription>
            Unable to verify your identity. Please log in again.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-primary sansation-bold">Medication Management</h1>
        <p className="text-lg text-muted-foreground mt-2">
          Create and manage medication schedules for your patient
        </p>
      </div>

      <MedicineScheduleManager patientId={patientId} caregiverId={caregiverId} />
    </div>
  );
};

export default CaregiverMedications;
