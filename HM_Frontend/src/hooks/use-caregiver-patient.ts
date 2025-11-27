"use client";

import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";
import { useSupabase } from "@/components/SessionContextProvider"; // Re-added useSupabase import

interface CaregiverPatient {
  patient_id: string;
}

export function useCaregiverPatient() {
  const { session } = useSupabase(); // Get session from context
  const [patientId, setPatientId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAssignedPatient = async () => {
      setLoading(true);
      setError(null);
      if (!session?.user?.id) {
        setPatientId(null);
        setLoading(false);
        setError("No authenticated user found.");
        return;
      }

      try {
        const caregiverId = session.user.id;

        const { data, error: fetchError } = await supabase
          .from('caregiver_patients')
          .select('patient_id')
          .eq('caregiver_id', caregiverId)
          .single();

        if (fetchError && fetchError.code !== 'PGRST116') { // PGRST116 means no rows found
          throw fetchError;
        }

        if (data) {
          setPatientId(data.patient_id);
        } else {
          setPatientId(null);
          showError("No patient assigned to this caregiver. Please assign a patient.");
        }
      } catch (err: any) {
        console.error("Error fetching assigned patient:", err);
        setError(err.message || "Failed to fetch assigned patient.");
        showError(err.message || "Failed to fetch assigned patient.");
      } finally {
        setLoading(false);
      }
    };

    fetchAssignedPatient();

    // Re-added realtime subscription for caregiver_patients changes
    const channel = supabase
      .channel('caregiver_patient_changes')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'caregiver_patients', filter: `caregiver_id=eq.${session?.user?.id}` },
        (payload) => {
          if (payload.eventType === 'INSERT' || payload.eventType === 'UPDATE') {
            const newAssignment = payload.new as CaregiverPatient;
            setPatientId(newAssignment.patient_id);
          } else if (payload.eventType === 'DELETE') {
            setPatientId(null);
            showError("Your patient assignment has been removed.");
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [session]); // Depend on session to re-run when auth state changes

  return { patientId, loading, error };
}