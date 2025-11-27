import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { MedicationSchedule } from '@/types/medication';
import { toast } from 'sonner';

interface UseMedicationScheduleReturn {
  medications: MedicationSchedule[];
  loading: boolean;
  error: string | null;
  createMedication: (med: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'>) => Promise<void>;
  updateMedication: (med: MedicationSchedule) => Promise<void>;
  deleteMedication: (medId: string) => Promise<void>;
  refetch: () => Promise<void>;
}

export function useMedicationSchedule(patientId: string): UseMedicationScheduleReturn {
  const [medications, setMedications] = useState<MedicationSchedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMedications = useCallback(async () => {
    if (!patientId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const { data, error: fetchError } = await supabase
        .from('medications')
        .select('*')
        .eq('patient_id', patientId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (fetchError) throw fetchError;

      setMedications(data || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch medication schedules';
      setError(errorMessage);
      console.error('Error fetching medication schedules:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchMedications();
  }, [fetchMedications]);

  // Subscribe to realtime updates
  useEffect(() => {
    if (!patientId) return;

    const channel = supabase
      .channel(`medications:${patientId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'medications',
          filter: `patient_id=eq.${patientId}`,
        },
        (payload) => {
          if (payload.eventType === 'INSERT') {
            const newMed = payload.new as MedicationSchedule;
            if (newMed.is_active) {
              setMedications((prev) => [newMed, ...prev]);
            }
          } else if (payload.eventType === 'UPDATE') {
            const updatedMed = payload.new as MedicationSchedule;
            setMedications((prev) => {
              if (!updatedMed.is_active) {
                return prev.filter((med) => med.id !== updatedMed.id);
              }
              return prev.map((med) =>
                med.id === updatedMed.id ? updatedMed : med
              );
            });
          } else if (payload.eventType === 'DELETE') {
            setMedications((prev) => prev.filter((med) => med.id !== payload.old.id));
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [patientId]);

  const createMedication = useCallback(
    async (med: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'>) => {
      try {
        const { error: insertError } = await supabase
          .from('medications')
          .insert([med]);

        if (insertError) throw insertError;

        toast.success('Medication schedule created successfully');
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to create medication schedule';
        toast.error(errorMessage);
        throw err;
      }
    },
    []
  );

  const updateMedication = useCallback(async (med: MedicationSchedule) => {
    try {
      const { error: updateError } = await supabase
        .from('medications')
        .update({
          name: med.name,
          dosage: med.dosage,
          frequency: med.frequency,
          times: med.times,
          instructions: med.instructions,
          start_date: med.start_date,
          end_date: med.end_date,
          is_active: med.is_active,
          updated_at: new Date().toISOString(),
        })
        .eq('id', med.id);

      if (updateError) throw updateError;

      toast.success('Medication schedule updated successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update medication schedule';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  const deleteMedication = useCallback(async (medId: string) => {
    try {
      // Soft delete by setting is_active to false
      const { error: updateError } = await supabase
        .from('medications')
        .update({ is_active: false, updated_at: new Date().toISOString() })
        .eq('id', medId);

      if (updateError) throw updateError;

      toast.success('Medication schedule deleted successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete medication schedule';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  return {
    medications,
    loading,
    error,
    createMedication,
    updateMedication,
    deleteMedication,
    refetch: fetchMedications,
  };
}
