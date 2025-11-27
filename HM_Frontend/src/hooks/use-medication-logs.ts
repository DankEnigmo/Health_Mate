import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { MedicationLog, MedicationWithStatus, AdherenceStats, MedicationSchedule } from '@/types/medication';
import { toast } from 'sonner';
import { startOfDay, endOfDay, subDays, parseISO, format, isAfter, isBefore } from 'date-fns';

interface UseMedicationLogsReturn {
  logs: MedicationLog[];
  todaysMedications: MedicationWithStatus[];
  loading: boolean;
  error: string | null;
  logMedication: (medicationId: string, scheduledTime: string) => Promise<void>;
  adherenceRate: number;
  adherenceStats: AdherenceStats;
  refetch: () => Promise<void>;
}

export function useMedicationLogs(
  patientId: string,
  days: number = 7
): UseMedicationLogsReturn {
  const [logs, setLogs] = useState<MedicationLog[]>([]);
  const [todaysMedications, setTodaysMedications] = useState<MedicationWithStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [adherenceStats, setAdherenceStats] = useState<AdherenceStats>({
    totalScheduled: 0,
    totalTaken: 0,
    totalMissed: 0,
    adherenceRate: 0,
    recentLogs: [],
  });

  const fetchLogs = useCallback(async () => {
    if (!patientId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const startDate = subDays(new Date(), days);
      const endDate = new Date();

      // Fetch medication logs for the specified period
      const { data: logsData, error: logsError } = await supabase
        .from('medication_logs')
        .select('*')
        .eq('patient_id', patientId)
        .gte('scheduled_time', startDate.toISOString())
        .lte('scheduled_time', endDate.toISOString())
        .order('scheduled_time', { ascending: false });

      if (logsError) throw logsError;

      setLogs(logsData || []);

      // Calculate adherence statistics
      const totalScheduled = logsData?.length || 0;
      const totalTaken = logsData?.filter(log => log.status === 'taken').length || 0;
      const totalMissed = logsData?.filter(log => log.status === 'missed').length || 0;
      const adherenceRate = totalScheduled > 0 ? (totalTaken / totalScheduled) * 100 : 0;

      setAdherenceStats({
        totalScheduled,
        totalTaken,
        totalMissed,
        adherenceRate,
        recentLogs: logsData || [],
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch medication logs';
      setError(errorMessage);
      console.error('Error fetching medication logs:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId, days]);

  const fetchTodaysMedications = useCallback(async () => {
    if (!patientId) return;

    try {
      // Fetch active medication schedules
      const { data: medications, error: medsError } = await supabase
        .from('medications')
        .select('*')
        .eq('patient_id', patientId)
        .eq('is_active', true);

      if (medsError) throw medsError;

      if (!medications || medications.length === 0) {
        setTodaysMedications([]);
        return;
      }

      const today = new Date();
      const todayStart = startOfDay(today);
      const todayEnd = endOfDay(today);

      // Fetch today's logs
      const { data: todaysLogs, error: logsError } = await supabase
        .from('medication_logs')
        .select('*')
        .eq('patient_id', patientId)
        .gte('scheduled_time', todayStart.toISOString())
        .lte('scheduled_time', todayEnd.toISOString());

      if (logsError) throw logsError;

      // Build medications with status for today
      const medicationsWithStatus: MedicationWithStatus[] = medications.map((med: MedicationSchedule) => {
        const medLogs = (todaysLogs || []).filter(log => log.medication_id === med.id);
        
        // Find next scheduled time for today
        const todayDateStr = format(today, 'yyyy-MM-dd');
        const scheduledTimes = med.times.map(time => `${todayDateStr}T${time}:00`);
        
        const nextScheduledTime = scheduledTimes.find(time => {
          const timeDate = parseISO(time);
          return isAfter(timeDate, today) || Math.abs(timeDate.getTime() - today.getTime()) < 3600000; // Within 1 hour
        });

        // Find last taken time
        const takenLogs = medLogs.filter(log => log.status === 'taken' && log.taken_time);
        const lastTakenTime = takenLogs.length > 0 
          ? takenLogs.sort((a, b) => new Date(b.taken_time!).getTime() - new Date(a.taken_time!).getTime())[0].taken_time
          : undefined;

        return {
          ...med,
          todaysLogs: medLogs,
          nextScheduledTime,
          lastTakenTime,
        };
      });

      setTodaysMedications(medicationsWithStatus);

    } catch (err) {
      console.error('Error fetching today\'s medications:', err);
    }
  }, [patientId]);

  useEffect(() => {
    fetchLogs();
    fetchTodaysMedications();
  }, [fetchLogs, fetchTodaysMedications]);

  // Subscribe to realtime updates for medication logs
  useEffect(() => {
    if (!patientId) return;

    const channel = supabase
      .channel(`medication_logs:${patientId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'medication_logs',
          filter: `patient_id=eq.${patientId}`,
        },
        () => {
          // Refetch when logs change
          fetchLogs();
          fetchTodaysMedications();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [patientId, fetchLogs, fetchTodaysMedications]);

  const logMedication = useCallback(
    async (medicationId: string, scheduledTime: string) => {
      try {
        const now = new Date().toISOString();

        // Check if a log already exists for this medication and scheduled time
        const { data: existingLogs, error: checkError } = await supabase
          .from('medication_logs')
          .select('*')
          .eq('medication_id', medicationId)
          .eq('patient_id', patientId)
          .eq('scheduled_time', scheduledTime);

        if (checkError) throw checkError;

        if (existingLogs && existingLogs.length > 0) {
          // Update existing log
          const { error: updateError } = await supabase
            .from('medication_logs')
            .update({
              status: 'taken',
              taken_time: now,
            })
            .eq('id', existingLogs[0].id);

          if (updateError) throw updateError;
        } else {
          // Create new log
          const { error: insertError } = await supabase
            .from('medication_logs')
            .insert([{
              medication_id: medicationId,
              patient_id: patientId,
              scheduled_time: scheduledTime,
              taken_time: now,
              status: 'taken',
            }]);

          if (insertError) throw insertError;
        }

        toast.success('Medication marked as taken');
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to log medication';
        toast.error(errorMessage);
        throw err;
      }
    },
    [patientId]
  );

  return {
    logs,
    todaysMedications,
    loading,
    error,
    logMedication,
    adherenceRate: adherenceStats.adherenceRate,
    adherenceStats,
    refetch: fetchLogs,
  };
}
