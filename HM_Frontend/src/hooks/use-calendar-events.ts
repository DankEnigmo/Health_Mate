import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { CalendarEvent } from '@/types/calendar';
import { toast } from 'sonner';

interface UseCalendarEventsReturn {
  events: CalendarEvent[];
  loading: boolean;
  error: string | null;
  createEvent: (event: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'>) => Promise<void>;
  updateEvent: (event: CalendarEvent) => Promise<void>;
  deleteEvent: (eventId: string) => Promise<void>;
  refetch: () => Promise<void>;
}

export function useCalendarEvents(patientId: string): UseCalendarEventsReturn {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchEvents = useCallback(async () => {
    if (!patientId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const { data, error: fetchError } = await supabase
        .from('calendar_events')
        .select('*')
        .eq('patient_id', patientId)
        .order('event_date', { ascending: true });

      if (fetchError) throw fetchError;

      setEvents(data || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch calendar events';
      setError(errorMessage);
      console.error('Error fetching calendar events:', err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // Subscribe to realtime updates
  useEffect(() => {
    if (!patientId) return;

    const channel = supabase
      .channel(`calendar_events:${patientId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'calendar_events',
          filter: `patient_id=eq.${patientId}`,
        },
        (payload) => {
          if (payload.eventType === 'INSERT') {
            setEvents((prev) => [...prev, payload.new as CalendarEvent]);
          } else if (payload.eventType === 'UPDATE') {
            setEvents((prev) =>
              prev.map((event) =>
                event.id === payload.new.id ? (payload.new as CalendarEvent) : event
              )
            );
          } else if (payload.eventType === 'DELETE') {
            setEvents((prev) => prev.filter((event) => event.id !== payload.old.id));
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [patientId]);

  const createEvent = useCallback(
    async (event: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'>) => {
      try {
        const { error: insertError } = await supabase
          .from('calendar_events')
          .insert([event]);

        if (insertError) throw insertError;

        toast.success('Event created successfully');
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to create event';
        toast.error(errorMessage);
        throw err;
      }
    },
    []
  );

  const updateEvent = useCallback(async (event: CalendarEvent) => {
    try {
      const { error: updateError } = await supabase
        .from('calendar_events')
        .update({
          title: event.title,
          description: event.description,
          event_date: event.event_date,
          event_time: event.event_time,
          duration_minutes: event.duration_minutes,
          event_type: event.event_type,
          location: event.location,
          updated_at: new Date().toISOString(),
        })
        .eq('id', event.id);

      if (updateError) throw updateError;

      toast.success('Event updated successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update event';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  const deleteEvent = useCallback(async (eventId: string) => {
    try {
      const { error: deleteError } = await supabase
        .from('calendar_events')
        .delete()
        .eq('id', eventId);

      if (deleteError) throw deleteError;

      toast.success('Event deleted successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete event';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  return {
    events,
    loading,
    error,
    createEvent,
    updateEvent,
    deleteEvent,
    refetch: fetchEvents,
  };
}
