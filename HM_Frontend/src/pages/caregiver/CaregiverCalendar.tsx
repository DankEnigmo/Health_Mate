"use client";

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";
import { CalendarView } from "@/components/calendar/CalendarView";
import { EventDialog } from "@/components/calendar/EventDialog";
import { useCalendarEvents } from "@/hooks/use-calendar-events";
import { CalendarEvent } from "@/types/calendar";

const CaregiverCalendar = () => {
  const { patientId, loading: patientLoading, error: patientError } = useCaregiverPatient();
  const [patientName, setPatientName] = useState<string | null>(null);
  const [caregiverId, setCaregiverId] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | undefined>(undefined);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(undefined);

  const {
    events,
    loading: eventsLoading,
    error: eventsError,
    createEvent,
    updateEvent,
    deleteEvent,
  } = useCalendarEvents(patientId || '');

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

  // Get current caregiver ID
  useEffect(() => {
    const fetchCaregiverId = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCaregiverId(user.id);
      }
    };

    fetchCaregiverId();
  }, []);

  const handleEventClick = (event: CalendarEvent) => {
    setSelectedEvent(event);
    setSelectedDate(undefined);
    setIsDialogOpen(true);
  };

  const handleDateClick = (date: Date) => {
    setSelectedDate(date);
    setSelectedEvent(undefined);
    setIsDialogOpen(true);
  };

  const handleCreateClick = () => {
    setSelectedEvent(undefined);
    setSelectedDate(new Date());
    setIsDialogOpen(true);
  };

  const handleDialogClose = () => {
    setIsDialogOpen(false);
    setSelectedEvent(undefined);
    setSelectedDate(undefined);
  };

  const handleSaveEvent = async (eventData: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'>) => {
    if (selectedEvent) {
      await updateEvent({ ...selectedEvent, ...eventData });
    } else {
      await createEvent(eventData);
    }
  };

  if (patientLoading) {
    return <p className="text-muted-foreground p-4">Loading calendar...</p>;
  }

  if (patientError) {
    return <p className="text-destructive p-4">Error: {patientError}</p>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Shared Calendar for {patientName || "Your Patient"}</h1>
      <p className="text-lg text-muted-foreground">
        View and manage medical appointments and important events for {patientName || "your patient"}.
      </p>
      {!patientId ? (
        <Card className="p-4 border rounded-lg bg-card">
          <p className="text-destructive">No patient assigned. Please assign a patient to view their calendar.</p>
        </Card>
      ) : (
        <>
          {eventsError && (
            <Card className="p-4 border rounded-lg bg-destructive/10">
              <p className="text-destructive">Error loading events: {eventsError}</p>
            </Card>
          )}
          
          {eventsLoading ? (
            <Card className="p-4 border rounded-lg bg-card">
              <p className="text-muted-foreground">Loading calendar events...</p>
            </Card>
          ) : (
            <CalendarView
              events={events}
              isReadOnly={false}
              onEventClick={handleEventClick}
              onDateClick={handleDateClick}
              onCreateClick={handleCreateClick}
            />
          )}

          {caregiverId && (
            <EventDialog
              isOpen={isDialogOpen}
              onClose={handleDialogClose}
              event={selectedEvent}
              initialDate={selectedDate}
              patientId={patientId}
              caregiverId={caregiverId}
              onSave={handleSaveEvent}
              onDelete={deleteEvent}
            />
          )}
        </>
      )}
    </div>
  );
};

export default CaregiverCalendar;