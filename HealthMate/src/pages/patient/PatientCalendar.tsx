"use client";

import React, { useState } from "react";
import { CalendarView } from "@/components/calendar/CalendarView";
import { useCalendarEvents } from "@/hooks/use-calendar-events";
import { useSupabase } from "@/components/SessionContextProvider";
import { CalendarEvent } from "@/types/calendar";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { format, parseISO } from "date-fns";
import { Calendar, Clock, MapPin } from "lucide-react";

const PatientCalendar = () => {
  const { profile } = useSupabase();
  const patientId = profile?.id || "";
  const { events, loading } = useCalendarEvents(patientId);
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const handleEventClick = (event: CalendarEvent) => {
    setSelectedEvent(event);
    setIsDialogOpen(true);
  };

  const getEventTypeColor = (type: CalendarEvent['event_type']) => {
    switch (type) {
      case 'appointment':
        return 'bg-blue-500';
      case 'medication':
        return 'bg-green-500';
      case 'reminder':
        return 'bg-yellow-500';
      case 'other':
        return 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-primary">Your Appointments & Events</h1>
        <p className="text-lg text-muted-foreground">
          View your upcoming medical appointments and important events.
        </p>
        <div className="animate-pulse space-y-4">
          <div className="h-96 bg-muted rounded-lg"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary">Your Appointments & Events</h1>
      <p className="text-lg text-muted-foreground">
        View your upcoming medical appointments and important events.
      </p>

      <CalendarView
        events={events}
        isReadOnly={true}
        onEventClick={handleEventClick}
      />

      {/* Event Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Event Details</DialogTitle>
          </DialogHeader>
          {selectedEvent && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge
                  variant="secondary"
                  className={`${getEventTypeColor(selectedEvent.event_type)} text-white`}
                >
                  {selectedEvent.event_type}
                </Badge>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">{selectedEvent.title}</h3>
                {selectedEvent.description && (
                  <p className="text-muted-foreground mb-4">{selectedEvent.description}</p>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span>{format(parseISO(selectedEvent.event_date), 'MMMM d, yyyy')}</span>
                </div>
                {selectedEvent.event_time && (
                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>{selectedEvent.event_time}</span>
                    {selectedEvent.duration_minutes && (
                      <span className="text-muted-foreground">
                        ({selectedEvent.duration_minutes} minutes)
                      </span>
                    )}
                  </div>
                )}
                {selectedEvent.location && (
                  <div className="flex items-center gap-2 text-sm">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span>{selectedEvent.location}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PatientCalendar;