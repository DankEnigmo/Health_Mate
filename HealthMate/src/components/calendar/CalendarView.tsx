import { useState } from 'react';
import { Calendar } from '@/components/ui/calendar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CalendarEvent } from '@/types/calendar';
import { format, isSameDay, parseISO } from 'date-fns';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';

interface CalendarViewProps {
  events: CalendarEvent[];
  isReadOnly: boolean;
  onEventClick?: (event: CalendarEvent) => void;
  onDateClick?: (date: Date) => void;
  onCreateClick?: () => void;
}

export function CalendarView({
  events,
  isReadOnly,
  onEventClick,
  onDateClick,
  onCreateClick,
}: CalendarViewProps) {
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());

  // Get events for a specific date
  const getEventsForDate = (date: Date): CalendarEvent[] => {
    return events.filter((event) => {
      const eventDate = parseISO(event.event_date);
      return isSameDay(eventDate, date);
    });
  };

  // Get all dates that have events
  const datesWithEvents = events.map((event) => parseISO(event.event_date));

  const handleDateSelect = (date: Date | undefined) => {
    setSelectedDate(date);
    if (date && onDateClick) {
      onDateClick(date);
    }
  };

  const selectedDateEvents = selectedDate ? getEventsForDate(selectedDate) : [];

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

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Calendar</CardTitle>
            {!isReadOnly && onCreateClick && (
              <Button onClick={onCreateClick} size="sm">
                <Plus className="h-4 w-4 mr-2" />
                New Event
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <Calendar
            mode="single"
            selected={selectedDate}
            onSelect={handleDateSelect}
            modifiers={{
              hasEvent: datesWithEvents,
            }}
            modifiersClassNames={{
              hasEvent: 'font-bold underline',
            }}
            className="rounded-md border"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            {selectedDate ? format(selectedDate, 'MMMM d, yyyy') : 'Select a date'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {selectedDateEvents.length === 0 ? (
            <p className="text-muted-foreground text-sm">No events scheduled for this day</p>
          ) : (
            <div className="space-y-4">
              {selectedDateEvents.map((event) => (
                <div
                  key={event.id}
                  className={`p-4 rounded-lg border ${
                    !isReadOnly ? 'cursor-pointer hover:bg-accent' : ''
                  }`}
                  onClick={() => !isReadOnly && onEventClick && onEventClick(event)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge
                          variant="secondary"
                          className={`${getEventTypeColor(event.event_type)} text-white`}
                        >
                          {event.event_type}
                        </Badge>
                        {event.event_time && (
                          <span className="text-sm text-muted-foreground">
                            {event.event_time}
                          </span>
                        )}
                      </div>
                      <h4 className="font-semibold mb-1">{event.title}</h4>
                      {event.description && (
                        <p className="text-sm text-muted-foreground">{event.description}</p>
                      )}
                      {event.location && (
                        <p className="text-sm text-muted-foreground mt-1">
                          📍 {event.location}
                        </p>
                      )}
                      {event.duration_minutes && (
                        <p className="text-sm text-muted-foreground mt-1">
                          Duration: {event.duration_minutes} minutes
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
