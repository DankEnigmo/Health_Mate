import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CalendarEvent } from '@/types/calendar';
import { format, parseISO, isAfter, isSameDay } from 'date-fns';
import { Calendar, Clock, MapPin } from 'lucide-react';
import { Link } from 'react-router-dom';

interface UpcomingEventsProps {
  events: CalendarEvent[];
  limit?: number;
  loading?: boolean;
  calendarLink?: string;
}

export function UpcomingEvents({
  events,
  limit = 5,
  loading = false,
  calendarLink,
}: UpcomingEventsProps) {
  // Filter and sort upcoming events
  const now = new Date();
  const upcomingEvents = events
    .filter((event) => {
      const eventDate = parseISO(event.event_date);
      return isAfter(eventDate, now) || isSameDay(eventDate, now);
    })
    .sort((a, b) => {
      const dateA = parseISO(a.event_date);
      const dateB = parseISO(b.event_date);
      
      // Sort by date first
      if (dateA.getTime() !== dateB.getTime()) {
        return dateA.getTime() - dateB.getTime();
      }
      
      // If same date, sort by time
      if (a.event_time && b.event_time) {
        return a.event_time.localeCompare(b.event_time);
      }
      
      // Events with time come before events without time
      if (a.event_time && !b.event_time) return -1;
      if (!a.event_time && b.event_time) return 1;
      
      return 0;
    })
    .slice(0, limit);

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

  const formatEventDate = (event: CalendarEvent) => {
    const date = parseISO(event.event_date);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    if (isSameDay(date, today)) {
      return 'Today';
    } else if (isSameDay(date, tomorrow)) {
      return 'Tomorrow';
    } else {
      return format(date, 'MMM d, yyyy');
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Upcoming Events</CardTitle>
          {calendarLink && (
            <Link to={calendarLink}>
              <Button variant="ghost" size="sm">
                View All
              </Button>
            </Link>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-20 bg-muted rounded-lg"></div>
              </div>
            ))}
          </div>
        ) : upcomingEvents.length === 0 ? (
          <div className="text-center py-8">
            <Calendar className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
            <p className="text-muted-foreground">No upcoming events</p>
          </div>
        ) : (
          <div className="space-y-3">
            {upcomingEvents.map((event) => (
              <div
                key={event.id}
                className="p-4 rounded-lg border hover:bg-accent transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge
                        variant="secondary"
                        className={`${getEventTypeColor(event.event_type)} text-white text-xs`}
                      >
                        {event.event_type}
                      </Badge>
                    </div>
                    <h4 className="font-semibold mb-1 truncate">{event.title}</h4>
                    {event.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                        {event.description}
                      </p>
                    )}
                    <div className="flex flex-col gap-1 text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>{formatEventDate(event)}</span>
                      </div>
                      {event.event_time && (
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          <span>{event.event_time}</span>
                          {event.duration_minutes && (
                            <span className="text-xs">({event.duration_minutes} min)</span>
                          )}
                        </div>
                      )}
                      {event.location && (
                        <div className="flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          <span className="truncate">{event.location}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
