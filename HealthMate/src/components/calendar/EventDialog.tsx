import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { CalendarEvent, CalendarEventFormData, CalendarEventType } from '@/types/calendar';
import { format } from 'date-fns';

interface EventDialogProps {
  isOpen: boolean;
  onClose: () => void;
  event?: CalendarEvent;
  initialDate?: Date;
  patientId: string;
  caregiverId: string;
  onSave: (event: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'>) => Promise<void>;
  onDelete?: (eventId: string) => Promise<void>;
}

export function EventDialog({
  isOpen,
  onClose,
  event,
  initialDate,
  patientId,
  caregiverId,
  onSave,
  onDelete,
}: EventDialogProps) {
  const [formData, setFormData] = useState<CalendarEventFormData>({
    title: '',
    description: '',
    event_date: initialDate || new Date(),
    event_time: '',
    duration_minutes: undefined,
    event_type: 'appointment',
    location: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (event) {
      setFormData({
        title: event.title,
        description: event.description || '',
        event_date: new Date(event.event_date),
        event_time: event.event_time || '',
        duration_minutes: event.duration_minutes,
        event_type: event.event_type,
        location: event.location || '',
      });
    } else if (initialDate) {
      setFormData((prev) => ({
        ...prev,
        event_date: initialDate,
      }));
    }
  }, [event, initialDate]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.title.trim()) {
      newErrors.title = 'Title is required';
    }

    if (!formData.event_date) {
      newErrors.event_date = 'Date is required';
    }

    if (formData.event_time && !/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/.test(formData.event_time)) {
      newErrors.event_time = 'Time must be in HH:mm format';
    }

    if (
      formData.duration_minutes !== undefined &&
      (formData.duration_minutes < 0 || formData.duration_minutes > 1440)
    ) {
      newErrors.duration_minutes = 'Duration must be between 0 and 1440 minutes';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      const eventData: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'> = {
        patient_id: patientId,
        caregiver_id: caregiverId,
        title: formData.title.trim(),
        description: formData.description?.trim() || undefined,
        event_date: format(formData.event_date, 'yyyy-MM-dd'),
        event_time: formData.event_time || undefined,
        duration_minutes: formData.duration_minutes,
        event_type: formData.event_type,
        location: formData.location?.trim() || undefined,
      };

      await onSave(eventData);
      handleClose();
    } catch (error) {
      console.error('Error saving event:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async () => {
    if (!event || !onDelete) return;

    if (confirm('Are you sure you want to delete this event?')) {
      setIsSubmitting(true);
      try {
        await onDelete(event.id);
        handleClose();
      } catch (error) {
        console.error('Error deleting event:', error);
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const handleClose = () => {
    setFormData({
      title: '',
      description: '',
      event_date: initialDate || new Date(),
      event_time: '',
      duration_minutes: undefined,
      event_type: 'appointment',
      location: '',
    });
    setErrors({});
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>{event ? 'Edit Event' : 'Create Event'}</DialogTitle>
          <DialogDescription>
            {event ? 'Update the event details below.' : 'Fill in the details for the new event.'}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="title">
                Title <span className="text-destructive">*</span>
              </Label>
              <Input
                id="title"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                placeholder="Event title"
              />
              {errors.title && <p className="text-sm text-destructive">{errors.title}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Event description"
                rows={3}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="event_date">
                  Date <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="event_date"
                  type="date"
                  value={format(formData.event_date, 'yyyy-MM-dd')}
                  onChange={(e) =>
                    setFormData({ ...formData, event_date: new Date(e.target.value) })
                  }
                />
                {errors.event_date && (
                  <p className="text-sm text-destructive">{errors.event_date}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="event_time">Time</Label>
                <Input
                  id="event_time"
                  type="time"
                  value={formData.event_time}
                  onChange={(e) => setFormData({ ...formData, event_time: e.target.value })}
                />
                {errors.event_time && (
                  <p className="text-sm text-destructive">{errors.event_time}</p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="event_type">Type</Label>
                <Select
                  value={formData.event_type}
                  onValueChange={(value: CalendarEventType) =>
                    setFormData({ ...formData, event_type: value })
                  }
                >
                  <SelectTrigger id="event_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="appointment">Appointment</SelectItem>
                    <SelectItem value="medication">Medication</SelectItem>
                    <SelectItem value="reminder">Reminder</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="duration_minutes">Duration (minutes)</Label>
                <Input
                  id="duration_minutes"
                  type="number"
                  min="0"
                  max="1440"
                  value={formData.duration_minutes || ''}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      duration_minutes: e.target.value ? parseInt(e.target.value) : undefined,
                    })
                  }
                  placeholder="Optional"
                />
                {errors.duration_minutes && (
                  <p className="text-sm text-destructive">{errors.duration_minutes}</p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="location">Location</Label>
              <Input
                id="location"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                placeholder="Event location"
              />
            </div>
          </div>

          <DialogFooter>
            {event && onDelete && (
              <Button
                type="button"
                variant="destructive"
                onClick={handleDelete}
                disabled={isSubmitting}
              >
                Delete
              </Button>
            )}
            <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Saving...' : event ? 'Update' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
