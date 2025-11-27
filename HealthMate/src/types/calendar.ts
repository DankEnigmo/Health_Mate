export type CalendarEventType = 'appointment' | 'medication' | 'reminder' | 'other';

export interface CalendarEvent {
  id: string;
  patient_id: string;
  caregiver_id: string;
  title: string;
  description?: string;
  event_date: string; // ISO date string
  event_time?: string; // HH:mm format
  duration_minutes?: number;
  event_type: CalendarEventType;
  location?: string;
  created_at: string;
  updated_at: string;
}

export interface CalendarEventFormData {
  title: string;
  description?: string;
  event_date: Date;
  event_time?: string;
  duration_minutes?: number;
  event_type: CalendarEventType;
  location?: string;
}
