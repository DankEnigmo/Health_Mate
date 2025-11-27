-- Create calendar_events table
-- This table stores calendar events for patients
-- Requirements: 11.3

CREATE TABLE IF NOT EXISTS calendar_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  caregiver_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  event_date DATE NOT NULL,
  event_time TIME,
  duration_minutes INTEGER,
  event_type TEXT DEFAULT 'appointment' CHECK (event_type IN ('appointment', 'medication', 'reminder', 'other')),
  location TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_calendar_events_patient ON calendar_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_calendar_events_date ON calendar_events(event_date);
CREATE INDEX IF NOT EXISTS idx_calendar_events_type ON calendar_events(event_type);

-- Create updated_at trigger
CREATE TRIGGER update_calendar_events_updated_at
  BEFORE UPDATE ON calendar_events
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) Policies
ALTER TABLE calendar_events ENABLE ROW LEVEL SECURITY;

-- Policy: Patients can view their own calendar events
CREATE POLICY "Patients can view their own calendar events"
  ON calendar_events
  FOR SELECT
  USING (auth.uid() = patient_id);

-- Policy: Caregivers can view calendar events for their assigned patients
CREATE POLICY "Caregivers can view assigned patient calendar events"
  ON calendar_events
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = calendar_events.patient_id
    )
  );

-- Policy: Caregivers can insert calendar events for their assigned patients
CREATE POLICY "Caregivers can create calendar events for assigned patients"
  ON calendar_events
  FOR INSERT
  WITH CHECK (
    auth.uid() = caregiver_id AND
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = calendar_events.patient_id
    )
  );

-- Policy: Caregivers can update calendar events for their assigned patients
CREATE POLICY "Caregivers can update calendar events for assigned patients"
  ON calendar_events
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = calendar_events.patient_id
    )
  );

-- Policy: Caregivers can delete calendar events for their assigned patients
CREATE POLICY "Caregivers can delete calendar events for assigned patients"
  ON calendar_events
  FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = calendar_events.patient_id
    )
  );

-- Add comments to table
COMMENT ON TABLE calendar_events IS 'Stores calendar events and appointments for patients';
COMMENT ON COLUMN calendar_events.event_type IS 'Type of event: appointment, medication, reminder, or other';
COMMENT ON COLUMN calendar_events.duration_minutes IS 'Duration of the event in minutes';
