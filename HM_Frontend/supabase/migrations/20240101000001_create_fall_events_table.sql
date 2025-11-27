-- Create fall_events table
-- This table stores fall detection events with metadata
-- Requirements: 8.1, 8.2

CREATE TABLE IF NOT EXISTS fall_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  caregiver_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
  person_tracking_id INTEGER,
  fall_count INTEGER NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  video_frame_url TEXT,
  metadata JSONB,
  status TEXT DEFAULT 'new' CHECK (status IN ('new', 'reviewed', 'resolved')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_fall_events_patient ON fall_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_fall_events_timestamp ON fall_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fall_events_status ON fall_events(status);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_fall_events_updated_at
  BEFORE UPDATE ON fall_events
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) Policies
ALTER TABLE fall_events ENABLE ROW LEVEL SECURITY;

-- Policy: Patients can view their own fall events
CREATE POLICY "Patients can view their own fall events"
  ON fall_events
  FOR SELECT
  USING (auth.uid() = patient_id);

-- Policy: Caregivers can view fall events for their assigned patients
CREATE POLICY "Caregivers can view assigned patient fall events"
  ON fall_events
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = fall_events.patient_id
    )
  );

-- Policy: Backend service can insert fall events (using service role key)
CREATE POLICY "Service role can insert fall events"
  ON fall_events
  FOR INSERT
  WITH CHECK (true);

-- Policy: Caregivers can update fall events for their assigned patients
CREATE POLICY "Caregivers can update assigned patient fall events"
  ON fall_events
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = fall_events.patient_id
    )
  );

-- Add comment to table
COMMENT ON TABLE fall_events IS 'Stores fall detection events with timestamps, patient information, and metadata';
COMMENT ON COLUMN fall_events.person_tracking_id IS 'Tracking ID from the fall detection system';
COMMENT ON COLUMN fall_events.fall_count IS 'Total fall count at the time of this event';
COMMENT ON COLUMN fall_events.metadata IS 'Additional data such as bounding box coordinates, confidence scores, etc.';
COMMENT ON COLUMN fall_events.status IS 'Event status: new (unreviewed), reviewed (seen by caregiver), resolved (action taken)';
