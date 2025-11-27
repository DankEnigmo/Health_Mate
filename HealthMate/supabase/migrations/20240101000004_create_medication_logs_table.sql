-- Create medication_logs table
-- This table stores logs of when medications are taken or missed
-- Requirements: 15.3

CREATE TABLE IF NOT EXISTS medication_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  medication_id UUID NOT NULL REFERENCES medications(id) ON DELETE CASCADE,
  patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  scheduled_time TIMESTAMPTZ NOT NULL,
  taken_time TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'taken', 'missed', 'skipped')),
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_medication_logs_medication ON medication_logs(medication_id);
CREATE INDEX IF NOT EXISTS idx_medication_logs_patient ON medication_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_medication_logs_scheduled ON medication_logs(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_medication_logs_status ON medication_logs(status);

-- Row Level Security (RLS) Policies
ALTER TABLE medication_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Patients can view their own medication logs
CREATE POLICY "Patients can view their own medication logs"
  ON medication_logs
  FOR SELECT
  USING (auth.uid() = patient_id);

-- Policy: Caregivers can view medication logs for their assigned patients
CREATE POLICY "Caregivers can view assigned patient medication logs"
  ON medication_logs
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medication_logs.patient_id
    )
  );

-- Policy: Patients can insert their own medication logs (when marking as taken)
CREATE POLICY "Patients can create their own medication logs"
  ON medication_logs
  FOR INSERT
  WITH CHECK (auth.uid() = patient_id);

-- Policy: Backend service can insert medication logs (using service role key)
CREATE POLICY "Service role can insert medication logs"
  ON medication_logs
  FOR INSERT
  WITH CHECK (true);

-- Policy: Patients can update their own medication logs
CREATE POLICY "Patients can update their own medication logs"
  ON medication_logs
  FOR UPDATE
  USING (auth.uid() = patient_id);

-- Policy: Caregivers can update medication logs for their assigned patients
CREATE POLICY "Caregivers can update assigned patient medication logs"
  ON medication_logs
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medication_logs.patient_id
    )
  );

-- Add comments to table
COMMENT ON TABLE medication_logs IS 'Stores logs of medication adherence - when medications are taken, missed, or skipped';
COMMENT ON COLUMN medication_logs.scheduled_time IS 'When the medication was scheduled to be taken';
COMMENT ON COLUMN medication_logs.taken_time IS 'When the medication was actually taken (null if not taken)';
COMMENT ON COLUMN medication_logs.status IS 'Status: pending (not yet due), taken (marked as taken), missed (past due and not taken), skipped (intentionally skipped)';
COMMENT ON COLUMN medication_logs.notes IS 'Optional notes, especially useful for skipped medications to explain the reason';
