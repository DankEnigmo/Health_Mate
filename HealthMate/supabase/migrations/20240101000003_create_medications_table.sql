-- Create medications table
-- This table stores medication schedules for patients
-- Requirements: 14.2, 14.3

CREATE TABLE IF NOT EXISTS medications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  caregiver_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  dosage TEXT NOT NULL,
  frequency TEXT NOT NULL CHECK (frequency IN ('daily', 'twice_daily', 'three_times_daily', 'weekly', 'as_needed')),
  times TIME[] NOT NULL,
  instructions TEXT,
  start_date DATE NOT NULL,
  end_date DATE,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_medications_patient ON medications(patient_id);
CREATE INDEX IF NOT EXISTS idx_medications_active ON medications(is_active);

-- Create updated_at trigger
CREATE TRIGGER update_medications_updated_at
  BEFORE UPDATE ON medications
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) Policies
ALTER TABLE medications ENABLE ROW LEVEL SECURITY;

-- Policy: Patients can view their own medication schedules
CREATE POLICY "Patients can view their own medication schedules"
  ON medications
  FOR SELECT
  USING (auth.uid() = patient_id);

-- Policy: Caregivers can view medication schedules for their assigned patients
CREATE POLICY "Caregivers can view assigned patient medication schedules"
  ON medications
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medications.patient_id
    )
  );

-- Policy: Caregivers can insert medication schedules for their assigned patients
CREATE POLICY "Caregivers can create medication schedules for assigned patients"
  ON medications
  FOR INSERT
  WITH CHECK (
    auth.uid() = caregiver_id AND
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medications.patient_id
    )
  );

-- Policy: Caregivers can update medication schedules for their assigned patients
CREATE POLICY "Caregivers can update medication schedules for assigned patients"
  ON medications
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medications.patient_id
    )
  );

-- Policy: Caregivers can delete medication schedules for their assigned patients
CREATE POLICY "Caregivers can delete medication schedules for assigned patients"
  ON medications
  FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM caregiver_patients
      WHERE caregiver_patients.caregiver_id = auth.uid()
      AND caregiver_patients.patient_id = medications.patient_id
    )
  );

-- Add comments to table
COMMENT ON TABLE medications IS 'Stores medication schedules for patients';
COMMENT ON COLUMN medications.frequency IS 'How often medication should be taken: daily, twice_daily, three_times_daily, weekly, or as_needed';
COMMENT ON COLUMN medications.times IS 'Array of times (HH:MM format) when medication should be taken';
COMMENT ON COLUMN medications.is_active IS 'Whether the medication schedule is currently active';
COMMENT ON COLUMN medications.start_date IS 'Date when the medication schedule begins';
COMMENT ON COLUMN medications.end_date IS 'Optional date when the medication schedule ends';
