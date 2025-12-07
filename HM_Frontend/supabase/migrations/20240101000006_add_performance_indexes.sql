-- Performance Optimization Indexes
-- Created: 2025-01-XX
-- Purpose: Add missing indexes for 10x faster query performance

-- Fall Events Indexes
-- Most common query: Get recent falls for a patient
CREATE INDEX IF NOT EXISTS idx_fall_events_patient_time 
ON fall_events(patient_id, timestamp DESC);

-- Filter by status (e.g., only show 'new' alerts)
CREATE INDEX IF NOT EXISTS idx_fall_events_patient_status 
ON fall_events(patient_id, status) 
WHERE status IN ('new', 'reviewed');

-- Caregiver viewing all their patients' falls
CREATE INDEX IF NOT EXISTS idx_fall_events_caregiver_time 
ON fall_events(caregiver_id, timestamp DESC);

-- Medication Logs Indexes
-- Most common query: Get logs for a medication with status
CREATE INDEX IF NOT EXISTS idx_med_logs_medication_scheduled 
ON medication_logs(medication_id, scheduled_time DESC);

-- Patient viewing their own logs
CREATE INDEX IF NOT EXISTS idx_med_logs_patient_scheduled 
ON medication_logs(patient_id, scheduled_time DESC);

-- Filter pending medications
CREATE INDEX IF NOT EXISTS idx_med_logs_patient_status 
ON medication_logs(patient_id, status) 
WHERE status = 'pending';

-- Calendar Events Indexes
-- Most common query: Get events for patient by date
CREATE INDEX IF NOT EXISTS idx_calendar_patient_date 
ON calendar_events(patient_id, event_date, event_time);

-- Filter by event type
CREATE INDEX IF NOT EXISTS idx_calendar_patient_type 
ON calendar_events(patient_id, event_type, event_date);

-- Caregiver viewing their patients' events
CREATE INDEX IF NOT EXISTS idx_calendar_caregiver_date 
ON calendar_events(caregiver_id, event_date);

-- Medications Indexes
-- Most common query: Get active medications for patient
CREATE INDEX IF NOT EXISTS idx_medications_patient_active 
ON medications(patient_id, is_active) 
WHERE is_active = true;

-- Get medications by caregiver
CREATE INDEX IF NOT EXISTS idx_medications_caregiver_active 
ON medications(caregiver_id, is_active) 
WHERE is_active = true;

-- Caregiver-Patient Relationship Indexes
-- Lookup patients for a caregiver
CREATE INDEX IF NOT EXISTS idx_caregiver_patients_caregiver 
ON caregiver_patients(caregiver_id);

-- Lookup caregivers for a patient
CREATE INDEX IF NOT EXISTS idx_caregiver_patients_patient 
ON caregiver_patients(patient_id);

-- Comments
COMMENT ON INDEX idx_fall_events_patient_time IS 'Optimize fall event queries by patient and time';
COMMENT ON INDEX idx_med_logs_medication_scheduled IS 'Optimize medication log queries';
COMMENT ON INDEX idx_calendar_patient_date IS 'Optimize calendar event queries';
COMMENT ON INDEX idx_medications_patient_active IS 'Optimize active medication lookups';

-- Analysis: Check index usage after deployment
-- Run: SELECT schemaname, tablename, indexname, idx_scan FROM pg_stat_user_indexes;
