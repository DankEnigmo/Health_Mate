# Supabase Database Migrations

This directory contains SQL migration files for the HealthMate fall detection integration project.

## Prerequisites

Before applying these migrations, ensure you have:

1. A Supabase project set up
2. The Supabase CLI installed (optional, for automated migrations)
3. Access to the Supabase SQL Editor in your project dashboard

## Required Tables

These migrations assume the following tables already exist in your Supabase database:

- `profiles` - User profiles table with `id` column (UUID)
- `caregiver_patients` - Junction table linking caregivers to patients with columns:
  - `caregiver_id` (UUID, references profiles.id)
  - `patient_id` (UUID, references profiles.id)

If these tables don't exist, you'll need to create them first.

## Migration Files

The migrations are numbered in the order they should be applied:

1. **20240101000001_create_fall_events_table.sql**
   - Creates the `fall_events` table for storing fall detection events
   - Adds indexes for patient_id, timestamp, and status
   - Sets up Row Level Security (RLS) policies

2. **20240101000002_create_calendar_events_table.sql**
   - Creates the `calendar_events` table for patient appointments and reminders
   - Adds indexes for patient_id, event_date, and event_type
   - Sets up RLS policies

3. **20240101000003_create_medications_table.sql**
   - Creates the `medications` table for medication schedules
   - Adds indexes for patient_id and is_active
   - Sets up RLS policies

4. **20240101000004_create_medication_logs_table.sql**
   - Creates the `medication_logs` table for tracking medication adherence
   - Adds indexes for medication_id, patient_id, scheduled_time, and status
   - Sets up RLS policies

## Applying Migrations

### Option 1: Using Supabase SQL Editor (Recommended for Manual Setup)

1. Log in to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Copy and paste each migration file in order
4. Execute each migration

### Option 2: Using Supabase CLI

If you have the Supabase CLI installed and linked to your project:

```bash
# Link your project (first time only)
supabase link --project-ref your-project-ref

# Apply all migrations
supabase db push
```

### Option 3: Manual Execution via psql

If you have direct database access:

```bash
psql -h your-db-host -U postgres -d postgres -f 20240101000001_create_fall_events_table.sql
psql -h your-db-host -U postgres -d postgres -f 20240101000002_create_calendar_events_table.sql
psql -h your-db-host -U postgres -d postgres -f 20240101000003_create_medications_table.sql
psql -h your-db-host -U postgres -d postgres -f 20240101000004_create_medication_logs_table.sql
```

## Row Level Security (RLS) Policies

All tables have RLS enabled with the following access patterns:

### fall_events
- Patients can view their own fall events
- Caregivers can view fall events for assigned patients
- Service role (backend) can insert fall events
- Caregivers can update fall events for assigned patients

### calendar_events
- Patients can view their own calendar events (read-only)
- Caregivers can view, create, update, and delete calendar events for assigned patients

### medications
- Patients can view their own medication schedules (read-only)
- Caregivers can view, create, update, and delete medication schedules for assigned patients

### medication_logs
- Patients can view and update their own medication logs
- Caregivers can view and update medication logs for assigned patients
- Service role (backend) can insert medication logs for automated status updates

## Verification

After applying the migrations, verify the tables were created successfully:

```sql
-- Check if tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('fall_events', 'calendar_events', 'medications', 'medication_logs');

-- Check indexes
SELECT tablename, indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
AND tablename IN ('fall_events', 'calendar_events', 'medications', 'medication_logs');

-- Check RLS policies
SELECT tablename, policyname, cmd 
FROM pg_policies 
WHERE schemaname = 'public';
```

## Troubleshooting

### Error: relation "profiles" does not exist

You need to create the `profiles` table first. This is typically created during Supabase authentication setup.

### Error: relation "caregiver_patients" does not exist

You need to create the `caregiver_patients` junction table first:

```sql
CREATE TABLE caregiver_patients (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  caregiver_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(caregiver_id, patient_id)
);

ALTER TABLE caregiver_patients ENABLE ROW LEVEL SECURITY;
```

### Error: function "update_updated_at_column" does not exist

The function is created in the first migration file. Make sure to run migrations in order.

## Next Steps

After successfully applying these migrations:

1. Update your environment variables with Supabase credentials
2. Test the database connection from your backend
3. Implement the backend API endpoints
4. Create frontend components that interact with these tables

For more information, see the main project documentation in `.kiro/specs/fall-detection-integration/`.
