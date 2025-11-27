-- Update medications table to allow empty times array for "as_needed" frequency
-- This migration modifies the times column constraint

-- Drop the NOT NULL constraint on times column
ALTER TABLE medications ALTER COLUMN times DROP NOT NULL;

-- Add a check constraint: times can be empty only if frequency is 'as_needed'
ALTER TABLE medications ADD CONSTRAINT check_times_for_frequency
  CHECK (
    (frequency = 'as_needed' AND (times IS NULL OR array_length(times, 1) IS NULL OR array_length(times, 1) = 0))
    OR
    (frequency != 'as_needed' AND times IS NOT NULL AND array_length(times, 1) > 0)
  );

-- Add comment explaining the constraint
COMMENT ON CONSTRAINT check_times_for_frequency ON medications IS 
  'Ensures times array is empty for as_needed medications and non-empty for scheduled medications';

