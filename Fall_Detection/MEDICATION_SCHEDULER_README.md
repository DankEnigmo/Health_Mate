# Medication Scheduler

## Overview

The Medication Scheduler is an automated background service that manages medication adherence tracking by:

1. **Updating Missed Medications**: Automatically updates medication logs from 'pending' to 'missed' status when medications are past their scheduled time (runs every 15 minutes)
2. **Generating Daily Logs**: Creates pending medication logs for upcoming medications based on active schedules (runs daily at midnight)

## Architecture

The scheduler is integrated into the FastAPI backend server and runs as background tasks using Python's `asyncio`.

### Components

- **MedicationScheduler** (`medication_scheduler.py`): Main scheduler class
- **API Server Integration** (`api_server.py`): Initializes and manages the scheduler lifecycle

## Features

### 1. Automatic Status Updates (Every 15 Minutes)

The scheduler checks for medications that are past their scheduled time and updates their status:

```python
# Finds all logs with:
# - status = 'pending'
# - scheduled_time < current_time

# Updates them to:
# - status = 'missed'
```

**Requirements Satisfied**: 15.3

### 2. Daily Log Generation (Midnight)

The scheduler generates pending medication logs for the upcoming day:

```python
# For each active medication schedule:
# 1. Check if medication is active (start_date <= today <= end_date)
# 2. Check if frequency matches the day (daily, weekly, etc.)
# 3. Create a log for each scheduled time
# 4. Skip if log already exists (prevents duplicates)
```

**Requirements Satisfied**: 15.1

## Database Schema

### Medications Table
```sql
CREATE TABLE medications (
  id UUID PRIMARY KEY,
  patient_id UUID NOT NULL,
  caregiver_id UUID NOT NULL,
  name TEXT NOT NULL,
  dosage TEXT NOT NULL,
  frequency TEXT NOT NULL,  -- daily, twice_daily, three_times_daily, weekly, as_needed
  times TIME[] NOT NULL,    -- Array of scheduled times
  start_date DATE NOT NULL,
  end_date DATE,
  is_active BOOLEAN DEFAULT TRUE
);
```

### Medication Logs Table
```sql
CREATE TABLE medication_logs (
  id UUID PRIMARY KEY,
  medication_id UUID NOT NULL,
  patient_id UUID NOT NULL,
  scheduled_time TIMESTAMPTZ NOT NULL,
  taken_time TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'pending',  -- pending, taken, missed, skipped
  notes TEXT
);
```

## Status Flow

```
pending → taken     (patient marks as taken)
pending → missed    (automatic after scheduled_time passes)
pending → skipped   (caregiver/patient manually skips)
```

## Usage

### Starting the Scheduler

The scheduler starts automatically when the FastAPI server starts:

```python
# In api_server.py
medication_scheduler = MedicationScheduler(supabase_client)
await medication_scheduler.start()
```

### Stopping the Scheduler

The scheduler stops automatically when the FastAPI server shuts down:

```python
await medication_scheduler.stop()
```

### Manual Execution (for testing)

You can manually trigger the scheduler methods:

```python
# Update missed medications
count = await scheduler.update_missed_medications()
print(f"Updated {count} medications to 'missed' status")

# Generate logs for a specific date
from datetime import datetime
count = await scheduler.generate_daily_medication_logs(datetime.now())
print(f"Created {count} medication logs")
```

## Configuration

The scheduler uses the Supabase client configured in the API server. No additional configuration is required.

## Logging

The scheduler logs all operations:

- **INFO**: Scheduler start/stop, task execution, summary statistics
- **DEBUG**: Individual log updates, detailed processing
- **ERROR**: Failures, exceptions with stack traces

Example logs:
```
2024-01-01 00:00:00 - INFO - Starting medication scheduler...
2024-01-01 00:00:00 - INFO - Status update task started (runs every 15 minutes)
2024-01-01 00:00:00 - INFO - Log generation task started (runs daily at midnight)
2024-01-01 00:15:00 - INFO - Checking for missed medications
2024-01-01 00:15:00 - INFO - Found 3 missed medications
2024-01-01 00:15:00 - INFO - Successfully updated 3 medication logs to 'missed' status
```

## Error Handling

The scheduler is designed to be resilient:

1. **Database Errors**: Logged but don't crash the scheduler
2. **Individual Log Failures**: Logged but don't stop processing other logs
3. **Task Cancellation**: Gracefully handled during shutdown
4. **Unexpected Errors**: Logged with full stack trace, scheduler retries after 60 seconds

## Testing

Comprehensive test suite in `tests/test_medication_scheduler.py`:

- ✅ Update missed medications (no pending)
- ✅ Update missed medications (with pending)
- ✅ Generate daily logs (no active medications)
- ✅ Generate daily logs (with active medications)
- ✅ Skip existing logs (prevent duplicates)
- ✅ Frequency checks (daily, weekly, as_needed)
- ✅ Scheduler start/stop lifecycle

Run tests:
```bash
pytest tests/test_medication_scheduler.py -v
```

## Frequency Support

| Frequency | Behavior |
|-----------|----------|
| `daily` | Generates logs every day |
| `twice_daily` | Generates logs every day (2 times) |
| `three_times_daily` | Generates logs every day (3 times) |
| `weekly` | Generates logs on Mondays (configurable) |
| `as_needed` | Does not auto-generate logs |

## Performance Considerations

- **Status Updates**: Runs every 15 minutes, processes only pending logs
- **Log Generation**: Runs once per day at midnight
- **Database Queries**: Optimized with indexes on `status`, `scheduled_time`, `patient_id`
- **Duplicate Prevention**: Checks for existing logs before inserting

## Future Enhancements

Potential improvements:

1. **Configurable Schedule**: Allow customizing update frequency (currently 15 minutes)
2. **Notification Integration**: Send alerts when medications are missed
3. **Advanced Frequency**: Support custom schedules (every other day, specific days of week)
4. **Batch Processing**: Process multiple updates in a single transaction
5. **Metrics**: Track scheduler performance and medication adherence trends
6. **Retry Logic**: Exponential backoff for database failures

## Troubleshooting

### Scheduler Not Running

Check logs for initialization errors:
```bash
grep "medication scheduler" fall_detection_api.log
```

### Logs Not Being Generated

1. Verify medications are active (`is_active = true`)
2. Check `start_date` and `end_date` are valid
3. Verify `times` array is not empty
4. Check frequency matches the day

### Status Not Updating

1. Verify scheduled_time is in the past
2. Check status is 'pending'
3. Review error logs for database issues

## Dependencies

- Python 3.8+
- asyncio
- supabase-py
- pytest (for testing)
- pytest-asyncio (for async tests)

## License

Part of the HealthMate Fall Detection Integration project.
