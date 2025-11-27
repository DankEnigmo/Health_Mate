"""
Tests for Medication Scheduler

This module tests the medication scheduler functionality including:
- Updating missed medications
- Generating daily medication logs
"""

import asyncio
import pytest
from datetime import datetime, timedelta, time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from medication_scheduler import MedicationScheduler


class MockSupabaseResponse:
    """Mock Supabase response object."""
    def __init__(self, data=None):
        self.data = data if data is not None else []


class MockSupabaseTable:
    """Mock Supabase table for testing."""
    def __init__(self, data=None):
        self.data = data if data is not None else []
        self._filters = {}
        self._select_fields = "*"
    
    def select(self, fields):
        self._select_fields = fields
        return self
    
    def eq(self, field, value):
        self._filters[field] = value
        return self
    
    def lt(self, field, value):
        self._filters[f"{field}_lt"] = value
        return self
    
    def lte(self, field, value):
        self._filters[f"{field}_lte"] = value
        return self
    
    def execute(self):
        return MockSupabaseResponse(self.data)
    
    def insert(self, data):
        return self
    
    def update(self, data):
        return self


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = Mock()
    return client


@pytest.fixture
def scheduler(mock_supabase):
    """Create a medication scheduler instance."""
    return MedicationScheduler(mock_supabase)


@pytest.mark.asyncio
async def test_update_missed_medications_no_pending():
    """Test updating missed medications when there are no pending logs."""
    mock_supabase = Mock()
    mock_table = MockSupabaseTable(data=[])
    mock_supabase.table.return_value = mock_table
    
    scheduler = MedicationScheduler(mock_supabase)
    
    result = await scheduler.update_missed_medications()
    
    assert result == 0
    mock_supabase.table.assert_called_with("medication_logs")


@pytest.mark.asyncio
async def test_update_missed_medications_with_pending():
    """Test updating missed medications when there are pending logs past due."""
    mock_supabase = Mock()
    
    # Mock data for pending medications
    pending_logs = [
        {
            "id": "log-1",
            "scheduled_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            "patient_id": "patient-1",
            "medication_id": "med-1"
        },
        {
            "id": "log-2",
            "scheduled_time": (datetime.now() - timedelta(hours=1)).isoformat(),
            "patient_id": "patient-1",
            "medication_id": "med-2"
        }
    ]
    
    # Create mock table for select query
    select_table = MockSupabaseTable(data=pending_logs)
    
    # Create mock table for update query
    update_table = MockSupabaseTable(data=[])
    
    # Configure mock to return different tables for select and update
    call_count = 0
    def table_side_effect(table_name):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return select_table
        else:
            return update_table
    
    mock_supabase.table.side_effect = table_side_effect
    
    scheduler = MedicationScheduler(mock_supabase)
    
    result = await scheduler.update_missed_medications()
    
    assert result == 2
    assert mock_supabase.table.call_count >= 3  # 1 select + 2 updates


@pytest.mark.asyncio
async def test_generate_daily_medication_logs_no_active():
    """Test generating logs when there are no active medications."""
    mock_supabase = Mock()
    mock_table = MockSupabaseTable(data=[])
    mock_supabase.table.return_value = mock_table
    
    scheduler = MedicationScheduler(mock_supabase)
    
    result = await scheduler.generate_daily_medication_logs()
    
    assert result == 0
    mock_supabase.table.assert_called_with("medications")


@pytest.mark.asyncio
async def test_generate_daily_medication_logs_with_active():
    """Test generating logs for active medications."""
    mock_supabase = Mock()
    
    today = datetime.now()
    
    # Mock active medications
    medications = [
        {
            "id": "med-1",
            "patient_id": "patient-1",
            "times": ["08:00:00", "20:00:00"],
            "start_date": (today - timedelta(days=1)).date().isoformat(),
            "end_date": None,
            "frequency": "daily"
        }
    ]
    
    # Track table calls
    table_calls = []
    
    def table_side_effect(table_name):
        table_calls.append(table_name)
        if table_name == "medications":
            return MockSupabaseTable(data=medications)
        else:  # medication_logs
            # Return empty for existing check, then accept insert
            return MockSupabaseTable(data=[])
    
    mock_supabase.table.side_effect = table_side_effect
    
    scheduler = MedicationScheduler(mock_supabase)
    
    result = await scheduler.generate_daily_medication_logs(today)
    
    assert result == 2  # Should create 2 logs (one for each time)
    assert "medications" in table_calls
    assert "medication_logs" in table_calls


@pytest.mark.asyncio
async def test_generate_daily_medication_logs_skip_existing():
    """Test that existing logs are not duplicated."""
    mock_supabase = Mock()
    
    today = datetime.now()
    
    # Mock active medications
    medications = [
        {
            "id": "med-1",
            "patient_id": "patient-1",
            "times": ["08:00:00"],
            "start_date": (today - timedelta(days=1)).date().isoformat(),
            "end_date": None,
            "frequency": "daily"
        }
    ]
    
    # Track which table is being queried
    call_count = 0
    
    def table_side_effect(table_name):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: return medications
            return MockSupabaseTable(data=medications)
        else:
            # Subsequent calls: return existing log (to simulate duplicate)
            return MockSupabaseTable(data=[{"id": "existing-log"}])
    
    mock_supabase.table.side_effect = table_side_effect
    
    scheduler = MedicationScheduler(mock_supabase)
    
    result = await scheduler.generate_daily_medication_logs(today)
    
    assert result == 0  # Should not create any logs (already exists)


@pytest.mark.asyncio
async def test_should_generate_for_date_daily():
    """Test frequency check for daily medications."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    today = datetime.now()
    
    assert scheduler._should_generate_for_date("daily", today) is True
    assert scheduler._should_generate_for_date("twice_daily", today) is True
    assert scheduler._should_generate_for_date("three_times_daily", today) is True


@pytest.mark.asyncio
async def test_should_generate_for_date_as_needed():
    """Test frequency check for as-needed medications."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    today = datetime.now()
    
    assert scheduler._should_generate_for_date("as_needed", today) is False


@pytest.mark.asyncio
async def test_scheduler_start_stop():
    """Test starting and stopping the scheduler."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    # Start scheduler
    await scheduler.start()
    assert scheduler._running is True
    assert len(scheduler._tasks) == 2
    
    # Stop scheduler
    await scheduler.stop()
    assert scheduler._running is False
    assert len(scheduler._tasks) == 0


@pytest.mark.asyncio
async def test_scheduler_start_already_running():
    """Test starting scheduler when it's already running."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    await scheduler.start()
    assert scheduler._running is True
    
    # Try to start again
    await scheduler.start()
    assert scheduler._running is True
    assert len(scheduler._tasks) == 2  # Should still have 2 tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
