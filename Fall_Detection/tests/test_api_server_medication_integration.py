"""
Integration tests for Medication Scheduler with API Server

This module tests that the medication scheduler is properly integrated
into the API server lifecycle.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from medication_scheduler import MedicationScheduler


@pytest.mark.asyncio
async def test_medication_scheduler_initialization():
    """Test that medication scheduler can be initialized with a Supabase client."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    assert scheduler.supabase == mock_supabase
    assert scheduler._running is False
    assert len(scheduler._tasks) == 0


@pytest.mark.asyncio
async def test_medication_scheduler_lifecycle():
    """Test the full lifecycle of the medication scheduler."""
    mock_supabase = Mock()
    scheduler = MedicationScheduler(mock_supabase)
    
    # Start the scheduler
    await scheduler.start()
    assert scheduler._running is True
    assert len(scheduler._tasks) == 2  # Status update + log generation tasks
    
    # Stop the scheduler
    await scheduler.stop()
    assert scheduler._running is False
    assert len(scheduler._tasks) == 0


@pytest.mark.asyncio
async def test_scheduler_integrates_with_api_server_startup():
    """Test that scheduler can be started during API server initialization."""
    mock_supabase = Mock()
    
    # Simulate API server startup
    medication_scheduler = MedicationScheduler(mock_supabase)
    await medication_scheduler.start()
    
    # Verify scheduler is running
    assert medication_scheduler._running is True
    
    # Simulate API server shutdown
    await medication_scheduler.stop()
    
    # Verify scheduler is stopped
    assert medication_scheduler._running is False


@pytest.mark.asyncio
async def test_scheduler_handles_supabase_client_correctly():
    """Test that scheduler uses the Supabase client for database operations."""
    mock_supabase = Mock()
    
    # Mock the table method to return a mock table
    mock_table = Mock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.lt.return_value = mock_table
    mock_table.lte.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])
    
    mock_supabase.table.return_value = mock_table
    
    scheduler = MedicationScheduler(mock_supabase)
    
    # Test update_missed_medications uses the client
    await scheduler.update_missed_medications()
    mock_supabase.table.assert_called_with("medication_logs")
    
    # Test generate_daily_medication_logs uses the client
    await scheduler.generate_daily_medication_logs()
    # Should be called with both "medications" and "medication_logs"
    assert any(call[0][0] == "medications" for call in mock_supabase.table.call_args_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
