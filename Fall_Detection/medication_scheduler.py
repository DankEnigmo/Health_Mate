"""
Medication Scheduler Module

This module provides scheduled tasks for medication management:
1. Update medication status from 'pending' to 'missed' for overdue medications
2. Generate pending medication logs for upcoming medications

Requirements: 15.1, 15.3
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Any

from supabase import Client

LOG = logging.getLogger(__name__)


class MedicationScheduler:
    """
    Handles scheduled tasks for medication management.
    
    This class provides two main scheduled tasks:
    1. Update overdue medications to 'missed' status (runs every 15 minutes)
    2. Generate pending logs for upcoming medications (runs daily at midnight)
    """
    
    def __init__(self, supabase_client: Client):
        """
        Initialize the medication scheduler.
        
        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start all scheduled tasks."""
        if self._running:
            LOG.warning("Medication scheduler is already running")
            return
        
        self._running = True
        LOG.info("Starting medication scheduler...")
        
        # Start the status update task (every 15 minutes)
        status_task = asyncio.create_task(self._run_status_update_task())
        self._tasks.append(status_task)
        
        # Start the log generation task (daily at midnight)
        log_gen_task = asyncio.create_task(self._run_log_generation_task())
        self._tasks.append(log_gen_task)
        
        LOG.info("Medication scheduler started successfully")
    
    async def stop(self):
        """Stop all scheduled tasks."""
        if not self._running:
            return
        
        LOG.info("Stopping medication scheduler...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        LOG.info("Medication scheduler stopped")
    
    async def _run_status_update_task(self):
        """
        Run the status update task periodically.
        
        This task checks for medications past their scheduled time and updates
        their status from 'pending' to 'missed'. Runs every 15 minutes.
        
        Requirements: 15.3
        """
        LOG.info("Status update task started (runs every 15 minutes)")
        
        while self._running:
            try:
                await self.update_missed_medications()
                
                # Wait 15 minutes before next run
                await asyncio.sleep(15 * 60)
                
            except asyncio.CancelledError:
                LOG.info("Status update task cancelled")
                break
            except Exception as e:
                LOG.error(f"Error in status update task: {e}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(60)
    
    async def _run_log_generation_task(self):
        """
        Run the log generation task daily at midnight.
        
        This task generates pending medication logs for the upcoming day
        based on active medication schedules.
        
        Requirements: 15.1
        """
        LOG.info("Log generation task started (runs daily at midnight)")
        
        while self._running:
            try:
                # Calculate time until next midnight
                now = datetime.now()
                tomorrow = now + timedelta(days=1)
                next_midnight = datetime.combine(tomorrow.date(), time.min)
                seconds_until_midnight = (next_midnight - now).total_seconds()
                
                LOG.info(f"Next log generation in {seconds_until_midnight / 3600:.2f} hours")
                
                # Wait until midnight
                await asyncio.sleep(seconds_until_midnight)
                
                # Generate logs for the new day
                await self.generate_daily_medication_logs()
                
            except asyncio.CancelledError:
                LOG.info("Log generation task cancelled")
                break
            except Exception as e:
                LOG.error(f"Error in log generation task: {e}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(60)
    
    async def update_missed_medications(self) -> int:
        """
        Update medication logs from 'pending' to 'missed' for overdue medications.
        
        This method finds all medication logs with status 'pending' where the
        scheduled_time is in the past, and updates their status to 'missed'.
        
        Returns:
            Number of medication logs updated
            
        Requirements: 15.3
        """
        try:
            current_time = datetime.now().isoformat()
            
            LOG.info(f"Checking for missed medications (current time: {current_time})")
            
            # Query for pending medications that are past their scheduled time
            response = self.supabase.table("medication_logs") \
                .select("id, scheduled_time, patient_id, medication_id") \
                .eq("status", "pending") \
                .lt("scheduled_time", current_time) \
                .execute()
            
            pending_logs = response.data if response.data else []
            
            if not pending_logs:
                LOG.debug("No missed medications found")
                return 0
            
            LOG.info(f"Found {len(pending_logs)} missed medications")
            
            # Update each log to 'missed' status
            updated_count = 0
            for log in pending_logs:
                try:
                    self.supabase.table("medication_logs") \
                        .update({"status": "missed"}) \
                        .eq("id", log["id"]) \
                        .execute()
                    
                    updated_count += 1
                    LOG.debug(f"Updated medication log {log['id']} to 'missed' status")
                    
                except Exception as e:
                    LOG.error(f"Failed to update medication log {log['id']}: {e}")
            
            LOG.info(f"Successfully updated {updated_count} medication logs to 'missed' status")
            return updated_count
            
        except Exception as e:
            LOG.error(f"Error updating missed medications: {e}", exc_info=True)
            return 0
    
    async def generate_daily_medication_logs(self, target_date: Optional[datetime] = None) -> int:
        """
        Generate pending medication logs for a specific day.
        
        This method fetches all active medication schedules and creates
        pending logs for each scheduled time on the target date.
        
        Args:
            target_date: Date to generate logs for (defaults to today)
            
        Returns:
            Number of medication logs created
            
        Requirements: 15.1
        """
        try:
            if target_date is None:
                target_date = datetime.now()
            
            target_date_str = target_date.date().isoformat()
            
            LOG.info(f"Generating medication logs for {target_date_str}")
            
            # Query for active medication schedules
            response = self.supabase.table("medications") \
                .select("id, patient_id, times, start_date, end_date, frequency") \
                .eq("is_active", True) \
                .lte("start_date", target_date_str) \
                .execute()
            
            medications = response.data if response.data else []
            
            # Filter medications that haven't ended yet
            active_medications = [
                med for med in medications
                if med.get("end_date") is None or med["end_date"] >= target_date_str
            ]
            
            if not active_medications:
                LOG.info("No active medications found")
                return 0
            
            LOG.info(f"Found {len(active_medications)} active medication schedules")
            
            created_count = 0
            
            for medication in active_medications:
                medication_id = medication["id"]
                patient_id = medication["patient_id"]
                times = medication.get("times", [])
                frequency = medication.get("frequency", "daily")
                
                # Skip if no times specified
                if not times:
                    LOG.warning(f"Medication {medication_id} has no scheduled times")
                    continue
                
                # Check if we should generate logs based on frequency
                if not self._should_generate_for_date(frequency, target_date):
                    LOG.debug(f"Skipping medication {medication_id} - not scheduled for {target_date_str}")
                    continue
                
                # Create a log for each scheduled time
                for scheduled_time_str in times:
                    try:
                        # Combine date and time
                        scheduled_datetime = datetime.combine(
                            target_date.date(),
                            datetime.strptime(scheduled_time_str, "%H:%M:%S").time()
                        )
                        
                        # Check if log already exists
                        existing = self.supabase.table("medication_logs") \
                            .select("id") \
                            .eq("medication_id", medication_id) \
                            .eq("patient_id", patient_id) \
                            .eq("scheduled_time", scheduled_datetime.isoformat()) \
                            .execute()
                        
                        if existing.data:
                            LOG.debug(f"Log already exists for medication {medication_id} at {scheduled_datetime}")
                            continue
                        
                        # Create new log
                        log_data = {
                            "medication_id": medication_id,
                            "patient_id": patient_id,
                            "scheduled_time": scheduled_datetime.isoformat(),
                            "status": "pending"
                        }
                        
                        self.supabase.table("medication_logs") \
                            .insert(log_data) \
                            .execute()
                        
                        created_count += 1
                        LOG.debug(f"Created log for medication {medication_id} at {scheduled_datetime}")
                        
                    except Exception as e:
                        LOG.error(f"Failed to create log for medication {medication_id} at {scheduled_time_str}: {e}")
            
            LOG.info(f"Successfully created {created_count} medication logs for {target_date_str}")
            return created_count
            
        except Exception as e:
            LOG.error(f"Error generating daily medication logs: {e}", exc_info=True)
            return 0
    
    def _should_generate_for_date(self, frequency: str, target_date: datetime) -> bool:
        """
        Determine if logs should be generated for a given date based on frequency.
        
        Args:
            frequency: Medication frequency (daily, twice_daily, etc.)
            target_date: Date to check
            
        Returns:
            True if logs should be generated for this date
        """
        # For daily frequencies, always generate
        if frequency in ['daily', 'twice_daily', 'three_times_daily']:
            return True
        
        # For weekly, only generate on the same day of week as start date
        # This is a simplified implementation - could be enhanced
        if frequency == 'weekly':
            # For now, generate on Mondays (weekday 0)
            return target_date.weekday() == 0
        
        # For as_needed, don't auto-generate
        if frequency == 'as_needed':
            return False
        
        # Default to generating
        return True
