import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MedicineChecklist } from './MedicineChecklist';
import { MedicationWithStatus, MedicationLog } from '@/types/medication';
import * as fc from 'fast-check';
import { supabase } from '@/integrations/supabase/client';

/**
 * Property-Based Tests for MedicineChecklist Component
 * 
 * These tests verify the correctness properties of medication checklist UI:
 * - Property 53: Medication completion visual feedback
 */

// Mock the Supabase client
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    from: vi.fn(),
    channel: vi.fn(),
    removeChannel: vi.fn(),
  },
}));

// Mock toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock the useMedicationLogs hook
vi.mock('@/hooks/use-medication-logs', () => ({
  useMedicationLogs: vi.fn(),
}));

import { useMedicationLogs } from '@/hooks/use-medication-logs';

describe('MedicineChecklist - Property-Based Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * Arbitrary generator for valid ISO timestamp strings
   */
  const isoTimestampArbitrary = (): fc.Arbitrary<string> => {
    return fc.date({ min: new Date('2020-01-01T00:00:00.000Z'), max: new Date('2030-12-31T23:59:59.999Z') })
      .map(d => {
        if (isNaN(d.getTime())) {
          return new Date('2025-01-01T00:00:00.000Z').toISOString();
        }
        return d.toISOString();
      });
  };

  /**
   * Arbitrary generator for time strings in HH:mm format
   */
  const timeStringArbitrary = (): fc.Arbitrary<string> => {
    return fc.record({
      hour: fc.integer({ min: 0, max: 23 }),
      minute: fc.integer({ min: 0, max: 59 }),
    }).map(({ hour, minute }) => 
      `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`
    );
  };

  /**
   * Arbitrary generator for MedicationLog
   */
  const medicationLogArbitrary = (medicationId: string, time: string, status: 'pending' | 'taken' | 'missed' | 'skipped'): fc.Arbitrary<MedicationLog> => {
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];
    const scheduledTime = `${todayStr}T${time}:00.000Z`;

    return fc.record({
      id: fc.uuid(),
      medication_id: fc.constant(medicationId),
      patient_id: fc.uuid(),
      scheduled_time: fc.constant(scheduledTime),
      taken_time: status === 'taken' ? fc.option(isoTimestampArbitrary(), { nil: undefined }) : fc.constant(undefined),
      status: fc.constant(status),
      notes: fc.option(fc.string({ minLength: 0, maxLength: 100 }), { nil: undefined }),
      created_at: isoTimestampArbitrary(),
    });
  };

  /**
   * Arbitrary generator for MedicationWithStatus
   */
  const medicationWithStatusArbitrary = (): fc.Arbitrary<MedicationWithStatus> => {
    return fc.record({
      id: fc.uuid(),
      patient_id: fc.uuid(),
      caregiver_id: fc.uuid(),
      name: fc.string({ minLength: 1, maxLength: 50 }).filter(s => s.trim().length > 0),
      dosage: fc.string({ minLength: 1, maxLength: 30 }).filter(s => s.trim().length > 0),
      frequency: fc.constantFrom('daily', 'twice_daily', 'three_times_daily', 'weekly', 'as_needed'),
      times: fc.array(timeStringArbitrary(), { minLength: 1, maxLength: 3 }),
      instructions: fc.option(fc.string({ minLength: 0, maxLength: 200 }), { nil: undefined }),
      start_date: fc.date({ min: new Date('2020-01-01'), max: new Date('2030-12-31') })
        .map(d => d.toISOString().split('T')[0]),
      end_date: fc.option(
        fc.date({ min: new Date('2020-01-01'), max: new Date('2030-12-31') })
          .map(d => d.toISOString().split('T')[0]),
        { nil: undefined }
      ),
      is_active: fc.constant(true),
      created_at: isoTimestampArbitrary(),
      updated_at: isoTimestampArbitrary(),
      todaysLogs: fc.constant([] as MedicationLog[]),
      nextScheduledTime: fc.option(isoTimestampArbitrary(), { nil: undefined }),
      lastTakenTime: fc.option(isoTimestampArbitrary(), { nil: undefined }),
    });
  };

  /**
   * Property 53: Medication completion visual feedback
   * 
   * Feature: fall-detection-integration, Property 53: Medication completion visual feedback
   * Validates: Requirements 15.4
   * 
   * For any medication marked as taken, visual indication (checkmark or strikethrough)
   * should be displayed.
   * 
   * This test verifies that:
   * 1. When a medication is marked as taken, a checkmark badge is displayed
   * 2. The medication time text has a strikethrough style applied
   * 3. The checkbox is checked and disabled
   * 4. The visual feedback is consistent across all taken medications
   * 5. Pending medications do not show the taken visual feedback
   */
  it('Property 53: should display visual feedback (checkmark and strikethrough) for taken medications', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        medicationWithStatusArbitrary(), // medication
        fc.constantFrom('taken', 'pending', 'missed'), // status to test
        async (patientId, medication, testStatus) => {
          // Create a log for the first time slot with the test status
          const firstTime = medication.times[0];
          const today = new Date();
          const todayStr = today.toISOString().split('T')[0];
          const scheduledTime = `${todayStr}T${firstTime}:00.000Z`;

          const log: MedicationLog = {
            id: 'test-log-id',
            medication_id: medication.id,
            patient_id: patientId,
            scheduled_time: scheduledTime,
            taken_time: testStatus === 'taken' ? new Date().toISOString() : undefined,
            status: testStatus,
            notes: undefined,
            created_at: new Date().toISOString(),
          };

          const medicationWithLog: MedicationWithStatus = {
            ...medication,
            todaysLogs: [log],
          };

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [medicationWithLog],
            logs: [log],
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: 100,
            adherenceStats: {
              totalScheduled: 1,
              totalTaken: testStatus === 'taken' ? 1 : 0,
              totalMissed: testStatus === 'missed' ? 1 : 0,
              adherenceRate: testStatus === 'taken' ? 100 : 0,
              recentLogs: [log],
            },
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicineChecklist patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            expect(screen.getByText(medication.name)).toBeInTheDocument();
          });

          // Find the time element for the first time slot
          const timeElement = screen.getByText(firstTime);
          expect(timeElement).toBeInTheDocument();

          if (testStatus === 'taken') {
            // Verify visual feedback for taken medication

            // 1. Check for "Taken" badge with checkmark
            const takenBadge = screen.getByText('Taken');
            expect(takenBadge).toBeInTheDocument();

            // 2. Verify strikethrough is applied to the time label
            // The label should have the line-through class
            const label = timeElement.closest('label');
            expect(label).toHaveClass('line-through');

            // 3. Verify checkbox is checked
            const checkbox = container.querySelector(`input[type="checkbox"][id="${medication.id}-${firstTime}"]`);
            expect(checkbox).toBeInTheDocument();
            expect(checkbox).toBeChecked();
            expect(checkbox).toBeDisabled();

            // 4. Verify the container has the green background styling
            const timeContainer = timeElement.closest('.flex.items-center.justify-between');
            expect(timeContainer).toHaveClass('bg-green-50');

          } else {
            // Verify NO taken visual feedback for non-taken medications

            // 1. Should NOT have "Taken" badge
            const takenBadges = screen.queryAllByText('Taken');
            expect(takenBadges.length).toBe(0);

            // 2. Verify NO strikethrough on the time label
            const label = timeElement.closest('label');
            expect(label).not.toHaveClass('line-through');

            // 3. Verify checkbox is NOT checked
            const checkbox = container.querySelector(`input[type="checkbox"][id="${medication.id}-${firstTime}"]`);
            expect(checkbox).toBeInTheDocument();
            expect(checkbox).not.toBeChecked();
            expect(checkbox).not.toBeDisabled();

            // 4. Verify the container does NOT have the green background
            const timeContainer = timeElement.closest('.flex.items-center.justify-between');
            expect(timeContainer).not.toHaveClass('bg-green-50');
          }
        }
      ),
      { numRuns: 10, timeout: 10000 } // Reduced for UI tests
    );
  }, 30000);

  /**
   * Property: Multiple medications with mixed statuses show correct visual feedback
   * 
   * For any set of medications with different statuses, each medication should
   * display the correct visual feedback independently.
   */
  it('Property: should display correct visual feedback for multiple medications with mixed statuses', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.array(medicationWithStatusArbitrary(), { minLength: 2, maxLength: 5 }), // multiple medications
        async (patientId, medications) => {
          // Assign random statuses to each medication's first time slot
          const medicationsWithLogs: MedicationWithStatus[] = medications.map((med, index) => {
            const status = index % 2 === 0 ? 'taken' : 'pending';
            const firstTime = med.times[0];
            const today = new Date();
            const todayStr = today.toISOString().split('T')[0];
            const scheduledTime = `${todayStr}T${firstTime}:00.000Z`;

            const log: MedicationLog = {
              id: `log-${index}`,
              medication_id: med.id,
              patient_id: patientId,
              scheduled_time: scheduledTime,
              taken_time: status === 'taken' ? new Date().toISOString() : undefined,
              status: status,
              notes: undefined,
              created_at: new Date().toISOString(),
            };

            return {
              ...med,
              todaysLogs: [log],
            };
          });

          const allLogs = medicationsWithLogs.flatMap(m => m.todaysLogs);

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: medicationsWithLogs,
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: 50,
            adherenceStats: {
              totalScheduled: allLogs.length,
              totalTaken: allLogs.filter(l => l.status === 'taken').length,
              totalMissed: 0,
              adherenceRate: 50,
              recentLogs: allLogs,
            },
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicineChecklist patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            expect(screen.getByText(medications[0].name)).toBeInTheDocument();
          });

          // Verify each medication has correct visual feedback
          medicationsWithLogs.forEach((med, index) => {
            const status = index % 2 === 0 ? 'taken' : 'pending';
            const firstTime = med.times[0];
            const timeElement = screen.getByText(firstTime);

            if (status === 'taken') {
              // Should have strikethrough
              const label = timeElement.closest('label');
              expect(label).toHaveClass('line-through');

              // Should have checked checkbox
              const checkbox = container.querySelector(`input[type="checkbox"][id="${med.id}-${firstTime}"]`);
              expect(checkbox).toBeChecked();
            } else {
              // Should NOT have strikethrough
              const label = timeElement.closest('label');
              expect(label).not.toHaveClass('line-through');

              // Should NOT have checked checkbox
              const checkbox = container.querySelector(`input[type="checkbox"][id="${med.id}-${firstTime}"]`);
              expect(checkbox).not.toBeChecked();
            }
          });

          // Count "Taken" badges - should equal number of taken medications
          const takenCount = medicationsWithLogs.filter((_, i) => i % 2 === 0).length;
          const takenBadges = screen.getAllByText('Taken');
          expect(takenBadges.length).toBe(takenCount);
        }
      ),
      { numRuns: 10, timeout: 10000 } // Reduced for UI tests
    );
  }, 30000);

  /**
   * Property: Visual feedback is consistent across all time slots
   * 
   * For any medication with multiple time slots, if a time slot is marked as taken,
   * only that specific time slot should show the taken visual feedback.
   */
  it('Property: should show visual feedback only for the specific time slot that is taken', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        medicationWithStatusArbitrary(), // medication with multiple times
        fc.integer({ min: 0, max: 2 }), // index of time slot to mark as taken
        async (patientId, medication, takenIndex) => {
          // Ensure medication has at least takenIndex + 1 time slots
          if (medication.times.length <= takenIndex) {
            medication.times.push('14:00', '18:00', '22:00');
          }

          // Create logs for all time slots, marking one as taken
          const today = new Date();
          const todayStr = today.toISOString().split('T')[0];

          const logs: MedicationLog[] = medication.times.map((time, index) => {
            const scheduledTime = `${todayStr}T${time}:00.000Z`;
            const status = index === takenIndex ? 'taken' : 'pending';

            return {
              id: `log-${index}`,
              medication_id: medication.id,
              patient_id: patientId,
              scheduled_time: scheduledTime,
              taken_time: status === 'taken' ? new Date().toISOString() : undefined,
              status: status,
              notes: undefined,
              created_at: new Date().toISOString(),
            };
          });

          const medicationWithLogs: MedicationWithStatus = {
            ...medication,
            todaysLogs: logs,
          };

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [medicationWithLogs],
            logs: logs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: 100,
            adherenceStats: {
              totalScheduled: logs.length,
              totalTaken: 1,
              totalMissed: 0,
              adherenceRate: (1 / logs.length) * 100,
              recentLogs: logs,
            },
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicineChecklist patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            expect(screen.getByText(medication.name)).toBeInTheDocument();
          });

          // Verify each time slot has correct visual feedback
          medication.times.forEach((time, index) => {
            const timeElement = screen.getByText(time);
            const label = timeElement.closest('label');
            const checkbox = container.querySelector(`input[type="checkbox"][id="${medication.id}-${time}"]`);

            if (index === takenIndex) {
              // This time slot should show taken visual feedback
              expect(label).toHaveClass('line-through');
              expect(checkbox).toBeChecked();
              expect(checkbox).toBeDisabled();
            } else {
              // Other time slots should NOT show taken visual feedback
              expect(label).not.toHaveClass('line-through');
              expect(checkbox).not.toBeChecked();
              expect(checkbox).not.toBeDisabled();
            }
          });

          // Should have exactly one "Taken" badge
          const takenBadges = screen.getAllByText('Taken');
          expect(takenBadges.length).toBe(1);
        }
      ),
      { numRuns: 10, timeout: 10000 } // Reduced for UI tests
    );
  }, 30000);
});
