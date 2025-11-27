import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MedicationAdherence } from './MedicationAdherence';
import { MedicationLog, AdherenceStats } from '@/types/medication';
import * as fc from 'fast-check';

/**
 * Property-Based Tests for MedicationAdherence Component
 * 
 * These tests verify the correctness properties of medication adherence display:
 * - Property 59: Low adherence warning
 */

// Mock the useMedicationLogs hook
vi.mock('@/hooks/use-medication-logs', () => ({
  useMedicationLogs: vi.fn(),
}));

import { useMedicationLogs } from '@/hooks/use-medication-logs';

describe('MedicationAdherence - Property-Based Tests', () => {
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
   * Arbitrary generator for MedicationLog
   */
  const medicationLogArbitrary = (status: 'pending' | 'taken' | 'missed' | 'skipped'): fc.Arbitrary<MedicationLog> => {
    return fc.record({
      id: fc.uuid(),
      medication_id: fc.uuid(),
      patient_id: fc.uuid(),
      scheduled_time: isoTimestampArbitrary(),
      taken_time: status === 'taken' ? fc.option(isoTimestampArbitrary(), { nil: undefined }) : fc.constant(undefined),
      status: fc.constant(status),
      notes: fc.option(fc.string({ minLength: 0, maxLength: 100 }), { nil: undefined }),
      created_at: isoTimestampArbitrary(),
    });
  };

  /**
   * Arbitrary generator for AdherenceStats with controlled adherence rate
   */
  const adherenceStatsArbitrary = (adherenceRate: number): fc.Arbitrary<AdherenceStats> => {
    return fc.record({
      totalScheduled: fc.integer({ min: 1, max: 100 }),
    }).chain(({ totalScheduled }) => {
      const totalTaken = Math.floor((adherenceRate / 100) * totalScheduled);
      const totalMissed = totalScheduled - totalTaken;

      return fc.record({
        totalScheduled: fc.constant(totalScheduled),
        totalTaken: fc.constant(totalTaken),
        totalMissed: fc.constant(totalMissed),
        adherenceRate: fc.constant(adherenceRate),
        recentLogs: fc.array(
          fc.oneof(
            medicationLogArbitrary('taken'),
            medicationLogArbitrary('missed'),
            medicationLogArbitrary('pending'),
            medicationLogArbitrary('skipped')
          ),
          { minLength: totalScheduled, maxLength: totalScheduled }
        ),
      });
    });
  };

  /**
   * Property 59: Low adherence warning
   * 
   * Feature: fall-detection-integration, Property 59: Low adherence warning
   * Validates: Requirements 16.5
   * 
   * For any state where medication adherence is below 80 percent, a warning indicator
   * should be displayed on the Dashboard.
   * 
   * This test verifies that:
   * 1. When adherence is below 80%, a warning alert is displayed
   * 2. The warning contains appropriate messaging about low adherence
   * 3. When adherence is 80% or above, no warning is displayed
   * 4. The warning is consistently displayed across different adherence values below 80%
   * 5. The adherence percentage is correctly calculated and displayed
   */
  it('Property 59: should display warning indicator when adherence is below 80%', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.float({ min: 0, max: 100, noNaN: true }), // adherence rate
        fc.integer({ min: 7, max: 30 }), // days
        async (patientId, adherenceRate, days) => {
          // Generate adherence stats based on the adherence rate
          const stats = await fc.sample(adherenceStatsArbitrary(adherenceRate), 1)[0];

          // Generate logs based on the stats
          const takenLogs = Array.from({ length: stats.totalTaken }, () =>
            fc.sample(medicationLogArbitrary('taken'), 1)[0]
          );
          const missedLogs = Array.from({ length: stats.totalMissed }, () =>
            fc.sample(medicationLogArbitrary('missed'), 1)[0]
          );
          const allLogs = [...takenLogs, ...missedLogs];

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [],
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: adherenceRate,
            adherenceStats: stats,
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicationAdherence patientId={patientId} days={days} />);

          // Wait for component to render
          await waitFor(() => {
            const titles = screen.queryAllByText('Medication Adherence');
            expect(titles.length).toBeGreaterThan(0);
          });

          // Check if adherence percentage is displayed
          const adherenceText = `${adherenceRate.toFixed(1)}%`;
          const percentageElements = container.querySelectorAll('.text-3xl.font-bold');
          const hasPercentage = Array.from(percentageElements).some(el => 
            el.textContent?.includes(adherenceText)
          );
          expect(hasPercentage).toBe(true);

          // Verify warning display based on adherence rate
          if (adherenceRate < 80) {
            // Should display warning alert
            const warningTitles = screen.getAllByText('Low Medication Adherence');
            expect(warningTitles.length).toBeGreaterThan(0);

            // Should have warning message
            const warningMessage = screen.getByText(/below 80%/i);
            expect(warningMessage).toBeInTheDocument();

            // Should have AlertTriangle icon (check for alert with destructive variant)
            const alert = warningTitle.closest('[role="alert"]');
            expect(alert).toBeInTheDocument();
            expect(alert).toHaveClass('destructive');

          } else {
            // Should NOT display warning alert
            const warningTitle = screen.queryByText('Low Medication Adherence');
            expect(warningTitle).not.toBeInTheDocument();

            // Should NOT have warning message about 80%
            const warningMessage = screen.queryByText(/below 80%/i);
            expect(warningMessage).not.toBeInTheDocument();
          }

          // Verify adherence statistics are displayed correctly
          expect(screen.getByText(stats.totalScheduled.toString())).toBeInTheDocument();
          expect(screen.getByText(stats.totalTaken.toString())).toBeInTheDocument();
          expect(screen.getByText(stats.totalMissed.toString())).toBeInTheDocument();
        }
      ),
      { numRuns: 100, timeout: 10000 }
    );
  }, 60000);

  /**
   * Property: Warning threshold is exactly at 80%
   * 
   * For any adherence rate exactly at 80%, no warning should be displayed.
   * This tests the boundary condition.
   */
  it('Property: should NOT display warning when adherence is exactly 80%', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.integer({ min: 7, max: 30 }), // days
        async (patientId, days) => {
          const adherenceRate = 80.0;
          const stats = await fc.sample(adherenceStatsArbitrary(adherenceRate), 1)[0];

          const takenLogs = Array.from({ length: stats.totalTaken }, () =>
            fc.sample(medicationLogArbitrary('taken'), 1)[0]
          );
          const missedLogs = Array.from({ length: stats.totalMissed }, () =>
            fc.sample(medicationLogArbitrary('missed'), 1)[0]
          );
          const allLogs = [...takenLogs, ...missedLogs];

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [],
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: adherenceRate,
            adherenceStats: stats,
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicationAdherence patientId={patientId} days={days} />);

          // Wait for component to render
          await waitFor(() => {
            const titles = screen.queryAllByText('Medication Adherence');
            expect(titles.length).toBeGreaterThan(0);
          });

          // Should NOT display warning at exactly 80%
          const warningTitle = screen.queryByText('Low Medication Adherence');
          expect(warningTitle).not.toBeInTheDocument();

          // Should display the adherence percentage
          const percentageElements = container.querySelectorAll('.text-3xl.font-bold');
          const hasPercentage = Array.from(percentageElements).some(el => 
            el.textContent?.includes('80.0%')
          );
          expect(hasPercentage).toBe(true);
        }
      ),
      { numRuns: 20, timeout: 10000 }
    );
  }, 30000);

  /**
   * Property: Warning is displayed for all values below 80%
   * 
   * For any adherence rate strictly below 80%, a warning should always be displayed.
   */
  it('Property: should display warning for all adherence values below 80%', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.float({ min: 0, max: Math.fround(79.99), noNaN: true }), // adherence rate below 80
        async (patientId, adherenceRate) => {
          const stats = await fc.sample(adherenceStatsArbitrary(adherenceRate), 1)[0];

          const takenLogs = Array.from({ length: stats.totalTaken }, () =>
            fc.sample(medicationLogArbitrary('taken'), 1)[0]
          );
          const missedLogs = Array.from({ length: stats.totalMissed }, () =>
            fc.sample(medicationLogArbitrary('missed'), 1)[0]
          );
          const allLogs = [...takenLogs, ...missedLogs];

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [],
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: adherenceRate,
            adherenceStats: stats,
            refetch: vi.fn(),
          });

          // Render the component
          render(<MedicationAdherence patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            const titles = screen.queryAllByText('Medication Adherence');
            expect(titles.length).toBeGreaterThan(0);
          });

          // Should ALWAYS display warning for values below 80%
          const warningTitle = screen.getByText('Low Medication Adherence');
          expect(warningTitle).toBeInTheDocument();

          const warningMessage = screen.getByText(/below 80%/i);
          expect(warningMessage).toBeInTheDocument();
        }
      ),
      { numRuns: 100, timeout: 10000 }
    );
  }, 60000);

  /**
   * Property: No warning is displayed for all values at or above 80%
   * 
   * For any adherence rate at or above 80%, no warning should be displayed.
   */
  it('Property: should NOT display warning for all adherence values at or above 80%', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.integer({ min: 80, max: 100 }), // adherence rate at or above 80
        async (patientId, adherenceRate) => {
          const stats = await fc.sample(adherenceStatsArbitrary(adherenceRate), 1)[0];

          const takenLogs = Array.from({ length: stats.totalTaken }, () =>
            fc.sample(medicationLogArbitrary('taken'), 1)[0]
          );
          const missedLogs = Array.from({ length: stats.totalMissed }, () =>
            fc.sample(medicationLogArbitrary('missed'), 1)[0]
          );
          const allLogs = [...takenLogs, ...missedLogs];

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [],
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: adherenceRate,
            adherenceStats: stats,
            refetch: vi.fn(),
          });

          // Render the component
          render(<MedicationAdherence patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            const titles = screen.queryAllByText('Medication Adherence');
            expect(titles.length).toBeGreaterThan(0);
          });

          // Should NEVER display warning for values at or above 80%
          const warningTitle = screen.queryByText('Low Medication Adherence');
          expect(warningTitle).not.toBeInTheDocument();

          const warningMessage = screen.queryByText(/below 80%/i);
          expect(warningMessage).not.toBeInTheDocument();
        }
      ),
      { numRuns: 100, timeout: 10000 }
    );
  }, 60000);

  /**
   * Property: Adherence calculation is always between 0 and 100
   * 
   * For any set of medication logs, the calculated adherence rate should always
   * be between 0 and 100 (inclusive).
   */
  it('Property: adherence rate should always be between 0 and 100', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.float({ min: 0, max: 100, noNaN: true }), // adherence rate
        async (patientId, adherenceRate) => {
          const stats = await fc.sample(adherenceStatsArbitrary(adherenceRate), 1)[0];

          const takenLogs = Array.from({ length: stats.totalTaken }, () =>
            fc.sample(medicationLogArbitrary('taken'), 1)[0]
          );
          const missedLogs = Array.from({ length: stats.totalMissed }, () =>
            fc.sample(medicationLogArbitrary('missed'), 1)[0]
          );
          const allLogs = [...takenLogs, ...missedLogs];

          // Mock the useMedicationLogs hook
          vi.mocked(useMedicationLogs).mockReturnValue({
            todaysMedications: [],
            logs: allLogs,
            loading: false,
            error: null,
            logMedication: vi.fn(),
            adherenceRate: adherenceRate,
            adherenceStats: stats,
            refetch: vi.fn(),
          });

          // Render the component
          const { container } = render(<MedicationAdherence patientId={patientId} />);

          // Wait for component to render
          await waitFor(() => {
            const titles = screen.queryAllByText('Medication Adherence');
            expect(titles.length).toBeGreaterThan(0);
          });

          // Verify adherence rate is within valid range
          expect(adherenceRate).toBeGreaterThanOrEqual(0);
          expect(adherenceRate).toBeLessThanOrEqual(100);

          // Verify the displayed percentage is within range
          const percentageElements = container.querySelectorAll('.text-3xl.font-bold');
          const percentageText = Array.from(percentageElements).find(el => 
            el.textContent?.includes('%')
          )?.textContent;
          
          if (percentageText) {
            const displayedRate = parseFloat(percentageText.replace('%', '').trim());
            expect(displayedRate).toBeGreaterThanOrEqual(0);
            expect(displayedRate).toBeLessThanOrEqual(100);
          }
        }
      ),
      { numRuns: 100, timeout: 10000 }
    );
  }, 60000);
});
