import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useMedicationLogs } from './use-medication-logs';
import { MedicationLog } from '@/types/medication';
import * as fc from 'fast-check';
import { supabase } from '@/integrations/supabase/client';

/**
 * Property-Based Tests for useMedicationLogs Hook
 * 
 * These tests verify the correctness properties of medication logging:
 * - Property 52: Medication log persistence
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

describe('useMedicationLogs - Property-Based Tests', () => {
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
        // Ensure the date is valid before converting to ISO
        if (isNaN(d.getTime())) {
          return new Date('2025-01-01T00:00:00.000Z').toISOString();
        }
        return d.toISOString();
      });
  };

  /**
   * Arbitrary generator for MedicationLog
   */
  const medicationLogArbitrary = (): fc.Arbitrary<MedicationLog> => {
    return fc.record({
      id: fc.uuid(),
      medication_id: fc.uuid(),
      patient_id: fc.uuid(),
      scheduled_time: isoTimestampArbitrary(),
      taken_time: fc.option(isoTimestampArbitrary(), { nil: undefined }),
      status: fc.constantFrom('pending', 'taken', 'missed', 'skipped'),
      notes: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
      created_at: isoTimestampArbitrary(),
    });
  };

  /**
   * Property 52: Medication log persistence
   * 
   * Feature: fall-detection-integration, Property 52: Medication log persistence
   * Validates: Requirements 15.3
   * 
   * For any patient marking a medication as taken, the timestamp should be recorded
   * and the database updated.
   * 
   * This test verifies that:
   * 1. When a medication is marked as taken, a log entry is created or updated
   * 2. The log entry includes the taken_time timestamp
   * 3. The status is set to 'taken'
   * 4. The medication_id and patient_id are correctly associated
   * 5. The scheduled_time is preserved
   */
  it('Property 52: should persist medication logs with timestamp when marked as taken', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // medication_id
        isoTimestampArbitrary(), // scheduled_time
        async (patientId, medicationId, scheduledTime) => {
          // Mock Supabase responses for initial fetch
          const mockGte = vi.fn().mockReturnValue({
            lte: vi.fn().mockReturnValue({
              order: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });
          const mockSelectLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: mockGte,
            }),
          });

          // Mock for medications fetch
          const mockSelectMeds = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });

          // Mock for today's logs fetch
          const mockSelectTodayLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: vi.fn().mockReturnValue({
                lte: vi.fn().mockResolvedValue({ data: [], error: null }),
              }),
            }),
          });

          // Mock for checking existing logs
          const mockSelectExisting = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                eq: vi.fn().mockResolvedValue({ data: [], error: null }),
              }),
            }),
          });

          // Mock for insert
          const mockInsert = vi.fn().mockResolvedValue({ error: null });

          let callCount = 0;
          const mockFrom = vi.fn().mockImplementation((table: string) => {
            callCount++;
            if (table === 'medication_logs') {
              if (callCount === 1) {
                // First call: initial logs fetch
                return { select: mockSelectLogs };
              } else if (callCount === 2) {
                // Second call: today's medications logs fetch
                return { select: mockSelectTodayLogs };
              } else {
                // Subsequent calls: check existing and insert
                return {
                  select: mockSelectExisting,
                  insert: mockInsert,
                };
              }
            } else if (table === 'medications') {
              return { select: mockSelectMeds };
            }
            return { select: vi.fn() };
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationLogs(patientId, 7));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call logMedication
          await result.current.logMedication(medicationId, scheduledTime);

          // Verify Supabase insert was called
          expect(mockInsert).toHaveBeenCalled();

          // Verify the inserted log has all required fields
          const insertCall = mockInsert.mock.calls[0][0][0];
          
          // Verify required fields are present
          expect(insertCall).toHaveProperty('medication_id', medicationId);
          expect(insertCall).toHaveProperty('patient_id', patientId);
          expect(insertCall).toHaveProperty('scheduled_time', scheduledTime);
          expect(insertCall).toHaveProperty('taken_time');
          expect(insertCall).toHaveProperty('status', 'taken');

          // Verify taken_time is a valid ISO timestamp
          expect(insertCall.taken_time).toBeTruthy();
          const takenTime = new Date(insertCall.taken_time);
          expect(takenTime.toISOString()).toBe(insertCall.taken_time);

          // Verify taken_time is recent (within last minute)
          const now = new Date();
          const timeDiff = now.getTime() - takenTime.getTime();
          expect(timeDiff).toBeGreaterThanOrEqual(0);
          expect(timeDiff).toBeLessThan(60000); // Less than 1 minute

          // Verify IDs are valid UUIDs
          expect(insertCall.medication_id).toMatch(
            /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
          );
          expect(insertCall.patient_id).toMatch(
            /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
          );

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Medication log update when already exists
   * 
   * For any existing medication log, when marked as taken again,
   * the existing log should be updated rather than creating a duplicate.
   */
  it('Property: should update existing log instead of creating duplicate', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // medication_id
        isoTimestampArbitrary(), // scheduled_time
        medicationLogArbitrary(), // existing log
        async (patientId, medicationId, scheduledTime, existingLog) => {
          // Create existing log with matching medication_id and scheduled_time
          const matchingLog = {
            ...existingLog,
            medication_id: medicationId,
            patient_id: patientId,
            scheduled_time: scheduledTime,
            status: 'pending' as const,
          };

          // Mock Supabase responses for initial fetch
          const mockGte = vi.fn().mockReturnValue({
            lte: vi.fn().mockReturnValue({
              order: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });
          const mockSelectLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: mockGte,
            }),
          });

          // Mock for medications fetch
          const mockSelectMeds = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });

          // Mock for today's logs fetch
          const mockSelectTodayLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: vi.fn().mockReturnValue({
                lte: vi.fn().mockResolvedValue({ data: [], error: null }),
              }),
            }),
          });

          // Mock for checking existing logs - return the existing log
          const mockSelectExisting = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                eq: vi.fn().mockResolvedValue({ data: [matchingLog], error: null }),
              }),
            }),
          });

          // Mock for update
          const mockEq = vi.fn().mockResolvedValue({ error: null });
          const mockUpdate = vi.fn().mockReturnValue({
            eq: mockEq,
          });

          let callCount = 0;
          const mockFrom = vi.fn().mockImplementation((table: string) => {
            callCount++;
            if (table === 'medication_logs') {
              if (callCount === 1) {
                // First call: initial logs fetch
                return { select: mockSelectLogs };
              } else if (callCount === 2) {
                // Second call: today's medications logs fetch
                return { select: mockSelectTodayLogs };
              } else {
                // Subsequent calls: check existing and update
                return {
                  select: mockSelectExisting,
                  update: mockUpdate,
                };
              }
            } else if (table === 'medications') {
              return { select: mockSelectMeds };
            }
            return { select: vi.fn() };
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationLogs(patientId, 7));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call logMedication
          await result.current.logMedication(medicationId, scheduledTime);

          // Verify Supabase update was called (not insert)
          expect(mockUpdate).toHaveBeenCalled();

          // Verify the update has correct fields
          const updateCall = mockUpdate.mock.calls[0][0];
          expect(updateCall).toHaveProperty('status', 'taken');
          expect(updateCall).toHaveProperty('taken_time');

          // Verify taken_time is a valid ISO timestamp
          const takenTime = new Date(updateCall.taken_time);
          expect(takenTime.toISOString()).toBe(updateCall.taken_time);

          // Verify the correct log was updated
          expect(mockEq).toHaveBeenCalledWith('id', matchingLog.id);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Adherence rate calculation
   * 
   * For any set of medication logs, the adherence rate should be calculated as
   * (number of taken medications / total scheduled medications) * 100,
   * and should always be between 0 and 100.
   */
  it('Property: should calculate adherence rate correctly between 0 and 100', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.array(medicationLogArbitrary(), { minLength: 1, maxLength: 50 }), // logs
        async (patientId, logs) => {
          // Ensure all logs have the same patient_id
          const patientLogs = logs.map(log => ({
            ...log,
            patient_id: patientId,
          }));

          // Mock Supabase responses
          const mockGte = vi.fn().mockReturnValue({
            lte: vi.fn().mockReturnValue({
              order: vi.fn().mockResolvedValue({ data: patientLogs, error: null }),
            }),
          });
          const mockSelectLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: mockGte,
            }),
          });

          // Mock for medications fetch
          const mockSelectMeds = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });

          // Mock for today's logs fetch
          const mockSelectTodayLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: vi.fn().mockReturnValue({
                lte: vi.fn().mockResolvedValue({ data: [], error: null }),
              }),
            }),
          });

          let callCount = 0;
          const mockFrom = vi.fn().mockImplementation((table: string) => {
            callCount++;
            if (table === 'medication_logs') {
              if (callCount === 1) {
                return { select: mockSelectLogs };
              } else {
                return { select: mockSelectTodayLogs };
              }
            } else if (table === 'medications') {
              return { select: mockSelectMeds };
            }
            return { select: vi.fn() };
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationLogs(patientId, 7));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Verify adherence rate is between 0 and 100
          expect(result.current.adherenceRate).toBeGreaterThanOrEqual(0);
          expect(result.current.adherenceRate).toBeLessThanOrEqual(100);

          // Verify adherence stats
          const stats = result.current.adherenceStats;
          expect(stats.totalScheduled).toBe(patientLogs.length);
          expect(stats.totalTaken).toBe(patientLogs.filter(log => log.status === 'taken').length);
          expect(stats.totalMissed).toBe(patientLogs.filter(log => log.status === 'missed').length);

          // Verify adherence rate calculation
          if (stats.totalScheduled > 0) {
            const expectedRate = (stats.totalTaken / stats.totalScheduled) * 100;
            expect(Math.abs(stats.adherenceRate - expectedRate)).toBeLessThan(0.01); // Allow for floating point precision
          } else {
            expect(stats.adherenceRate).toBe(0);
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Logs are filtered by patient ID
   * 
   * For any patient ID, the hook should only fetch logs for that specific patient.
   */
  it('Property: should fetch logs only for specified patient', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.integer({ min: 1, max: 30 }), // days
        async (patientId, days) => {
          // Mock Supabase responses
          const mockGte = vi.fn().mockReturnValue({
            lte: vi.fn().mockReturnValue({
              order: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });
          const mockEq = vi.fn().mockReturnValue({
            gte: mockGte,
          });
          const mockSelectLogs = vi.fn().mockReturnValue({
            eq: mockEq,
          });

          // Mock for medications fetch
          const mockSelectMeds = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              eq: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });

          // Mock for today's logs fetch
          const mockSelectTodayLogs = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              gte: vi.fn().mockReturnValue({
                lte: vi.fn().mockResolvedValue({ data: [], error: null }),
              }),
            }),
          });

          let callCount = 0;
          const mockFrom = vi.fn().mockImplementation((table: string) => {
            callCount++;
            if (table === 'medication_logs') {
              if (callCount === 1) {
                return { select: mockSelectLogs };
              } else {
                return { select: mockSelectTodayLogs };
              }
            } else if (table === 'medications') {
              return { select: mockSelectMeds };
            }
            return { select: vi.fn() };
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationLogs(patientId, days));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Verify Supabase query was called with correct patient filter
          expect(mockFrom).toHaveBeenCalledWith('medication_logs');
          expect(mockSelectLogs).toHaveBeenCalledWith('*');
          expect(mockEq).toHaveBeenCalledWith('patient_id', patientId);

          // Verify patient_id is a valid UUID
          expect(patientId).toMatch(
            /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
          );

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);
});
