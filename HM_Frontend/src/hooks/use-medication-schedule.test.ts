import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useMedicationSchedule } from './use-medication-schedule';
import { MedicationSchedule, MedicationFrequency } from '@/types/medication';
import * as fc from 'fast-check';
import { supabase } from '@/integrations/supabase/client';

/**
 * Property-Based Tests for useMedicationSchedule Hook
 * 
 * These tests verify the correctness properties of medication schedule management:
 * - Property 46: Medication schedule persistence
 * - Property 47: Medication schedule patient association
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

describe('useMedicationSchedule - Property-Based Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * Arbitrary generator for MedicationSchedule
   * Generates random but valid medication schedules for property testing
   */
  const medicationScheduleArbitrary = (): fc.Arbitrary<MedicationSchedule> => {
    return fc.record({
      id: fc.uuid(),
      patient_id: fc.uuid(),
      caregiver_id: fc.uuid(),
      name: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
      dosage: fc.string({ minLength: 1, maxLength: 50 }).filter(s => s.trim().length > 0),
      frequency: fc.constantFrom(
        'daily' as MedicationFrequency,
        'twice_daily' as MedicationFrequency,
        'three_times_daily' as MedicationFrequency,
        'weekly' as MedicationFrequency,
        'as_needed' as MedicationFrequency
      ),
      times: fc.array(
        fc.record({
          hour: fc.integer({ min: 0, max: 23 }),
          minute: fc.integer({ min: 0, max: 59 }),
        }).map(({ hour, minute }) => 
          `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`
        ),
        { minLength: 1, maxLength: 5 }
      ),
      instructions: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
      start_date: fc.date({ min: new Date('2020-01-01'), max: new Date('2030-12-31') })
        .map(d => {
          const year = d.getFullYear();
          const month = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          return `${year}-${month}-${day}`;
        }),
      end_date: fc.option(
        fc.date({ min: new Date('2020-01-01'), max: new Date('2030-12-31') })
          .map(d => {
            const year = d.getFullYear();
            const month = String(d.getMonth() + 1).padStart(2, '0');
            const day = String(d.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          }),
        { nil: undefined }
      ),
      is_active: fc.boolean(),
      created_at: fc.date({ min: new Date('2020-01-01'), max: new Date('2025-11-24') })
        .map(d => {
          const year = d.getFullYear();
          const month = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          const hours = String(d.getHours()).padStart(2, '0');
          const minutes = String(d.getMinutes()).padStart(2, '0');
          const seconds = String(d.getSeconds()).padStart(2, '0');
          return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.000Z`;
        }),
      updated_at: fc.date({ min: new Date('2020-01-01'), max: new Date('2025-11-24') })
        .map(d => {
          const year = d.getFullYear();
          const month = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          const hours = String(d.getHours()).padStart(2, '0');
          const minutes = String(d.getMinutes()).padStart(2, '0');
          const seconds = String(d.getSeconds()).padStart(2, '0');
          return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.000Z`;
        }),
    });
  };

  /**
   * Property 46: Medication schedule persistence
   * 
   * Feature: fall-detection-integration, Property 46: Medication schedule persistence
   * Validates: Requirements 14.2
   * 
   * For any caregiver-created medicine schedule, it should be stored in the database
   * with medicine name, dosage, frequency, and time.
   * 
   * This test verifies that:
   * 1. All required fields are included when creating a medication schedule
   * 2. The medication data is correctly formatted for database insertion
   * 3. The createMedication function properly calls the Supabase insert method
   * 4. Required fields (name, dosage, frequency, times) are present and valid
   */
  it('Property 46: should persist medication schedules with all required fields (name, dosage, frequency, times)', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // caregiver_id
        medicationScheduleArbitrary(), // medication template
        async (patientId, caregiverId, medTemplate) => {
          // Create medication data without id, created_at, updated_at
          const medData: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'> = {
            patient_id: patientId,
            caregiver_id: caregiverId,
            name: medTemplate.name,
            dosage: medTemplate.dosage,
            frequency: medTemplate.frequency,
            times: medTemplate.times,
            instructions: medTemplate.instructions,
            start_date: medTemplate.start_date,
            end_date: medTemplate.end_date,
            is_active: true,
          };

          // Mock Supabase responses
          const mockInsert = vi.fn().mockResolvedValue({ error: null });
          const mockFrom = vi.fn().mockReturnValue({
            insert: mockInsert,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                eq: vi.fn().mockReturnValue({
                  order: vi.fn().mockResolvedValue({ data: [], error: null }),
                }),
              }),
            }),
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationSchedule(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call createMedication
          await result.current.createMedication(medData);

          // Verify Supabase insert was called with correct data
          expect(mockInsert).toHaveBeenCalledWith([medData]);

          // Verify all required fields are present in the call
          const insertCall = mockInsert.mock.calls[0][0][0];
          
          // Verify required fields
          expect(insertCall).toHaveProperty('name');
          expect(insertCall.name.trim().length).toBeGreaterThan(0);
          
          expect(insertCall).toHaveProperty('dosage');
          expect(insertCall.dosage.trim().length).toBeGreaterThan(0);
          
          expect(insertCall).toHaveProperty('frequency');
          expect(['daily', 'twice_daily', 'three_times_daily', 'weekly', 'as_needed'])
            .toContain(insertCall.frequency);
          
          expect(insertCall).toHaveProperty('times');
          expect(Array.isArray(insertCall.times)).toBe(true);
          expect(insertCall.times.length).toBeGreaterThan(0);
          
          // Verify all times are in correct format (HH:mm)
          for (const time of insertCall.times) {
            expect(time).toMatch(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/);
          }

          // Verify start_date is in correct format (YYYY-MM-DD)
          expect(insertCall).toHaveProperty('start_date');
          expect(insertCall.start_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);

          // If end_date is present, verify format
          if (insertCall.end_date) {
            expect(insertCall.end_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
          }

          // Verify is_active is boolean
          expect(insertCall).toHaveProperty('is_active');
          expect(typeof insertCall.is_active).toBe('boolean');

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 } // Run 100 iterations with 30s timeout
    );
  }, 60000); // 60 second test timeout

  /**
   * Property 47: Medication schedule patient association
   * 
   * Feature: fall-detection-integration, Property 47: Medication schedule patient association
   * Validates: Requirements 14.3
   * 
   * For any created medicine schedule, it must be associated with the patient ID.
   * 
   * This test verifies that:
   * 1. The patient_id is always included when creating a medication schedule
   * 2. The patient_id matches the intended patient
   * 3. The hook fetches medications only for the specified patient
   * 4. The patient association is maintained throughout the lifecycle
   */
  it('Property 47: should associate medication schedules with patient ID', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // caregiver_id
        medicationScheduleArbitrary(), // medication template
        async (patientId, caregiverId, medTemplate) => {
          // Create medication data
          const medData: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'> = {
            patient_id: patientId,
            caregiver_id: caregiverId,
            name: medTemplate.name,
            dosage: medTemplate.dosage,
            frequency: medTemplate.frequency,
            times: medTemplate.times,
            instructions: medTemplate.instructions,
            start_date: medTemplate.start_date,
            end_date: medTemplate.end_date,
            is_active: true,
          };

          // Mock Supabase responses
          const mockInsert = vi.fn().mockResolvedValue({ error: null });
          const mockEq1 = vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              order: vi.fn().mockResolvedValue({ data: [], error: null }),
            }),
          });
          const mockSelect = vi.fn().mockReturnValue({
            eq: mockEq1,
          });
          const mockFrom = vi.fn().mockReturnValue({
            insert: mockInsert,
            select: mockSelect,
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationSchedule(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Verify initial fetch filters by patient_id
          expect(mockFrom).toHaveBeenCalledWith('medications');
          expect(mockSelect).toHaveBeenCalledWith('*');
          expect(mockEq1).toHaveBeenCalledWith('patient_id', patientId);

          // Call createMedication
          await result.current.createMedication(medData);

          // Verify the inserted medication has the correct patient_id
          const insertCall = mockInsert.mock.calls[0][0][0];
          expect(insertCall).toHaveProperty('patient_id', patientId);
          
          // Verify patient_id is a valid UUID
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
   * Property: Medication update preserves required fields
   * 
   * For any medication update, the patient_id and caregiver_id should remain unchanged,
   * and all required fields should still be present.
   */
  it('Property: should preserve required fields when updating medication schedules', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        medicationScheduleArbitrary(), // original medication
        medicationScheduleArbitrary(), // updated medication data
        async (patientId, originalMed, updatedData) => {
          // Mock Supabase responses
          const mockEq = vi.fn().mockResolvedValue({ error: null });
          const mockUpdate = vi.fn().mockReturnValue({
            eq: mockEq,
          });
          const mockFrom = vi.fn().mockReturnValue({
            update: mockUpdate,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                eq: vi.fn().mockReturnValue({
                  order: vi.fn().mockResolvedValue({ data: [originalMed], error: null }),
                }),
              }),
            }),
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationSchedule(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Create updated medication with new data but same IDs
          const updatedMed: MedicationSchedule = {
            ...originalMed,
            name: updatedData.name,
            dosage: updatedData.dosage,
            frequency: updatedData.frequency,
            times: updatedData.times,
            instructions: updatedData.instructions,
            start_date: updatedData.start_date,
            end_date: updatedData.end_date,
          };

          // Call updateMedication
          await result.current.updateMedication(updatedMed);

          // Verify Supabase update was called
          expect(mockUpdate).toHaveBeenCalled();

          // Verify the update call contains all required fields
          const updateCall = mockUpdate.mock.calls[0][0];
          expect(updateCall).toHaveProperty('name');
          expect(updateCall.name.trim().length).toBeGreaterThan(0);
          expect(updateCall).toHaveProperty('dosage');
          expect(updateCall.dosage.trim().length).toBeGreaterThan(0);
          expect(updateCall).toHaveProperty('frequency');
          expect(updateCall).toHaveProperty('times');
          expect(Array.isArray(updateCall.times)).toBe(true);
          expect(updateCall.times.length).toBeGreaterThan(0);
          expect(updateCall).toHaveProperty('start_date');
          expect(updateCall).toHaveProperty('updated_at');

          // Verify updated_at is a valid ISO string
          expect(new Date(updateCall.updated_at).toISOString()).toBe(updateCall.updated_at);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Medication deletion (soft delete) sets is_active to false
   * 
   * For any medication deletion, the medication should be soft-deleted by
   * setting is_active to false, not hard-deleted from the database.
   */
  it('Property: should soft delete medications by setting is_active to false', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // medication_id to delete
        async (patientId, medId) => {
          // Mock Supabase responses
          const mockEq = vi.fn().mockResolvedValue({ error: null });
          const mockUpdate = vi.fn().mockReturnValue({
            eq: mockEq,
          });
          const mockFrom = vi.fn().mockReturnValue({
            update: mockUpdate,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                eq: vi.fn().mockReturnValue({
                  order: vi.fn().mockResolvedValue({ data: [], error: null }),
                }),
              }),
            }),
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationSchedule(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call deleteMedication
          await result.current.deleteMedication(medId);

          // Verify Supabase update (not delete) was called
          expect(mockUpdate).toHaveBeenCalled();
          
          // Verify is_active was set to false
          const updateCall = mockUpdate.mock.calls[0][0];
          expect(updateCall).toHaveProperty('is_active', false);
          expect(updateCall).toHaveProperty('updated_at');
          
          // Verify the correct medication ID was targeted
          expect(mockEq).toHaveBeenCalledWith('id', medId);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Hook fetches only active medications for correct patient
   * 
   * For any patient ID, the hook should only fetch active medications
   * (is_active = true) associated with that specific patient.
   */
  it('Property: should fetch only active medications for specified patient', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        async (patientId) => {
          // Mock Supabase responses
          const mockEq2 = vi.fn().mockReturnValue({
            order: vi.fn().mockResolvedValue({ data: [], error: null }),
          });
          const mockEq1 = vi.fn().mockReturnValue({
            eq: mockEq2,
          });
          const mockSelect = vi.fn().mockReturnValue({
            eq: mockEq1,
          });
          const mockFrom = vi.fn().mockReturnValue({
            select: mockSelect,
          });

          vi.mocked(supabase.from).mockImplementation(mockFrom as any);

          // Mock channel subscription
          const mockChannel = {
            on: vi.fn().mockReturnThis(),
            subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
          };
          vi.mocked(supabase.channel).mockReturnValue(mockChannel as any);

          // Render the hook
          const { result, unmount } = renderHook(() => useMedicationSchedule(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Verify Supabase query was called with correct filters
          expect(mockFrom).toHaveBeenCalledWith('medications');
          expect(mockSelect).toHaveBeenCalledWith('*');
          expect(mockEq1).toHaveBeenCalledWith('patient_id', patientId);
          expect(mockEq2).toHaveBeenCalledWith('is_active', true);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);
});
