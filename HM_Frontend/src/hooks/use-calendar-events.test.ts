import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useCalendarEvents } from './use-calendar-events';
import { CalendarEvent } from '@/types/calendar';
import * as fc from 'fast-check';
import { supabase } from '@/integrations/supabase/client';

/**
 * Property-Based Tests for useCalendarEvents Hook
 * 
 * These tests verify the correctness properties of calendar event management:
 * - Property 35: Event persistence with required fields
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

describe('useCalendarEvents - Property-Based Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * Arbitrary generator for CalendarEvent
   * Generates random but valid calendar events for property testing
   */
  const calendarEventArbitrary = (): fc.Arbitrary<CalendarEvent> => {
    return fc.record({
      id: fc.uuid(),
      patient_id: fc.uuid(),
      caregiver_id: fc.uuid(),
      title: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
      description: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
      event_date: fc.date({ min: new Date('2020-01-01'), max: new Date('2030-12-31') })
        .map(d => {
          const year = d.getFullYear();
          const month = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          return `${year}-${month}-${day}`;
        }),
      event_time: fc.option(
        fc.record({
          hour: fc.integer({ min: 0, max: 23 }),
          minute: fc.integer({ min: 0, max: 59 }),
        }).map(({ hour, minute }) => 
          `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`
        ),
        { nil: undefined }
      ),
      duration_minutes: fc.option(fc.integer({ min: 1, max: 1440 }), { nil: undefined }),
      event_type: fc.constantFrom('appointment' as const, 'medication' as const, 'reminder' as const, 'other' as const),
      location: fc.option(fc.string({ minLength: 0, maxLength: 200 }), { nil: undefined }),
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
   * Property 35: Event persistence with required fields
   * 
   * Feature: fall-detection-integration, Property 35: Event persistence with required fields
   * Validates: Requirements 11.3
   * 
   * For any caregiver-created event, it should be stored in the Supabase database
   * with patient ID and caregiver ID.
   * 
   * This test verifies that:
   * 1. All required fields are included when creating an event
   * 2. The patient_id and caregiver_id are properly associated
   * 3. The event data is correctly formatted for database insertion
   * 4. The createEvent function properly calls the Supabase insert method
   */
  it('Property 35: should persist events with all required fields (patient_id, caregiver_id)', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // caregiver_id
        calendarEventArbitrary(), // event template
        async (patientId, caregiverId, eventTemplate) => {
          // Create event data without id, created_at, updated_at
          const eventData: Omit<CalendarEvent, 'id' | 'created_at' | 'updated_at'> = {
            patient_id: patientId,
            caregiver_id: caregiverId,
            title: eventTemplate.title,
            description: eventTemplate.description,
            event_date: eventTemplate.event_date,
            event_time: eventTemplate.event_time,
            duration_minutes: eventTemplate.duration_minutes,
            event_type: eventTemplate.event_type,
            location: eventTemplate.location,
          };

          // Mock Supabase responses
          const mockInsert = vi.fn().mockResolvedValue({ error: null });
          const mockFrom = vi.fn().mockReturnValue({
            insert: mockInsert,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                order: vi.fn().mockResolvedValue({ data: [], error: null }),
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
          const { result, unmount } = renderHook(() => useCalendarEvents(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call createEvent
          await result.current.createEvent(eventData);

          // Verify Supabase insert was called with correct data
          expect(mockInsert).toHaveBeenCalledWith([eventData]);

          // Verify all required fields are present in the call
          const insertCall = mockInsert.mock.calls[0][0][0];
          expect(insertCall).toHaveProperty('patient_id', patientId);
          expect(insertCall).toHaveProperty('caregiver_id', caregiverId);
          expect(insertCall).toHaveProperty('title');
          expect(insertCall.title.trim().length).toBeGreaterThan(0);
          expect(insertCall).toHaveProperty('event_date');
          expect(insertCall).toHaveProperty('event_type');

          // Verify event_date is in correct format (YYYY-MM-DD)
          expect(insertCall.event_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);

          // Verify event_type is valid
          expect(['appointment', 'medication', 'reminder', 'other']).toContain(insertCall.event_type);

          // If event_time is present, verify format (HH:mm)
          if (insertCall.event_time) {
            expect(insertCall.event_time).toMatch(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/);
          }

          // If duration_minutes is present, verify it's a positive number
          if (insertCall.duration_minutes !== undefined) {
            expect(insertCall.duration_minutes).toBeGreaterThan(0);
            expect(insertCall.duration_minutes).toBeLessThanOrEqual(1440);
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 } // Run 100 iterations with 30s timeout
    );
  }, 60000); // 60 second test timeout

  /**
   * Property: Event update preserves required fields
   * 
   * For any event update, the patient_id and caregiver_id should remain unchanged,
   * and all required fields should still be present.
   */
  it('Property: should preserve required fields when updating events', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        calendarEventArbitrary(), // original event
        calendarEventArbitrary(), // updated event data
        async (patientId, originalEvent, updatedData) => {
          // Mock Supabase responses
          const mockUpdate = vi.fn().mockReturnValue({
            eq: vi.fn().mockResolvedValue({ error: null }),
          });
          const mockFrom = vi.fn().mockReturnValue({
            update: mockUpdate,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                order: vi.fn().mockResolvedValue({ data: [originalEvent], error: null }),
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
          const { result, unmount } = renderHook(() => useCalendarEvents(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Create updated event with new data but same IDs
          const updatedEvent: CalendarEvent = {
            ...originalEvent,
            title: updatedData.title,
            description: updatedData.description,
            event_date: updatedData.event_date,
            event_time: updatedData.event_time,
            duration_minutes: updatedData.duration_minutes,
            event_type: updatedData.event_type,
            location: updatedData.location,
          };

          // Call updateEvent
          await result.current.updateEvent(updatedEvent);

          // Verify Supabase update was called
          expect(mockUpdate).toHaveBeenCalled();

          // Verify the update call contains all required fields
          const updateCall = mockUpdate.mock.calls[0][0];
          expect(updateCall).toHaveProperty('title');
          expect(updateCall.title.trim().length).toBeGreaterThan(0);
          expect(updateCall).toHaveProperty('event_date');
          expect(updateCall).toHaveProperty('event_type');
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
   * Property: Event deletion removes the correct event
   * 
   * For any event deletion, only the specified event should be deleted,
   * identified by its unique ID.
   */
  it('Property: should delete events by ID', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        fc.uuid(), // event_id to delete
        async (patientId, eventId) => {
          // Mock Supabase responses
          const mockEq = vi.fn().mockResolvedValue({ error: null });
          const mockDelete = vi.fn().mockReturnValue({
            eq: mockEq,
          });
          const mockFrom = vi.fn().mockReturnValue({
            delete: mockDelete,
            select: vi.fn().mockReturnValue({
              eq: vi.fn().mockReturnValue({
                order: vi.fn().mockResolvedValue({ data: [], error: null }),
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
          const { result, unmount } = renderHook(() => useCalendarEvents(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Call deleteEvent
          await result.current.deleteEvent(eventId);

          // Verify Supabase delete was called with correct ID
          expect(mockDelete).toHaveBeenCalled();
          expect(mockEq).toHaveBeenCalledWith('id', eventId);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Hook fetches events for correct patient
   * 
   * For any patient ID, the hook should only fetch events associated
   * with that specific patient.
   */
  it('Property: should fetch events only for specified patient', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uuid(), // patient_id
        async (patientId) => {
          // Mock Supabase responses
          const mockEq = vi.fn().mockReturnValue({
            order: vi.fn().mockResolvedValue({ data: [], error: null }),
          });
          const mockSelect = vi.fn().mockReturnValue({
            eq: mockEq,
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
          const { result, unmount } = renderHook(() => useCalendarEvents(patientId));

          // Wait for initial fetch to complete
          await waitFor(() => {
            expect(result.current.loading).toBe(false);
          });

          // Verify Supabase query was called with correct patient_id filter
          expect(mockFrom).toHaveBeenCalledWith('calendar_events');
          expect(mockSelect).toHaveBeenCalledWith('*');
          expect(mockEq).toHaveBeenCalledWith('patient_id', patientId);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);
});
