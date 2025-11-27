import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { UpcomingEvents } from './UpcomingEvents';
import { CalendarEvent } from '@/types/calendar';
import * as fc from 'fast-check';
import { BrowserRouter } from 'react-router-dom';

/**
 * Property-Based Tests for UpcomingEvents Component
 * 
 * These tests verify the correctness properties of upcoming events display:
 * - Property 39: Upcoming events ordering and limiting
 */

// Wrapper component for Router context
const RouterWrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>{children}</BrowserRouter>
);

describe('UpcomingEvents - Property-Based Tests', () => {
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
   * Property 39: Upcoming events ordering and limiting
   * 
   * Feature: fall-detection-integration, Property 39: Upcoming events ordering and limiting
   * Validates: Requirements 12.2
   * 
   * For any set of calendar events retrieved, the next N upcoming events should be
   * displayed in chronological order (earliest first).
   * 
   * This test verifies that:
   * 1. Only upcoming events (today or future) are displayed
   * 2. Events are displayed in chronological order (earliest first)
   * 3. The number of displayed events respects the limit parameter
   * 4. Events on the same date are ordered by time (if time is present)
   * 5. Past events are filtered out
   */
  it('Property 39: should display upcoming events in chronological order with limit', async () => {
    await fc.assert(
      fc.asyncProperty(
        // Generate an array of events with various dates (past, present, future)
        fc.array(calendarEventArbitrary(), { minLength: 0, maxLength: 20 }),
        fc.integer({ min: 1, max: 10 }), // limit
        async (events, limit) => {
          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={events} limit={limit} />
            </RouterWrapper>
          );

          // Get current date for comparison
          const now = new Date();
          now.setHours(0, 0, 0, 0); // Reset to start of day for comparison

          // Filter to only upcoming events (today or future)
          const upcomingEvents = events.filter((event) => {
            const eventDate = new Date(event.event_date);
            eventDate.setHours(0, 0, 0, 0);
            return eventDate >= now;
          });

          // Sort upcoming events chronologically
          const sortedUpcoming = [...upcomingEvents].sort((a, b) => {
            const dateA = new Date(a.event_date);
            const dateB = new Date(b.event_date);

            // Sort by date first
            if (dateA.getTime() !== dateB.getTime()) {
              return dateA.getTime() - dateB.getTime();
            }

            // If same date, sort by time
            if (a.event_time && b.event_time) {
              return a.event_time.localeCompare(b.event_time);
            }

            // Events with time come before events without time
            if (a.event_time && !b.event_time) return -1;
            if (!a.event_time && b.event_time) return 1;

            return 0;
          });

          // Apply limit
          const expectedEvents = sortedUpcoming.slice(0, limit);

          if (expectedEvents.length === 0) {
            // Verify empty state is displayed
            expect(screen.getByText('No upcoming events')).toBeInTheDocument();
          } else {
            // Verify no more than 'limit' events are displayed
            // Count the number of event cards by counting h4 headings
            const allH4Titles = screen.queryAllByRole('heading', { level: 4 });
            const eventCardCount = allH4Titles.length;
            
            expect(eventCardCount).toBeLessThanOrEqual(limit);
            expect(eventCardCount).toBe(Math.min(expectedEvents.length, limit));

            // Verify chronological ordering by checking the order of titles in the DOM
            const displayedTitleTexts = allH4Titles.map(el => el.textContent);

            // The displayed titles should match the expected order
            for (let i = 0; i < Math.min(expectedEvents.length, displayedTitleTexts.length); i++) {
              expect(displayedTitleTexts[i]).toBe(expectedEvents[i].title);
            }

            // Verify dates are in ascending order
            for (let i = 0; i < expectedEvents.length - 1; i++) {
              const date1 = new Date(expectedEvents[i].event_date);
              const date2 = new Date(expectedEvents[i + 1].event_date);
              
              // Date should be less than or equal to next date
              expect(date1.getTime()).toBeLessThanOrEqual(date2.getTime());

              // If same date and both have times, verify time ordering
              if (date1.getTime() === date2.getTime() && 
                  expectedEvents[i].event_time && 
                  expectedEvents[i + 1].event_time) {
                expect(expectedEvents[i].event_time!.localeCompare(expectedEvents[i + 1].event_time!))
                  .toBeLessThanOrEqual(0);
              }
            }
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 } // Run 100 iterations with 30s timeout
    );
  }, 60000); // 60 second test timeout

  /**
   * Property: Past events are never displayed
   * 
   * For any set of events including past events, only events from today
   * onwards should be displayed.
   */
  it('Property: should never display past events', async () => {
    await fc.assert(
      fc.asyncProperty(
        // Generate events with dates in the past
        fc.array(
          fc.record({
            id: fc.uuid(),
            patient_id: fc.uuid(),
            caregiver_id: fc.uuid(),
            title: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
            description: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
            event_date: fc.date({ min: new Date('2020-01-01'), max: new Date('2023-12-31') })
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
            created_at: fc.date({ min: new Date('2020-01-01'), max: new Date('2023-12-31') })
              .map(d => {
                const year = d.getFullYear();
                const month = String(d.getMonth() + 1).padStart(2, '0');
                const day = String(d.getDate()).padStart(2, '0');
                const hours = String(d.getHours()).padStart(2, '0');
                const minutes = String(d.getMinutes()).padStart(2, '0');
                const seconds = String(d.getSeconds()).padStart(2, '0');
                return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.000Z`;
              }),
            updated_at: fc.date({ min: new Date('2020-01-01'), max: new Date('2023-12-31') })
              .map(d => {
                const year = d.getFullYear();
                const month = String(d.getMonth() + 1).padStart(2, '0');
                const day = String(d.getDate()).padStart(2, '0');
                const hours = String(d.getHours()).padStart(2, '0');
                const minutes = String(d.getMinutes()).padStart(2, '0');
                const seconds = String(d.getSeconds()).padStart(2, '0');
                return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.000Z`;
              }),
          }),
          { minLength: 1, maxLength: 10 }
        ),
        async (pastEvents) => {
          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={pastEvents} limit={5} />
            </RouterWrapper>
          );

          // Since all events are in the past, should show empty state
          expect(screen.getByText('No upcoming events')).toBeInTheDocument();

          // Verify no event titles are displayed
          for (const event of pastEvents) {
            expect(screen.queryByText(event.title)).not.toBeInTheDocument();
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Events with same date are ordered by time
   * 
   * For any set of events on the same date with different times,
   * they should be ordered by time (earliest first).
   */
  it('Property: should order same-day events by time', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.date({ min: new Date(), max: new Date('2030-12-31') })
          .map(d => {
            const year = d.getFullYear();
            const month = String(d.getMonth() + 1).padStart(2, '0');
            const day = String(d.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          }),
        fc.array(
          fc.record({
            hour: fc.integer({ min: 0, max: 23 }),
            minute: fc.integer({ min: 0, max: 59 }),
          }),
          { minLength: 2, maxLength: 5 }
        ),
        async (eventDate, times) => {
          // Create unique times by adding index to minutes if needed
          const uniqueTimes = times.map((time, index) => {
            const totalMinutes = time.hour * 60 + time.minute + index;
            const hour = Math.floor(totalMinutes / 60) % 24;
            const minute = totalMinutes % 60;
            return { hour, minute };
          });

          // Create events with same date but different times
          const events: CalendarEvent[] = uniqueTimes.map((time, index) => ({
            id: `event-${index}`,
            patient_id: 'test-patient',
            caregiver_id: 'test-caregiver',
            title: `Event ${index}`,
            event_date: eventDate,
            event_time: `${time.hour.toString().padStart(2, '0')}:${time.minute.toString().padStart(2, '0')}`,
            event_type: 'appointment',
            created_at: '2020-01-01T00:00:00.000Z',
            updated_at: '2020-01-01T00:00:00.000Z',
          }));

          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={events} limit={10} />
            </RouterWrapper>
          );

          // Get all displayed event titles
          const allTitles = screen.queryAllByRole('heading', { level: 4 });
          const displayedTitleTexts = allTitles.map(el => el.textContent);

          // Sort events by time to get expected order
          const sortedEvents = [...events].sort((a, b) => 
            a.event_time!.localeCompare(b.event_time!)
          );

          // Verify displayed order matches expected order
          expect(displayedTitleTexts.length).toBe(sortedEvents.length);
          for (let i = 0; i < Math.min(displayedTitleTexts.length, sortedEvents.length); i++) {
            expect(displayedTitleTexts[i]).toBe(sortedEvents[i].title);
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Limit parameter is respected
   * 
   * For any limit value, the component should display at most that many events,
   * even if more upcoming events exist.
   */
  it('Property: should respect the limit parameter', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.integer({ min: 1, max: 5 }), // limit
        fc.array(
          fc.record({
            id: fc.uuid(),
            patient_id: fc.uuid(),
            caregiver_id: fc.uuid(),
            title: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
            description: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
            event_date: fc.date({ min: new Date(), max: new Date('2030-12-31') })
              .map(d => d.toISOString().split('T')[0]),
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
            created_at: fc.date({ min: new Date('2020-01-01'), max: new Date() }).map(d => {
              try {
                return d.toISOString();
              } catch {
                return new Date('2020-01-01').toISOString();
              }
            }),
            updated_at: fc.date({ min: new Date('2020-01-01'), max: new Date() }).map(d => {
              try {
                return d.toISOString();
              } catch {
                return new Date('2020-01-01').toISOString();
              }
            }),
          }),
          { minLength: 10, maxLength: 20 } // More events than any limit
        ),
        async (limit, futureEvents) => {
          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={futureEvents} limit={limit} />
            </RouterWrapper>
          );

          // Count displayed event cards by counting h4 elements
          const allTitles = screen.queryAllByRole('heading', { level: 4 });
          
          // Should display at most 'limit' events
          expect(allTitles.length).toBeLessThanOrEqual(limit);

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);

  /**
   * Property: Empty events array displays empty state
   * 
   * For an empty events array, the component should display
   * the "No upcoming events" message.
   */
  it('Property: should display empty state for empty events array', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.constant([] as CalendarEvent[]), // Empty array
        async (emptyEvents) => {
          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={emptyEvents} limit={5} />
            </RouterWrapper>
          );

          // Verify empty state is displayed
          expect(screen.getByText('No upcoming events')).toBeInTheDocument();

          // Verify calendar icon is present (part of empty state)
          const calendarIcon = document.querySelector('[class*="lucide-calendar"]');
          expect(calendarIcon).toBeInTheDocument();

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Property: All displayed events have required fields
   * 
   * For any displayed event, it must show the title, date, and event type.
   * Optional fields (time, location, duration) should be displayed when present.
   */
  it('Property: should display all required fields for each event', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(
          fc.record({
            id: fc.uuid(),
            patient_id: fc.uuid(),
            caregiver_id: fc.uuid(),
            title: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
            description: fc.option(fc.string({ minLength: 0, maxLength: 500 }), { nil: undefined }),
            event_date: fc.date({ min: new Date(), max: new Date('2030-12-31') })
              .map(d => d.toISOString().split('T')[0]),
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
            created_at: fc.date({ min: new Date('2020-01-01'), max: new Date() }).map(d => {
              try {
                return d.toISOString();
              } catch {
                return new Date('2020-01-01').toISOString();
              }
            }),
            updated_at: fc.date({ min: new Date('2020-01-01'), max: new Date() }).map(d => {
              try {
                return d.toISOString();
              } catch {
                return new Date('2020-01-01').toISOString();
              }
            }),
          }),
          { minLength: 1, maxLength: 5 }
        ),
        async (futureEvents) => {
          // Render the component
          const { unmount } = render(
            <RouterWrapper>
              <UpcomingEvents events={futureEvents} limit={10} />
            </RouterWrapper>
          );

          // Verify each event has required fields displayed
          for (const event of futureEvents) {
            // Title should be present (use getAllByText since titles might not be unique)
            // Use a flexible matcher that handles whitespace
            const titleElements = screen.queryAllByText(event.title.trim());
            expect(titleElements.length).toBeGreaterThan(0);

            // Event type badge should be present
            const typeElements = screen.getAllByText(event.event_type);
            expect(typeElements.length).toBeGreaterThan(0);

            // Date should be present (in some format - Today, Tomorrow, or formatted date)
            // We can't check exact text due to date formatting, but calendar icon should be present
            const calendarIcons = document.querySelectorAll('[class*="lucide-calendar"]');
            expect(calendarIcons.length).toBeGreaterThan(0);

            // If event has time, verify clock icon is present
            if (event.event_time) {
              const clockIcons = document.querySelectorAll('[class*="lucide-clock"]');
              expect(clockIcons.length).toBeGreaterThan(0);
            }

            // If event has location, verify map pin icon is present
            if (event.location) {
              const mapPinIcons = document.querySelectorAll('[class*="lucide-map-pin"]');
              expect(mapPinIcons.length).toBeGreaterThan(0);
            }
          }

          // Cleanup
          unmount();
        }
      ),
      { numRuns: 100, timeout: 30000 }
    );
  }, 60000);
});
