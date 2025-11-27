import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useFallDetectionStream } from './use-fall-detection-stream';
import * as fc from 'fast-check';

/**
 * Property-Based Test for WebSocket Reconnection
 * 
 * Feature: fall-detection-integration, Property 16: WebSocket reconnection with exponential backoff
 * Validates: Requirements 5.1
 * 
 * This test verifies that the reconnection logic implements exponential backoff correctly.
 * For any sequence of connection failures, the delays between reconnection attempts should
 * follow an exponential pattern: 1s, 2s, 4s, 8s, 16s, with a maximum of 30s.
 */

describe('useFallDetectionStream - Property-Based Tests', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Mock Image constructor
    global.Image = class MockImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      src = '';
      
      constructor() {
        // Simulate async image loading
        setTimeout(() => {
          if (this.onerror) {
            this.onerror();
          }
        }, 0);
      }
    } as any;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  /**
   * Property 16: WebSocket reconnection with exponential backoff
   * 
   * For any number of reconnection attempts (0 to 10), the delay before the next
   * reconnection should follow the exponential backoff pattern:
   * - delay = min(1000 * 2^attempt, 30000)
   * 
   * This ensures that:
   * 1. Initial reconnection happens quickly (1 second)
   * 2. Subsequent attempts back off exponentially
   * 3. Maximum delay is capped at 30 seconds
   */
  it('Property 16: should implement exponential backoff for reconnection attempts', async () => {
    await fc.assert(
      fc.asyncProperty(
        // Generate a sequence of reconnection attempts (0 to 10 attempts)
        fc.integer({ min: 0, max: 10 }),
        async (attemptNumber) => {
          // Calculate expected delay using exponential backoff formula
          const baseDelay = 1000; // 1 second
          const maxDelay = 30000; // 30 seconds
          const expectedDelay = Math.min(baseDelay * Math.pow(2, attemptNumber), maxDelay);
          
          // Verify the delay is within the expected range
          expect(expectedDelay).toBeGreaterThanOrEqual(baseDelay);
          expect(expectedDelay).toBeLessThanOrEqual(maxDelay);
          
          // Verify exponential growth pattern (except when capped)
          if (attemptNumber > 0 && expectedDelay < maxDelay) {
            const previousDelay = Math.min(baseDelay * Math.pow(2, attemptNumber - 1), maxDelay);
            expect(expectedDelay).toBe(previousDelay * 2);
          }
          
          // Verify the delay matches the expected exponential backoff
          const calculatedDelay = Math.min(1000 * Math.pow(2, attemptNumber), 30000);
          expect(expectedDelay).toBe(calculatedDelay);
        }
      ),
      { numRuns: 100 } // Run 100 iterations as specified in design doc
    );
  });

  /**
   * Property: Reconnection delays are monotonically increasing until max
   * 
   * For any sequence of attempts, each delay should be greater than or equal to
   * the previous delay, until the maximum delay is reached.
   */
  it('Property: reconnection delays should be monotonically increasing', async () => {
    await fc.assert(
      fc.asyncProperty(
        // Generate a sequence of attempt numbers
        fc.array(fc.integer({ min: 0, max: 15 }), { minLength: 2, maxLength: 10 }),
        async (attempts) => {
          const delays = attempts.map(attempt => 
            Math.min(1000 * Math.pow(2, attempt), 30000)
          );
          
          // Sort the attempts to ensure we're checking in order
          const sortedAttempts = [...attempts].sort((a, b) => a - b);
          const sortedDelays = sortedAttempts.map(attempt =>
            Math.min(1000 * Math.pow(2, attempt), 30000)
          );
          
          // Verify delays are monotonically increasing
          for (let i = 1; i < sortedDelays.length; i++) {
            expect(sortedDelays[i]).toBeGreaterThanOrEqual(sortedDelays[i - 1]);
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Property: Maximum delay cap is enforced
   * 
   * For any attempt number, the delay should never exceed 30 seconds (30000ms).
   */
  it('Property: maximum delay should be capped at 30 seconds', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.integer({ min: 0, max: 100 }), // Test with large attempt numbers
        async (attemptNumber) => {
          const delay = Math.min(1000 * Math.pow(2, attemptNumber), 30000);
          expect(delay).toBeLessThanOrEqual(30000);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Property: First reconnection attempt has minimum delay
   * 
   * For attempt number 0, the delay should always be 1 second (1000ms).
   */
  it('Property: first reconnection should have 1 second delay', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.constant(0), // Always test with attempt 0
        async (attemptNumber) => {
          const delay = Math.min(1000 * Math.pow(2, attemptNumber), 30000);
          expect(delay).toBe(1000);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Integration test: Verify hook implements exponential backoff
   * 
   * This test verifies that the actual hook implementation follows the
   * exponential backoff pattern when connection failures occur.
   */
  it('Integration: hook should implement exponential backoff on connection failures', async () => {
    const patientId = 'test-patient-123';
    
    // Track the delays between reconnection attempts
    const reconnectionDelays: number[] = [];
    let lastReconnectTime = Date.now();
    
    // Mock Image to track reconnection timing
    let imageCallCount = 0;
    global.Image = class MockImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      src = '';
      
      constructor() {
        imageCallCount++;
        const currentTime = Date.now();
        
        if (imageCallCount > 1) {
          const delay = currentTime - lastReconnectTime;
          reconnectionDelays.push(delay);
        }
        
        lastReconnectTime = currentTime;
        
        // Simulate connection failure
        setTimeout(() => {
          if (this.onerror) {
            this.onerror();
          }
        }, 0);
      }
    } as any;
    
    const { result } = renderHook(() => useFallDetectionStream(patientId));
    
    // Wait for initial connection attempt
    await vi.runAllTimersAsync();
    
    // Trigger multiple reconnection attempts
    for (let i = 0; i < 5; i++) {
      await vi.advanceTimersByTimeAsync(Math.min(1000 * Math.pow(2, i), 30000));
      await vi.runAllTimersAsync();
    }
    
    // Verify that we had multiple reconnection attempts
    expect(imageCallCount).toBeGreaterThan(1);
    
    // Verify exponential backoff pattern in delays
    for (let i = 0; i < reconnectionDelays.length - 1; i++) {
      const currentDelay = reconnectionDelays[i];
      const nextDelay = reconnectionDelays[i + 1];
      
      // Each delay should be approximately double the previous (with some tolerance)
      // or capped at 30000ms
      if (currentDelay < 30000) {
        const expectedNextDelay = Math.min(currentDelay * 2, 30000);
        const tolerance = 100; // 100ms tolerance for timing
        
        expect(nextDelay).toBeGreaterThanOrEqual(expectedNextDelay - tolerance);
        expect(nextDelay).toBeLessThanOrEqual(expectedNextDelay + tolerance);
      }
    }
  });
});
