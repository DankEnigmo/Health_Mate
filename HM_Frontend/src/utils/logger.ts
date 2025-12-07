/**
 * Logger Utility
 * 
 * Provides environment-aware logging that only outputs in development mode.
 * In production, console logs are disabled to improve performance and security.
 * 
 * Usage:
 *   import { logger } from '@/utils/logger';
 *   logger.log('Debug info', data);
 *   logger.error('Error occurred', error);
 */

const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;

// Error tracking service configuration (e.g., Sentry, LogRocket)
// TODO: Add your error tracking service here
const sendToErrorTracking = (level: string, message: string, data?: any) => {
  if (isProd) {
    // Example: Sentry.captureMessage(message, level, { extra: data });
    // For now, we'll just suppress in production
  }
};

export const logger = {
  /**
   * Log general information (dev only)
   */
  log: isDev
    ? console.log.bind(console)
    : () => {},

  /**
   * Log errors (always tracked, console in dev only)
   */
  error: isDev
    ? console.error.bind(console)
    : (message: any, ...args: any[]) => {
        sendToErrorTracking('error', String(message), args);
      },

  /**
   * Log warnings (dev only)
   */
  warn: isDev
    ? console.warn.bind(console)
    : () => {},

  /**
   * Log debug information (dev only)
   */
  debug: isDev
    ? console.debug.bind(console)
    : () => {},

  /**
   * Log informational messages (dev only)
   */
  info: isDev
    ? console.info.bind(console)
    : () => {},

  /**
   * Group logs together (dev only)
   */
  group: isDev
    ? console.group.bind(console)
    : () => {},

  /**
   * End log group (dev only)
   */
  groupEnd: isDev
    ? console.groupEnd.bind(console)
    : () => {},

  /**
   * Time operations (dev only)
   */
  time: isDev
    ? console.time.bind(console)
    : () => {},

  /**
   * End timing (dev only)
   */
  timeEnd: isDev
    ? console.timeEnd.bind(console)
    : () => {},

  /**
   * Log table data (dev only)
   */
  table: isDev
    ? console.table.bind(console)
    : () => {},
};

// Export type for TypeScript
export type Logger = typeof logger;

// Default export
export default logger;
