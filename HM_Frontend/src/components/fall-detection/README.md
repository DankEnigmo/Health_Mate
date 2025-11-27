# Fall Detection Components

This directory contains the frontend components for the fall detection video monitoring system.

## Components

### FallDetectionPlayer

A React component that displays a live MJPEG video stream from the fall detection backend with real-time performance metrics and error handling.

**Location:** `FallDetectionPlayer.tsx`

**Features:**
- Real-time MJPEG video streaming
- Performance metrics overlay (FPS, latency, fall count)
- Automatic reconnection with exponential backoff
- Loading and error states
- Manual reconnection control
- Toggleable metrics display
- Connection status indicator

**Props:**
```typescript
interface FallDetectionPlayerProps {
  patientId: string;              // Patient ID to monitor
  onFallDetected?: (fallData: FallEvent) => void;  // Optional callback
  showMetrics?: boolean;          // Show/hide metrics (default: true)
}
```

**Usage Example:**
```tsx
import { FallDetectionPlayer } from '@/components/fall-detection/FallDetectionPlayer';

function CaregiverFallDetection() {
  const patientId = 'patient-123';
  
  const handleFallDetected = (fallData) => {
    console.log('Fall detected:', fallData);
    // Handle fall alert
  };
  
  return (
    <FallDetectionPlayer
      patientId={patientId}
      onFallDetected={handleFallDetected}
      showMetrics={true}
    />
  );
}
```

## Hooks

### useFallDetectionStream

A custom React hook that manages the video stream connection with automatic reconnection logic.

**Location:** `../../hooks/use-fall-detection-stream.ts`

**Features:**
- Generates stream URL for patient
- Monitors connection status
- Implements exponential backoff reconnection (1s, 2s, 4s, 8s, 16s, max 30s)
- Provides manual reconnection function

**Return Value:**
```typescript
interface UseFallDetectionStreamReturn {
  streamUrl: string;      // Full URL to video stream
  isConnected: boolean;   // Connection status
  error: string | null;   // Error message if any
  reconnect: () => void;  // Manual reconnection function
}
```

**Usage Example:**
```tsx
import { useFallDetectionStream } from '@/hooks/use-fall-detection-stream';

function MyComponent() {
  const { streamUrl, isConnected, error, reconnect } = useFallDetectionStream('patient-123');
  
  return (
    <div>
      {isConnected ? (
        <img src={streamUrl} alt="Video stream" />
      ) : (
        <button onClick={reconnect}>Reconnect</button>
      )}
    </div>
  );
}
```

## Testing

### Property-Based Tests

The reconnection logic is validated using property-based testing with fast-check.

**Test File:** `../../hooks/use-fall-detection-stream.test.ts`

**Properties Tested:**
1. **Property 16**: Exponential backoff pattern (1s, 2s, 4s, 8s, 16s, max 30s)
2. Monotonically increasing delays until max
3. Maximum delay cap at 30 seconds
4. First reconnection has 1 second delay

**Running Tests:**
```bash
cd HealthMate
pnpm test:run use-fall-detection-stream.test.ts
```

## Configuration

The components require the following environment variables:

```env
VITE_BACKEND_API_URL=http://localhost:8000
```

This should be set in `HealthMate/.env` file.

## Requirements Validated

This implementation validates the following requirements from the design document:

- **Requirement 1.1**: Display live video stream with fall detection
- **Requirement 1.2**: Display pose estimation overlays
- **Requirement 1.3**: Display red bounding box on falls
- **Requirement 1.4**: Display loading indicator
- **Requirement 1.5**: Display error message with reconnection options
- **Requirement 5.1**: Implement exponential backoff reconnection
- **Requirement 5.2**: Display connection status
- **Requirement 6.1**: Display FPS metric
- **Requirement 6.2**: Display latency metric

## Design Patterns

### Exponential Backoff

The reconnection logic implements exponential backoff to avoid overwhelming the backend:

```
Attempt 0: 1 second delay
Attempt 1: 2 seconds delay
Attempt 2: 4 seconds delay
Attempt 3: 8 seconds delay
Attempt 4: 16 seconds delay
Attempt 5+: 30 seconds delay (capped)
```

### Error Handling

The component handles multiple error scenarios:
- Backend unavailable
- Network interruption
- Video stream failure
- Image loading errors

Each error state provides appropriate user feedback and recovery options.

## Future Enhancements

Potential improvements for future iterations:
- WebSocket integration for real-time fall alerts
- Video recording on fall detection
- Multi-camera support
- Fullscreen mode
- Picture-in-picture mode
- Performance history graphs
