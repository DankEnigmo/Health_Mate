"""
Quick test to verify the optimized gaze tracker works.
Run this before integrating with frontend.
"""

import cv2
import time
from gaze_tracker_optimized import GazeTrackerOptimized

print("=" * 60)
print("Quick Gaze Tracker Test")
print("=" * 60)

# Initialize
print("\n1. Initializing tracker...")
tracker = GazeTrackerOptimized()
print("   ✓ Tracker initialized")

# Open camera
print("\n2. Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("   ✗ ERROR: Cannot open camera")
    print("   Make sure your webcam is connected and not in use")
    exit(1)
print("   ✓ Camera opened")

# Test for 5 seconds
print("\n3. Testing tracking for 5 seconds...")
print("   Look at the camera!")

start_time = time.time()
frame_count = 0
success_count = 0

while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    result = tracker.get_gaze(frame)
    
    if result:
        success_count += 1
        # Print first result as sample
        if success_count == 1:
            print(f"\n   Sample result:")
            print(f"   - Position: ({result.x:.3f}, {result.y:.3f})")
            print(f"   - Confidence: {result.confidence:.2f}")
            print(f"   - Quality: {result.quality.value}")

cap.release()

# Results
print(f"\n4. Results:")
print(f"   - Frames processed: {frame_count}")
print(f"   - Successful tracks: {success_count}")
print(f"   - Success rate: {(success_count/frame_count)*100:.1f}%")

stats = tracker.get_statistics()
print(f"\n5. Tracker statistics:")
print(f"   - Detection rate: {stats['detection_rate']:.1%}")
print(f"   - Frames lost: {stats['frames_lost']}")

print("\n" + "=" * 60)
if success_count > 0:
    print("✓ SUCCESS - Gaze tracker is working correctly!")
    print("\nNext steps:")
    print("1. Start the API server: python -m Fall_Detection.api_server")
    print("2. Integrate with frontend using the provided hooks")
else:
    print("✗ WARNING - No successful detections")
    print("\nTroubleshooting:")
    print("- Check lighting (avoid backlighting)")
    print("- Position face clearly in camera view")
    print("- Make sure camera is not blocked")
print("=" * 60)
