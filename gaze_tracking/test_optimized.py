"""
Test script for the optimized gaze tracker.
Run this to verify the optimization works correctly.
"""

import cv2
import time
import numpy as np
from gaze_tracker_optimized import GazeTrackerOptimized, TrackingQuality

def test_performance():
    """Test the performance of the optimized tracker."""
    print("=" * 60)
    print("Testing Optimized Gaze Tracker Performance")
    print("=" * 60)
    
    # Initialize tracker
    print("\n1. Initializing tracker...")
    tracker = GazeTrackerOptimized(
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        smoothing_alpha=0.40,
        enable_preprocessing=True
    )
    print("✓ Tracker initialized")
    
    # Open camera
    print("\n2. Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return
    print("✓ Camera opened")
    
    # Warm up
    print("\n3. Warming up (processing 30 frames)...")
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            _ = tracker.get_gaze(frame)
    print("✓ Warm up complete")
    
    # Performance test
    print("\n4. Running performance test (300 frames)...")
    frame_times = []
    successful_tracks = 0
    total_frames = 300
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        
        start = time.time()
        result = tracker.get_gaze(frame)
        elapsed = time.time() - start
        
        frame_times.append(elapsed)
        
        if result and result.quality != TrackingQuality.LOST:
            successful_tracks += 1
        
        # Show progress
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{total_frames} frames...")
    
    cap.release()
    
    # Calculate statistics
    avg_time = np.mean(frame_times) * 1000  # ms
    std_time = np.std(frame_times) * 1000
    min_time = np.min(frame_times) * 1000
    max_time = np.max(frame_times) * 1000
    avg_fps = 1.0 / np.mean(frame_times)
    success_rate = (successful_tracks / total_frames) * 100
    
    # Get tracker stats
    stats = tracker.get_statistics()
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"\nProcessing Time:")
    print(f"  Average: {avg_time:.2f} ms ± {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"\nThroughput:")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"\nTracker Statistics:")
    print(f"  Total Frames: {stats['total_frames']}")
    print(f"  Successful Detections: {stats['successful_detections']}")
    print(f"  Detection Rate: {stats['detection_rate']:.1%}")
    print(f"  Frames Lost: {stats['frames_lost']}")
    print(f"  Calibration Points: {stats['calibration_points']}")
    print(f"  Is Calibrated: {stats['is_calibrated']}")
    
    # Performance rating
    print("\nPerformance Rating:")
    if avg_time < 25 and success_rate > 90:
        print("  ★★★★★ EXCELLENT - Ready for production")
    elif avg_time < 40 and success_rate > 80:
        print("  ★★★★☆ GOOD - Acceptable performance")
    elif avg_time < 60 and success_rate > 70:
        print("  ★★★☆☆ FAIR - May need optimization")
    else:
        print("  ★★☆☆☆ POOR - Check system resources")
    
    print("\n" + "=" * 60)


def test_calibration():
    """Test the calibration system."""
    print("\n" + "=" * 60)
    print("Testing Calibration System")
    print("=" * 60)
    
    tracker = GazeTrackerOptimized()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return
    
    # Calibration targets (3x3 grid)
    targets = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]
    
    print("\nCalibration Process:")
    print("  Press SPACE to collect sample")
    print("  Press N for next point")
    print("  Press Q to quit")
    
    current_target = 0
    samples_collected = 0
    samples_per_point = 10
    
    while current_target < len(targets):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Draw target
        tx, ty = targets[current_target]
        cv2.circle(frame, (int(tx * w), int(ty * h)), 20, (0, 255, 0), -1)
        
        # Draw instructions
        text = f"Target {current_target + 1}/9 - Samples: {samples_collected}/{samples_per_point}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "SPACE: Collect | N: Next | Q: Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Collect sample
            features, _ = tracker.extract_features(frame)
            if features is not None:
                tracker.add_calibration_point(features, targets[current_target])
                samples_collected += 1
                print(f"✓ Collected sample {samples_collected} for target {current_target + 1}")
            else:
                print("✗ No face detected")
        elif key == ord('n') or samples_collected >= samples_per_point:
            current_target += 1
            samples_collected = 0
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Test calibration
    stats = tracker.get_statistics()
    print(f"\n✓ Calibration complete!")
    print(f"  Total points: {stats['calibration_points']}")
    print(f"  Is calibrated: {stats['is_calibrated']}")
    
    if stats['is_calibrated']:
        print("\n★ Calibration successful - Affine transformation computed")
    else:
        print("\n✗ Calibration failed - Need at least 3 points")


def test_realtime_tracking():
    """Test real-time tracking with visualization."""
    print("\n" + "=" * 60)
    print("Testing Real-time Tracking")
    print("=" * 60)
    print("\nPress Q to quit")
    
    tracker = GazeTrackerOptimized(smoothing_alpha=0.40)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return
    
    fps_history = []
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Track gaze
        result = tracker.get_gaze(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        last_time = current_time
        
        # Draw results
        if result:
            # Draw gaze point
            gx = int(result.x * w)
            gy = int(result.y * h)
            
            # Color based on quality
            if result.quality == TrackingQuality.EXCELLENT:
                color = (0, 255, 0)
            elif result.quality == TrackingQuality.GOOD:
                color = (0, 255, 255)
            elif result.quality == TrackingQuality.POOR:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.circle(frame, (gx, gy), 15, color, -1)
            cv2.circle(frame, (gx, gy), 18, (255, 255, 255), 2)
            
            # Draw iris centers
            if result.iris_centers:
                for eye, (ix, iy) in result.iris_centers.items():
                    if eye in ['left', 'right']:
                        cv2.circle(frame, (int(ix), int(iy)), 3, (255, 0, 0), -1)
            
            # Info text
            info = [
                f"FPS: {avg_fps:.1f}",
                f"Gaze: ({result.x:.2f}, {result.y:.2f})",
                f"Confidence: {result.confidence:.2f}",
                f"Quality: {result.quality.value}"
            ]
            
            for i, line in enumerate(info):
                cv2.putText(frame, line, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Real-time Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    stats = tracker.get_statistics()
    print(f"\n✓ Tracking session complete")
    print(f"  Average FPS: {np.mean(fps_history):.1f}")
    print(f"  Detection Rate: {stats['detection_rate']:.1%}")


if __name__ == "__main__":
    print("\nOptimized Gaze Tracker Test Suite")
    print("=" * 60)
    
    tests = [
        ("Performance Test", test_performance),
        ("Calibration Test", test_calibration),
        ("Real-time Tracking", test_realtime_tracking)
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    print("  0. Run all tests")
    
    try:
        choice = input("\nSelect test (0-3): ").strip()
        choice = int(choice) if choice else 0
        
        if choice == 0:
            for name, test_func in tests:
                print(f"\n{'=' * 60}")
                print(f"Running: {name}")
                print(f"{'=' * 60}")
                test_func()
        elif 1 <= choice <= len(tests):
            tests[choice - 1][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test suite complete")
    print("=" * 60)
