# new_fall_detector.py

# We must import the 'core' module from your project, which contains
# the actual CentroidTracker and FallDetector classes.
from . import core 
import numpy as np
from collections import OrderedDict

class FallDetectionOrchestrator:
    """
    This class handles the complete logic for tracking person centroids,
    detecting falls, and counting new fall events.
    
    It is designed to be called once per frame.
    """
    def __init__(self):
        """
        Initializes the stateful components for fall detection.
        """
        self.fall_count = 0
        self.frame_count = 0
        
        # These are the core classes from your project's 'core' module
        self.ct = core.CentroidTracker()
        self.falls = core.FallDetector()
        
        # State to track who was fallen on the *previous* frame
        self.prev_fallen = OrderedDict()

    def _calculate_centroid(self, kps):
        """
        Calculates the centroid (mid-shoulder) for a single person's keypoints.
        This logic is copied directly from painter.py (lines 191-215).
        Returns (mid_x, mid_y) or None.
        """
        x = kps[:, 0]
        y = kps[:, 1]
        
        mid_x = 0
        if x[5] != 0 and x[6] == 0:
            mid_x = x[5]
        elif x[5] == 0 and x[6] != 0:
            mid_x = x[6]
        elif x[5] != 0 and x[6] != 0:
            mid_x = (x[5] + x[6]) / 2
        else:
            return None  # No shoulders detected

        mid_y = 0
        if y[5] != 0 and y[6] == 0:
            mid_y = y[5]
        elif y[5] == 0 and y[6] != 0:
            mid_y = y[6]
        elif y[5] != 0 and y[6] != 0:
            mid_y = (y[5] + y[6]) / 2
        else:
            return None # No shoulders detected

        if mid_x != 0 and mid_y != 0:
            return (mid_x, mid_y)
        
        return None

    def process_detections(self, annotations, fps):
        """
        This is the main function to call on every frame.
        It processes annotations, updates trackers, and counts falls.
        
        :param annotations: The list of 'Annotation' objects from the model.
        :param fps: The current frames-per-second of the video.
        :return: A tuple (current_fall_count, dict_of_fallen_people)
        """
        
        # 1. Calculate centroids for everyone in the current frame
        current_centroids = []
        for ann in annotations:
            # Get the bounding box
            x_, y_, w_, h_ = ann.bbox()
            
            # Calculate the centroid
            centroid = self._calculate_centroid(ann.data)
            
            # Add to list if valid
            if centroid is not None:
                mid_x, mid_y = centroid
                # The tracker needs the centroid (x,y) AND the bbox (x,y,w,h)
                current_centroids.append((mid_x, mid_y, x_, y_, w_, h_))
        
        # 2. Track people frame-by-frame
        # 'persons' will be an OrderedDict like {ID: (x, y, x_, y_, w_, h_)}
        persons = self.ct.update(current_centroids, fps)
        
        # 3. Analyze movement with the Fall Detector
        # 'fallen' will be an OrderedDict like {ID: (x_, y_, w_, h_)}
        fallen = self.falls.update(persons, self.frame_count, fps)
        
        # 4. Count only *new* falls
        for ID in fallen.keys():
            if ID not in self.prev_fallen:
                # This is a new fall
                self.fall_count += 1
                print(f"!!! NEW FALL DETECTED: ID {ID}, Total Count: {self.fall_count}")
        
        # 5. Update state for the next frame
        self.prev_fallen = fallen
        self.frame_count += 1
        
        # Return the total count and the dict of fallen people (for drawing boxes)
        return self.fall_count, fallen