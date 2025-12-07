# Human-Fall-Detection

## The Project

In this project, we developed a video analysis system that can detect events of human falling.

The system includes a **FastAPI-based API server** that provides:
- Real-time MJPEG video streaming with fall detection overlays
- WebSocket endpoint for instant fall alerts
- REST API for statistics and configuration
- Supabase integration for persisting fall events

The system receives video as input, scans each frame of the video, and then creates 17 key-points for each individual, each of which corresponds to a position of that person's body parts in that frame. This is done using the [YOLOv7-POSE](https://github.com/WongKinYiu/yolov7/tree/pose "YOLOv7-POSE") model.

For example

![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/Mydata/keypoints-example.png)

You can learn more about YOLOv7-POSE by reading this [document](https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf "document").

We initially created an LSTM-based neural network that learned from a database of photos of humans that were categorized as "falling" and "not falling."
We obtained roughly 500 images from the Internet for this purpose.
After we used the data to train the model, we discovered that the network did not succeed in learning, and we got poor results using the test images.

After looking at the pictures, we found that some of them labeld as "falling" but contained multiple people who some of them were not actually falling (but only one of them) . Therefore, in order to improve our data, we created a tool that go through all of the data and extracted every person from each photo into a different photo. After that, we once again cleaned and tagged all of the new data.

Unfortunately, the model did not succeed in learning on the new data either.
The amount of the data, which contains approximately 500 photos, is likely the key factor in the network learning failure, but this is the only data that is available on the Internet.

The original image database can be found [here](https://github.com/bakshtb/Human-Fall-Detection/tree/master/fall_dataset/old "here"), while the new image database we made can be found [here](https://github.com/bakshtb/Human-Fall-Detection/tree/master/fall_dataset/images "here").


We considered creating a straightforward if-else-based model that would detect a fall in accordance with logical conditions regarding the relative location of the body parts after the LSTM-based network failed to learn from our image library.

## Results

Using our own data, we tested our model, and the results were, in our opinion, pretty good, as follows:

| Accuracy  | Precision | Recall | F1 score |
| ------------- | ------------- | ------------- | ------------- |
|  81.18%  | 83.27% | 83.58%  | 83.42%  |

For the video analysis we used Nvidia's Tesla K80 GPU, the system analyzes the videos at a reasonable speed of 15Fps.

## How To Use

### Quick Start (API Server)

1. Clone this repository into your drive
2. Download the [YOLOv7-POSE](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt "YOLOv7-POSE") model into the `Fall_Detection` directory
3. Install all the requirements with `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure your settings
5. Run the API server: `python api_server.py`

### Standalone Scripts

- `video.py`: Run fall detection on video files
- `realtime.py`: Run real-time fall detection via webcam (standalone, no API)

## API Server

### Starting the Server

```bash
# From the Fall_Detection directory
python api_server.py
```

The server will start at `http://localhost:8000` by default.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns server status |
| `/api/stats` | GET | Get current detection statistics (FPS, fall count, etc.) |
| `/api/video/stream` | GET | MJPEG video stream with fall detection overlays |
| `/api/ws/alerts` | WebSocket | Real-time fall alert notifications |
| `/api/video/source` | POST | Configure video source (camera index or URL) |
| `/api/detector/reset` | POST | Reset detector state and fall counters |
| `/api/config` | GET | Get current detection configuration |

### Video Stream

Access the live video stream with fall detection overlays:
```
http://localhost:8000/api/video/stream?patient_id=<patient_uuid>
```

### WebSocket Alerts

Connect to receive real-time fall alerts:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/alerts?patient_id=<patient_uuid>');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'fall_detected') {
    console.log('Fall detected!', message.data);
  }
};
```

**WebSocket Message Types:**

| Type | Direction | Description |
|------|-----------|-------------|
| `connected` | Server → Client | Connection confirmation |
| `fall_detected` | Server → Client | Fall alert with details |
| `heartbeat` | Server → Client | Keep-alive with stats |
| `ping` | Client → Server | Request pong response |
| `pong` | Server → Client | Response to ping |
| `get_stats` | Client → Server | Request current stats |
| `stats` | Server → Client | Statistics response |
| `subscribe` | Client → Server | Subscribe to different patient |

### Statistics Response

```json
{
  "fps": 25.5,
  "latency": 45.2,
  "fall_count": 3,
  "falls_by_person": {"1": 2, "2": 1},
  "frames_processed": 1500,
  "is_processing": true,
  "active_trackers": 2,
  "video_source": "0",
  "ws_connections": 1,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Fall Alert Message

```json
{
  "type": "fall_detected",
  "data": {
    "patient_id": "uuid-here",
    "person_tracking_id": 1,
    "fall_count": 2,
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
      "tag": "SpeedDrop DownFlat",
      "debug": "v=85.2/60.0, dy=45.3/20.0, ar=0.45/0.35",
      "bounding_box": [100, 150, 300, 450],
      "fps": 25.5
    }
  }
}
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Supabase (for persisting fall events)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key

# Video source (camera index or path/URL)
VIDEO_SOURCE=0

# Server settings
API_HOST=0.0.0.0
API_PORT=8000

# Fall detection thresholds
FPS=30
WINDOW_SIZE=10
V_THRESH=60
DY_THRESH=20
ASPECT_RATIO_THRESH=0.35

# Streaming settings
JPEG_QUALITY=80
MAX_STREAM_FPS=30
FALL_COOLDOWN=5.0
```


## Examples
Examples of videos collected from the Internet and analyzed by our system are shown below.

These videos demonstrate how the model successfully and accurately recognizes human falls.

![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_1_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_2_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_4_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_5_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_6_keypoint.gif)

## Work in Progress
- Currently implementing real-time fall detection logic
- **Short Term**: Optimize the if-else-based model using a time-sliding window
- **Long Term**: Integrate a time-series model (e.g., LSTM) for more accurate detection

## Fall Detect Logic
The fall detection logic uses the following parameters:

| Parameter              | Description |
|------------------------|-------------|
| `FPS`                 | Frames per second. Used to calculate time and velocity. |
| `WINDOW_SIZE`         | Number of frames in each analysis window. Defines how many frames are used to evaluate pose changes. |
| `V_THRESH`            | Threshold for center-of-mass velocity. Movements faster than this are considered potentially abnormal. |
| `DY_THRESH`           | Threshold for vertical (Y-axis) displacement of the center of mass. |
| `ASPECT_RATIO_THRESH` | Threshold for change in aspect ratio (width/height) of the body. Indicates whether the body has become horizontal. |

These values can be configured via a `.env` file.  
If not provided, default values defined in the code will be used.

> **Note:** This logic is still under development and subject to change as part of our ongoing implementation and validation process.


## Frontend Integration

The API server is designed to integrate with the HealthMate frontend. The frontend uses:

- **MJPEG Stream**: Displayed via `<img>` tag pointing to `/api/video/stream`
- **WebSocket Alerts**: Real-time notifications via `/api/ws/alerts`
- **Stats Polling**: Periodic fetch from `/api/stats` for metrics display
- **Supabase Realtime**: Backup channel for fall events via database subscriptions

See `HM_Frontend/src/components/fall-detection/` for implementation examples.

## Possible Future Improvements

In order to alert human falls and save lives, the real-time system may be deployed and implemented in nursing homes, hospitals, and senior living facilities.

Additional planned improvements:
- GPU acceleration for higher FPS
- Multiple camera support
- Cloud deployment with load balancing
- Mobile app push notifications
- Fall severity classification

