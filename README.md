# 🏥 HealthMate - AI-Powered Fall Detection & Health Management System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![React](https://img.shields.io/badge/react-18.3+-61dafb.svg)

**A comprehensive healthcare monitoring solution combining real-time fall detection with medication management and caregiver coordination.**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Technologies Used](#-technologies-used)


---

## 🌟 Overview

HealthMate is a full-stack healthcare monitoring application designed for elderly care and patient management. It combines cutting-edge computer vision technology for fall detection with comprehensive health management features, providing peace of mind for both patients and caregivers.

### Key Components

- **🎥 Fall Detection Backend** - Real-time video processing with AI-powered pose estimation
- **💊 Medication Management** - Automated scheduling, reminders, and adherence tracking
- **👨‍⚕️ Caregiver Dashboard** - Centralized monitoring and alert management
- **🔔 Real-time Notifications** - WebSocket-based instant alerts for critical events
- **📊 Health Analytics** - Comprehensive tracking and reporting

---

## ✨ Features

### For Patients

- 🎥 **24/7 Fall Detection Monitoring** - Automatic detection with instant alerts
- 💊 **Medication Reminders** - Never miss a dose with smart scheduling
- 📅 **Calendar Management** - Track appointments and important dates
- 🆘 **Emergency Button** - One-touch emergency alert system
- 📱 **User-Friendly Interface** - Simplified UI optimized for elderly users

### For Caregivers

- 🚨 **Real-time Fall Alerts** - Instant notifications when incidents occur
- 👥 **Multi-Patient Management** - Monitor multiple patients from one dashboard
- 📊 **Medication Adherence Tracking** - View compliance reports and missed doses
- 📈 **Health Analytics Dashboard** - Comprehensive patient health insights
- 💬 **Communication Tools** - Direct messaging with patients

### Technical Features

- 🤖 **AI-Powered Pose Estimation** - Using OpenPifPaf for accurate human pose detection
- 🔄 **Real-time Video Streaming** - MJPEG streaming with fall detection overlays
- 🔐 **Secure Authentication** - Supabase-based authentication with role-based access
- 🌐 **RESTful API** - Well-documented API endpoints
- 📡 **WebSocket Support** - Real-time bidirectional communication
- 🎨 **Modern UI/UX** - Built with shadcn/ui and Tailwind CSS
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HealthMate System                       │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────▼────────┐                       ┌────────▼────────┐
│   Frontend     │                       │    Backend      │
│   (React)      │◄─────────────────────►│   (FastAPI)     │
│                │   REST API/WebSocket  │                 │
│  - Patient UI  │                       │  - Fall Detect  │
│  - Caregiver   │                       │  - Video Stream │
│  - Auth        │                       │  - Medication   │
└────────────────┘                       └─────────────────┘
        │                                         │
        │                                         │
        └────────────────┬────────────────────────┘
                         │
                  ┌──────▼──────┐
                  │   Supabase  │
                  │             │
                  │  - Auth     │
                  │  - Database │
                  │  - Storage  │
                  └─────────────┘
```

### Component Breakdown

#### Backend (`Fall_Detection/`)
- **FastAPI Server** - High-performance async API
- **OpenPifPaf Integration** - Pose estimation model
- **Video Stream Manager** - Handles video capture and processing
- **Fall Detection Algorithm** - Custom fall detection logic
- **Medication Scheduler** - Automated reminder system
- **WebSocket Manager** - Real-time communication

#### Frontend (`HM_Frontend/`)
- **React + TypeScript** - Type-safe component architecture
- **Vite** - Lightning-fast build tool
- **React Router** - Client-side routing with protected routes
- **TanStack Query** - Server state management
- **shadcn/ui** - Beautiful, accessible components
- **Tailwind CSS** - Utility-first styling

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

| Software | Minimum Version | Purpose |
|----------|----------------|---------|
| **Python** | 3.8+ | Backend runtime |
| **Node.js** | 16.x+ | Frontend runtime |
| **pnpm** | 8.x+ | Package manager |
| **pip** | Latest | Python package manager |

### Optional Software

- **CUDA Toolkit** (Optional) - For GPU-accelerated inference
- **Webcam** - For live fall detection
- **Git** - Version control

### External Services

- **Supabase Account** - For authentication and database ([Get started](https://supabase.com))

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DankEnigmo/Health_Mate.git
cd Health_Mate
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r Fall_Detection/requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd HM_Frontend

# Install dependencies
pnpm install

# Return to root
cd ..
```

---

## ⚙️ Configuration

### Backend Configuration

1. Copy the environment template:
```bash
cp Fall_Detection/.env.example Fall_Detection/.env
```

2. Edit `Fall_Detection/.env` with your settings:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# Video Source Configuration
VIDEO_SOURCE=0                    # 0 for webcam, or path to video file
# VIDEO_SOURCE=Fall_Detection/input/video.mp4

# Fall Detection Configuration
FALL_THRESHOLD=0.6                # Fall detection sensitivity (0.0-1.0)
MOVEMENT_THRESHOLD=0.3            # Movement detection threshold
FPS_TARGET=30                     # Target frames per second

# Model Configuration
ENABLE_CUDA=false                 # Set to true if you have CUDA GPU
MODEL_CHECKPOINT=shufflenetv2k16  # Model architecture

# Server Configuration
HOST=0.0.0.0                      # Server host
PORT=8000                         # Server port
```

### Frontend Configuration

1. Copy the environment template:
```bash
cp HM_Frontend/.env.example HM_Frontend/.env
```

2. Edit `HM_Frontend/.env` with your settings:

```env
# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here

# Backend API Configuration
VITE_BACKEND_API_URL=http://localhost:8000
```

### Database Setup

1. Create a Supabase project at [supabase.com](https://supabase.com)

2. Run the database migrations:

```bash
# Using Supabase CLI
cd HM_Frontend/supabase
supabase db push

# Or manually apply migrations in the Supabase dashboard
# Navigate to SQL Editor and run each migration file in order
```

The migrations will create:
- `fall_events` table - Stores fall detection events
- `calendar_events` table - Patient appointments and schedules
- `medications` table - Medication information
- `medication_logs` table - Medication adherence tracking

---

## 🚀 Running the Application

### Development Mode

#### Option 1: Run Both Services Separately

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run backend server
cd Fall_Detection
python api_server.py
```

The backend will be available at `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
# Run frontend development server
cd HM_Frontend
pnpm dev
```

The frontend will be available at `http://localhost:8080`

#### Option 2: Python Module Mode

```bash
# Run backend as module (from project root)
python -m Fall_Detection.api_server
```

### Production Mode

#### Backend

```bash
cd Fall_Detection
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Frontend

```bash
cd HM_Frontend
pnpm build
pnpm preview
```

Or serve the `dist/` directory with your preferred web server (nginx, Apache, etc.)

---

## 🧪 Testing

### Backend Tests

```bash
# Run all tests
cd Fall_Detection
pytest tests/

# Run specific test file
pytest tests/test_api_server.py

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Frontend Tests

```bash
cd HM_Frontend

# Run tests
pnpm test

# Run tests in watch mode
pnpm test:watch

# Run tests with UI
pnpm test:ui

# Run tests once (CI mode)
pnpm test:run
```

---

## 📁 Project Structure

```
Health_Mate/
├── Fall_Detection/               # Backend (Python/FastAPI)
│   ├── api_server.py            # Main FastAPI application
│   ├── config_manager.py        # Configuration management
│   ├── fall_detector.py         # Fall detection logic
│   ├── stream_manager.py        # Video stream handling
│   ├── websocket_manager.py     # WebSocket connections
│   ├── medication_scheduler.py  # Medication reminders
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example            # Environment template
│   │
│   ├── network/                # Neural network models
│   │   ├── factory.py          # Model loading
│   │   ├── basenetworks.py     # Base architectures
│   │   ├── nets.py             # Network wrappers
│   │   └── heads.py            # Head networks
│   │
│   ├── decoder/                # Pose decoding
│   ├── encoder/                # Pose encoding
│   ├── transforms/             # Image transformations
│   └── tests/                  # Backend tests
│
├── HM_Frontend/                 # Frontend (React/TypeScript)
│   ├── src/
│   │   ├── main.tsx            # Application entry point
│   │   ├── App.tsx             # Root component
│   │   │
│   │   ├── components/         # React components
│   │   │   ├── fall-detection/ # Fall detection UI
│   │   │   ├── medicine/       # Medication management
│   │   │   ├── calendar/       # Calendar components
│   │   │   ├── patient/        # Patient-specific
│   │   │   ├── caregiver/      # Caregiver-specific
│   │   │   └── ui/             # shadcn/ui components
│   │   │
│   │   ├── pages/              # Page components
│   │   │   ├── patient/        # Patient pages
│   │   │   └── caregiver/      # Caregiver pages
│   │   │
│   │   ├── hooks/              # Custom React hooks
│   │   ├── types/              # TypeScript types
│   │   ├── integrations/       # External integrations
│   │   └── utils/              # Utility functions
│   │
│   ├── supabase/               # Supabase configuration
│   │   └── migrations/         # Database migrations
│   │
│   ├── package.json            # npm dependencies
│   ├── vite.config.ts          # Vite configuration
│   ├── tsconfig.json           # TypeScript config
│   └── .env.example            # Environment template
│
├── FIXES_SUMMARY.md            # Technical fixes documentation
├── FINAL_STATUS.md             # System status report
├── TEST_SERVER_STARTUP.md      # Testing guide
└── README.md                   # This file
```

---

## 📚 API Documentation

### REST API Endpoints

#### Health & Status

```http
GET /health
```
Returns server health status and system information.

**Response:**
```json
{
  "status": "healthy",
  "video_stream_connected": true,
  "ai_system_status": "running",
  "websocket_connections": 0,
  "version": "1.0.0"
}
```

#### Video Streaming

```http
GET /api/video/stream?patient_id={patient_id}
```
Returns MJPEG video stream with fall detection overlays.

**Parameters:**
- `patient_id` (optional) - Patient identifier for logging

**Response:** Multipart MJPEG stream

#### Configuration

```http
GET /api/config
```
Returns current server configuration.

#### Medication Management

```http
GET /api/medications
POST /api/medications
PUT /api/medications/{id}
DELETE /api/medications/{id}
```

### WebSocket Endpoints

```javascript
ws://localhost:8000/ws
```

Connect to receive real-time fall detection alerts and system notifications.

**Message Format:**
```json
{
  "type": "fall_detected",
  "timestamp": "2024-01-01T12:00:00Z",
  "patient_id": "patient_123",
  "confidence": 0.95,
  "location": "living_room"
}
```

### Full API Documentation

Once the server is running, visit:
- **Interactive Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🛠 Technologies Used

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Runtime environment | 3.8+ |
| **FastAPI** | Web framework | Latest |
| **OpenCV** | Computer vision | 4.5+ |
| **PyTorch** | Deep learning | Latest |
| **OpenPifPaf** | Pose estimation | Vendorized |
| **Uvicorn** | ASGI server | Latest |
| **Supabase Python** | Database client | Latest |
| **WebSockets** | Real-time communication | Latest |

### Frontend

| Technology | Purpose | Version |
|------------|---------|---------|
| **React** | UI framework | 18.3+ |
| **TypeScript** | Type safety | 5.5+ |
| **Vite** | Build tool | 6.3+ |
| **React Router** | Routing | 6.26+ |
| **TanStack Query** | Server state | 5.56+ |
| **shadcn/ui** | UI components | Latest |
| **Tailwind CSS** | Styling | 3.4+ |
| **Supabase JS** | Database client | 2.80+ |
| **date-fns** | Date utilities | 3.6+ |
| **Recharts** | Charts/graphs | 2.12+ |
| **Lucide React** | Icons | Latest |

### Infrastructure

| Service | Purpose |
|---------|---------|
| **Supabase** | Authentication, Database, Real-time |
| **Vercel** | Frontend hosting (optional) |

---

## 🐛 Troubleshooting

### Common Issues

#### 503 Error on Video Stream

**Symptoms:** Video stream endpoint returns 503 Service Unavailable

**Solutions:**
1. Check if webcam is available and not in use:
   ```bash
   # On Windows
   Get-Process | Where-Object {$_.ProcessName -like "*camera*"}
   ```

2. Verify video source in `.env`:
   ```env
   VIDEO_SOURCE=0  # Try different values: 0, 1, 2
   ```

3. Check model loading in logs:
   ```bash
   tail -f Fall_Detection/fall_detection_api.log
   ```

#### Model Loading Issues

**Symptoms:** AI model fails to load, server starts without fall detection

**Solutions:**
1. The system will automatically build a model from scratch if download fails
2. For better results, download a pretrained model:
   ```bash
   # Models are cached in:
   # ~/.cache/torch/hub/checkpoints/
   ```


#### Frontend Can't Connect to Backend

**Symptoms:** Network errors in browser console

**Solutions:**
1. Verify backend is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check Vite proxy configuration in `HM_Frontend/vite.config.ts`

3. Verify CORS settings in `Fall_Detection/api_server.py`

#### Database Connection Errors

**Symptoms:** Authentication failures, database errors

**Solutions:**
1. Verify Supabase credentials in `.env` files
2. Check Supabase project status in dashboard
3. Ensure migrations have been applied
4. Verify RLS (Row Level Security) policies in Supabase



---


---

## 🙏 Acknowledgments

- **OpenPifPaf** - Pose estimation model (vendorized from [openpifpaf/openpifpaf](https://github.com/openpifpaf/openpifpaf))
- **shadcn/ui** - Beautiful UI components
- **Supabase** - Backend infrastructure
- **FastAPI** - Modern Python web framework
- Original fall detection implementation based on [Yashi5769/fall-detection](https://github.com/Yashi5769/fall-detection)

---


<div align="center">

**Made for Project Work**


</div>
