/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SUPABASE_URL: string;
  readonly VITE_SUPABASE_ANON_KEY: string;
  readonly VITE_BACKEND_API_URL?: string;
  readonly VITE_GAZE_TRACKING_ENABLED?: string;
  readonly VITE_GAZE_TRACKING_FPS?: string;
  readonly VITE_GAZE_DWELL_TIME?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
