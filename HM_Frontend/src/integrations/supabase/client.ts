import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase URL or Anon Key is missing. Please check your .env file.');
  // Provide dummy values or throw an error to prevent app from crashing in development
  // In a production environment, you might want to handle this more robustly.
  throw new Error('Supabase environment variables are not set.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);