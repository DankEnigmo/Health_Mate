export type MedicationFrequency = 
  | 'daily' 
  | 'twice_daily' 
  | 'three_times_daily' 
  | 'weekly' 
  | 'as_needed';

export interface MedicationSchedule {
  id: string;
  patient_id: string;
  caregiver_id: string;
  name: string;
  dosage: string;
  frequency: MedicationFrequency;
  times: string[]; // Array of HH:mm format times
  instructions?: string;
  start_date: string; // ISO date string
  end_date?: string; // ISO date string
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface MedicationScheduleFormData {
  name: string;
  dosage: string;
  frequency: MedicationFrequency;
  times: string[];
  instructions?: string;
  start_date: Date;
  end_date?: Date;
}

export interface MedicationLog {
  id: string;
  medication_id: string;
  patient_id: string;
  scheduled_time: string;
  taken_time?: string;
  status: 'pending' | 'taken' | 'missed' | 'skipped';
  notes?: string;
  created_at: string;
}

export interface MedicationWithStatus extends MedicationSchedule {
  todaysLogs: MedicationLog[];
  nextScheduledTime?: string;
  lastTakenTime?: string;
}

export interface AdherenceStats {
  totalScheduled: number;
  totalTaken: number;
  totalMissed: number;
  adherenceRate: number; // Percentage
  recentLogs: MedicationLog[];
}
