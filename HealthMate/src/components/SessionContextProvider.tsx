"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { Session, SupabaseClient, User } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";

interface Profile {
  id: string;
  first_name: string | null;
  last_name: string | null;
  avatar_url: string | null;
  role: 'patient' | 'caregiver' | null;
}

interface SupabaseContextType {
  supabase: SupabaseClient;
  session: Session | null;
  user: (User & { profile?: Profile }) | null;
  profile: Profile | null;
  loading: boolean;
  refreshProfile: () => Promise<void>;
}

const SupabaseContext = createContext<SupabaseContextType | undefined>(undefined);

export const SessionContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<(User & { profile?: Profile }) | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchUserProfile = useCallback(async (userId: string) => {
    const { data: profileData, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();

    if (error && error.code !== 'PGRST116') { // PGRST116 means no rows found
      console.error("Error fetching profile:", error);
      setProfile(null);
      return null;
    } else if (profileData) {
      setProfile(profileData);
      return profileData;
    }
    setProfile(null);
    return null;
  }, []);

  const refreshProfile = useCallback(async () => {
    if (session?.user?.id) {
      await fetchUserProfile(session.user.id);
    } else {
      setProfile(null);
    }
  }, [session?.user?.id, fetchUserProfile]);

  useEffect(() => {
    const handleAuthStateChange = async (currentSession: Session | null) => {
      setSession(currentSession);
      setUser(currentSession?.user || null);
      if (currentSession?.user) {
        const fetchedProfile = await fetchUserProfile(currentSession.user.id);
        setUser((prevUser) => (prevUser ? { ...prevUser, profile: fetchedProfile || undefined } : null));
      } else {
        setProfile(null);
        setUser(null);
      }
      setLoading(false);
    };

    // Initial session check
    supabase.auth.getSession().then(({ data: { session: initialSession } }) => {
      handleAuthStateChange(initialSession);
    });

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      handleAuthStateChange(currentSession);
    });

    return () => subscription.unsubscribe();
  }, [fetchUserProfile]);

  return (
    <SupabaseContext.Provider value={{ supabase, session, user, profile, loading, refreshProfile }}>
      {!loading && children}
    </SupabaseContext.Provider>
  );
};

export const useSupabase = () => {
  const context = useContext(SupabaseContext);
  if (context === undefined) {
    throw new Error("useSupabase must be used within a SessionContextProvider");
  }
  return context;
};