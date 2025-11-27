"use client";

import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { User, Stethoscope } from "lucide-react";
import { useSupabase } from "@/components/SessionContextProvider";
import { supabase } from "@/integrations/supabase/client";
import { showSuccess, showError } from "@/utils/toast";

const RoleSelectionPage: React.FC = () => {
  const { session, profile, loading, refreshProfile } = useSupabase();
  const navigate = useNavigate();

  useEffect(() => {
    if (loading) {
      return; // Still loading session/profile, do nothing yet
    }

    if (!session) {
      // Not logged in, redirect to front page
      navigate("/", { replace: true });
      return;
    }

    if (profile?.role) {
      // If role is already set, redirect to respective dashboard
      if (profile.role === 'patient') {
        navigate("/patient/message-input", { replace: true });
      } else if (profile.role === 'caregiver') {
        navigate("/caregiver/dashboard", { replace: true });
      }
    }
    // If session exists but no role (profile.role is null), stay on this page to select and save role
  }, [loading, session, profile, navigate]);

  const handleRoleSelect = async (role: 'patient' | 'caregiver') => {
    if (!session?.user) {
      showError("You must be logged in to set your role.");
      navigate("/"); 
      return;
    }

    try {
      const { error } = await supabase
        .from('profiles')
        .update({ role: role, updated_at: new Date().toISOString() })
        .eq('id', session.user.id);

      if (error) {
        throw error;
      }
      showSuccess(`Role set to ${role}. Redirecting...`);
      await refreshProfile(); // Refresh the profile in context
      // The useEffect will now handle the redirection after profile update
    } catch (error: any) {
      console.error("Error setting role:", error);
      showError(`Failed to set role: ${error.message}`);
    }
  };

  if (loading || !session || profile?.role) {
    // Show a loading state or nothing if already handled by useEffect
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    );
  }

  // If logged in but no role is set, show buttons to select and save role
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-primary to-primary-foreground p-4">
      <Card className="w-full max-w-md text-center shadow-lg">
        <CardHeader>
          <CardTitle className="text-4xl font-bold text-primary rubik-doodle-shadow-regular">Health Mate</CardTitle>
          <p className="text-lg text-muted-foreground mt-2">
            Welcome! Please select your role to continue.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Button
            className="w-full h-16 text-xl flex items-center justify-center gap-3"
            onClick={() => handleRoleSelect("patient")}
          >
            <User className="h-6 w-6" /> I am a Patient
          </Button>
          <Button
            className="w-full h-16 text-xl flex items-center justify-center gap-3"
            variant="outline"
            onClick={() => handleRoleSelect("caregiver")}
          >
            <Stethoscope className="h-6 w-6" /> I am a Caregiver
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default RoleSelectionPage;