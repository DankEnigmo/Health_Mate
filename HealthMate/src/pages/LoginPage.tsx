"use client";

import React, { useEffect } from "react";
import { Auth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate, useSearchParams } from "react-router-dom"; // Import useSearchParams
import { useSupabase } from "@/components/SessionContextProvider";

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams(); // Hook to read query parameters
  const { session, loading, profile } = useSupabase();

  // Determine the default view for the Auth component
  const defaultAuthView = searchParams.get('view') === 'signup' ? 'sign_up' : 'sign_in';

  useEffect(() => {
    if (!loading && session) {
      if (!profile?.role) {
        // If logged in but no role, redirect to role selection
        navigate("/select-role", { replace: true });
      } else {
        // If logged in and has a role, redirect to respective dashboard
        if (profile.role === 'patient') {
          navigate("/patient/message-input", { replace: true });
        } else if (profile.role === 'caregiver') {
          navigate("/caregiver/dashboard", { replace: true });
        }
      }
    }
  }, [session, loading, profile, navigate]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-primary to-primary-foreground p-4">
      <div className="w-full max-w-md p-8 bg-card rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-center text-primary mb-6 rubik-doodle-shadow-regular">
          Health Mate
        </h1>
        <Auth
          supabaseClient={supabase}
          providers={[]}
          appearance={{
            theme: ThemeSupa,
            variables: {
              default: {
                colors: {
                  brand: 'hsl(var(--primary))',
                  brandAccent: 'hsl(var(--primary-foreground))',
                },
              },
            },
          }}
          theme="light"
          defaultView={defaultAuthView} // Set the default view based on URL parameter
        />
      </div>
    </div>
  );
};

export default LoginPage;