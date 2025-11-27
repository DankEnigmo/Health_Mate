import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.0';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight request
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get('Authorization');
    console.log('Edge Function: Received Authorization header:', authHeader ? 'Present' : 'Missing'); // Log the header presence

    if (!authHeader) {
      console.error('Edge Function: Authorization header missing.');
      return new Response(JSON.stringify({ error: 'Unauthorized: Authorization header missing.' }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 401,
      });
    }

    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: authHeader }, // Use the validated header
        },
      }
    );

    const { data: { user }, error: userError } = await supabaseClient.auth.getUser();
    console.log('Edge Function: User data from getUser():', user ? `User ID: ${user.id}` : 'No user found'); // Log user ID or 'No user'
    if (userError) {
      console.error('Edge Function: Error getting user from token:', userError);
    }

    if (!user) {
      return new Response(JSON.stringify({ error: 'Unauthorized: No user found or invalid token.' }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 401,
      });
    }

    const { error: insertError } = await supabaseClient
      .from('emergency_alerts')
      .insert({ patient_id: user.id, message: 'SOS - Emergency!' });

    if (insertError) {
      console.error('Edge Function: Error inserting emergency alert:', insertError);
      return new Response(JSON.stringify({ error: insertError.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      });
    }

    return new Response(JSON.stringify({ message: 'Emergency alert sent successfully!' }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error: any) {
    console.error('Edge Function: Uncaught error:', error);
    return new Response(JSON.stringify({ error: error.message || 'An unexpected error occurred.' }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    });
  }
});