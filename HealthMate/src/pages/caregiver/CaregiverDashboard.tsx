"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CalendarDays, BellRing, MessageSquareText, User, AlertTriangle } from "lucide-react";
import EmergencyAlertsDisplay from "@/components/caregiver/EmergencyAlertsDisplay";
import { FallAlertsDisplay } from "@/components/fall-detection/FallAlertsDisplay";
import { UpcomingEvents } from "@/components/calendar/UpcomingEvents";
import { MedicationAdherence } from "@/components/medicine/MedicationAdherence";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { useCalendarEvents } from "@/hooks/use-calendar-events";
import { supabase } from "@/integrations/supabase/client";
import { showError } from "@/utils/toast";

const CaregiverDashboard: React.FC = () => {
  const { patientId, loading: patientLoading, error: patientError } = useCaregiverPatient();
  const { events, loading: eventsLoading } = useCalendarEvents(patientId || '');
  const [patientName, setPatientName] = useState<string | null>(null);
  const [fallStats, setFallStats] = useState<{
    totalFalls: number;
    newAlerts: number;
    todayFalls: number;
  }>({
    totalFalls: 0,
    newAlerts: 0,
    todayFalls: 0,
  });

  useEffect(() => {
    const fetchPatientProfile = async () => {
      if (patientId) {
        const { data, error } = await supabase
          .from('profiles')
          .select('first_name, last_name')
          .eq('id', patientId)
          .single();

        if (error) {
          console.error("Error fetching patient profile:", error);
          showError("Failed to fetch patient profile.");
          setPatientName(null);
        } else if (data) {
          setPatientName(`${data.first_name || ''} ${data.last_name || ''}`.trim());
        }
      } else {
        setPatientName(null);
      }
    };

    if (!patientLoading && !patientError) {
      fetchPatientProfile();
    }
  }, [patientId, patientLoading, patientError]);

  // Fetch fall event statistics
  useEffect(() => {
    const fetchFallStats = async () => {
      if (!patientId) {
        setFallStats({ totalFalls: 0, newAlerts: 0, todayFalls: 0 });
        return;
      }

      try {
        // Get total fall events
        const { count: totalCount, error: totalError } = await supabase
          .from('fall_events')
          .select('*', { count: 'exact', head: true })
          .eq('patient_id', patientId);

        if (totalError) {
          console.error("Error fetching total fall count:", totalError);
        }

        // Get new alerts (status = 'new')
        const { count: newCount, error: newError } = await supabase
          .from('fall_events')
          .select('*', { count: 'exact', head: true })
          .eq('patient_id', patientId)
          .eq('status', 'new');

        if (newError) {
          console.error("Error fetching new alerts count:", newError);
        }

        // Get today's falls
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todayISO = today.toISOString();

        const { count: todayCount, error: todayError } = await supabase
          .from('fall_events')
          .select('*', { count: 'exact', head: true })
          .eq('patient_id', patientId)
          .gte('timestamp', todayISO);

        if (todayError) {
          console.error("Error fetching today's fall count:", todayError);
        }

        setFallStats({
          totalFalls: totalCount || 0,
          newAlerts: newCount || 0,
          todayFalls: todayCount || 0,
        });
      } catch (err) {
        console.error("Error fetching fall statistics:", err);
      }
    };

    if (!patientLoading && !patientError) {
      fetchFallStats();

      // Subscribe to realtime updates for fall_events
      const channel = supabase
        .channel('fall_stats_updates')
        .on(
          'postgres_changes',
          {
            event: '*',
            schema: 'public',
            table: 'fall_events',
            filter: `patient_id=eq.${patientId}`,
          },
          () => {
            // Refetch stats when fall_events change
            fetchFallStats();
          }
        )
        .subscribe();

      return () => {
        supabase.removeChannel(channel);
      };
    }
  }, [patientId, patientLoading, patientError]);

  if (patientLoading) {
    return <p className="text-muted-foreground p-4">Loading caregiver dashboard...</p>;
  }

  if (patientError) {
    return <p className="text-destructive p-4">Error: {patientError}</p>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary sansation-bold">Caregiver Dashboard</h1>
      <p className="text-lg text-muted-foreground">
        Welcome back! Here's a quick overview for {patientName ? <span className="font-semibold">{patientName}</span> : "your assigned patient"}.
      </p>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {patientId ? (
          <>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Assigned Patient</CardTitle>
                <User className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{patientName || "Loading..."}</div>
                <p className="text-xs text-muted-foreground">
                  ID: {patientId.substring(0, 8)}...
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Fall Events</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{fallStats.totalFalls}</div>
                <p className="text-xs text-muted-foreground">
                  {fallStats.todayFalls} detected today
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">New Fall Alerts</CardTitle>
                <BellRing className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-destructive">{fallStats.newAlerts}</div>
                <p className="text-xs text-muted-foreground">
                  Requiring attention
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Upcoming Events
                </CardTitle>
                <CalendarDays className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {eventsLoading ? '...' : events.filter(e => new Date(e.event_date) >= new Date()).length}
                </div>
                <p className="text-xs text-muted-foreground">
                  Scheduled for {patientName || 'patient'}
                </p>
              </CardContent>
            </Card>
          </>
        ) : (
          <Card className="col-span-full">
            <CardHeader>
              <CardTitle className="text-lg font-medium text-destructive">No Patient Assigned</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                You currently do not have a patient assigned. Please ensure a patient is linked to your caregiver account.
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Fall Alerts Display */}
      {patientId && (
        <div className="grid gap-6 lg:grid-cols-2">
          <FallAlertsDisplay 
            patientId={patientId}
            maxAlerts={10}
            showHistory={false}
            className="lg:col-span-1"
          />
          
          {/* Emergency Alerts Display */}
          <div className="lg:col-span-1">
            <EmergencyAlertsDisplay />
          </div>
        </div>
      )}

      {!patientId && <EmergencyAlertsDisplay />}

      {/* Upcoming Events Widget */}
      {patientId && (
        <UpcomingEvents
          events={events}
          limit={5}
          loading={eventsLoading}
          calendarLink="/caregiver/calendar"
        />
      )}

      {/* Medication Adherence Display */}
      {patientId && (
        <MedicationAdherence patientId={patientId} days={30} />
      )}

      <div className="p-4 border rounded-lg bg-card">
        <h2 className="text-xl font-semibold mb-2">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>View Patient Profile</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Access detailed information for {patientName || "your patient"}.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Schedule New Appointment</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Book a new meeting or check-up for {patientName || "your patient"}.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default CaregiverDashboard;