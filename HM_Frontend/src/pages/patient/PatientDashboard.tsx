"use client";

import React, { useState, useEffect } from "react";
import { useSupabase } from "@/components/SessionContextProvider";
import UpdateProfileDialog from "@/components/patient/UpdateProfileDialog";
import { UpcomingEvents } from "@/components/calendar/UpcomingEvents";
import { useCalendarEvents } from "@/hooks/use-calendar-events";
import { MedicineChecklist } from "@/components/medicine/MedicineChecklist";

const PatientDashboard: React.FC = () => {
  const { profile, loading } = useSupabase();
  const [showProfileDialog, setShowProfileDialog] = useState(false);
  const patientId = profile?.id || "";
  const { events, loading: eventsLoading } = useCalendarEvents(patientId);

  useEffect(() => {
    if (!loading && profile && (!profile.first_name || !profile.last_name)) {
      setShowProfileDialog(true);
    } else if (!loading && profile && profile.first_name && profile.last_name) {
      setShowProfileDialog(false);
    }
  }, [loading, profile]);

  const patientName = profile?.first_name && profile?.last_name
    ? `${profile.first_name} ${profile.last_name}`
    : "Patient";

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary sansation-bold">Welcome, {patientName}!</h1>
      <p className="text-lg text-muted-foreground">
        Here's a quick overview of your health information.
      </p>

      <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-1">
        <UpcomingEvents
          events={events}
          limit={5}
          loading={eventsLoading}
          calendarLink="/patient/calendar"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-1">
        <MedicineChecklist patientId={patientId} />
      </div>

      <div className="p-4 border rounded-lg bg-card">
        <p className="text-muted-foreground">
          This is your health overview. Navigate to "Message Pad" to use the keyboard.
        </p>
      </div>

      <UpdateProfileDialog
        isOpen={showProfileDialog}
        onClose={() => setShowProfileDialog(false)}
      />
    </div>
  );
};

export default PatientDashboard;