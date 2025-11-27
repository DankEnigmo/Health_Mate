"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BellRing, User, Clock, CheckCircle } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { showError, showSuccess } from "@/utils/toast";
import { useCaregiverPatient } from "@/hooks/use-caregiver-patient";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface EmergencyAlert {
  id: string;
  patient_id: string;
  message: string;
  status: string;
  created_at: string;
  profiles?: {
    first_name: string | null;
    last_name: string | null;
  };
}

const EmergencyAlertsDisplay: React.FC = () => {
  const { patientId, loading: patientLoading, error: patientError } = useCaregiverPatient();
  const [alerts, setAlerts] = useState<EmergencyAlert[]>([]);
  const [loadingAlerts, setLoadingAlerts] = useState(true);

  const fetchAlerts = useCallback(async () => {
    setLoadingAlerts(true);
    if (!patientId) {
      setAlerts([]);
      setLoadingAlerts(false);
      return;
    }

    const { data, error } = await supabase
      .from('emergency_alerts')
      .select(`
        id,
        patient_id,
        message,
        status,
        created_at,
        profiles (first_name, last_name)
      `)
      .eq('patient_id', patientId)
      .order('created_at', { ascending: false });

    if (error) {
          console.error("Error fetching emergency alerts:", error);
          showError("Failed to load emergency alerts.");
        } else {
          // FIX STARTS HERE: Transform the raw data to match your Interface
          const formattedData: EmergencyAlert[] = (data || []).map((item: any) => ({
            ...item,
            profiles: Array.isArray(item.profiles) && item.profiles.length > 0
              ? item.profiles[0] 
              : { first_name: 'Unknown', last_name: 'User' } 
          }));

          setAlerts(formattedData);
        }
        setLoadingAlerts(false);
      }, [patientId]);

  useEffect(() => {
    if (patientLoading) return;

    if (patientError) {
      setLoadingAlerts(false);
      return;
    }

    fetchAlerts();

    // Realtime subscription for new alerts for the assigned patient
    const channel = supabase
      .channel(`emergency_alerts_for_${patientId}`)
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'emergency_alerts', filter: `patient_id=eq.${patientId}` },
        (payload) => {
          const newAlert = payload.new as EmergencyAlert;
          // Fetch profile data for the new alert
          supabase
            .from('profiles')
            .select('first_name, last_name')
            .eq('id', newAlert.patient_id)
            .single()
            .then(({ data: profileData, error: profileError }) => {
              if (profileError) {
                console.error("Error fetching profile for new alert:", profileError);
              } else {
                setAlerts((prevAlerts) => [{ ...newAlert, profiles: profileData || undefined }, ...prevAlerts]);
                showError(`New Emergency Alert from ${profileData?.first_name || 'your patient'}!`);
              }
            });
        }
      )
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'emergency_alerts', filter: `patient_id=eq.${patientId}` },
        (payload) => {
          const updatedAlert = payload.new as EmergencyAlert;
          setAlerts((prevAlerts) =>
            prevAlerts.map((alert) =>
              alert.id === updatedAlert.id ? { ...alert, ...updatedAlert } : alert
            )
          );
          showSuccess(`Alert for ${updatedAlert.profiles?.first_name || 'patient'} updated to ${updatedAlert.status}.`);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [patientId, patientLoading, patientError, fetchAlerts]);

  const handleMarkAsResolved = async (alertId: string) => {
    const { error } = await supabase
      .from('emergency_alerts')
      .update({ status: 'resolved' })
      .eq('id', alertId);

    if (error) {
      console.error("Error marking alert as resolved:", error);
      showError("Failed to mark alert as resolved.");
    } else {
      showSuccess("Alert marked as resolved.");
      // The realtime subscription will update the state, no need to manually refetch
    }
  };

  if (patientLoading || loadingAlerts) {
    return <p className="text-muted-foreground">Loading emergency alerts...</p>;
  }

  if (patientError) {
    return <p className="text-destructive">Error: {patientError}</p>;
  }

  if (!patientId) {
    return <p className="text-muted-foreground">No patient assigned. Please assign a patient to view alerts.</p>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Emergency Alerts for Your Patient</h2>
      {alerts.length === 0 ? (
        <p className="text-muted-foreground">No emergency alerts received from your patient.</p>
      ) : (
        <div className="grid gap-4">
          {alerts.map((alert) => {
            const patientFullName = alert.profiles?.first_name || alert.profiles?.last_name
              ? `${alert.profiles.first_name || ''} ${alert.profiles.last_name || ''}`.trim()
              : 'Unknown Patient';

            const isNew = alert.status.toLowerCase() === 'new';

            return (
              <Card
                key={alert.id}
                className={cn(
                  "hover:shadow-md transition-shadow",
                  isNew && "border-destructive ring-destructive ring-1 animate-pulse"
                )}
              >
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className={cn(
                    "text-lg font-semibold flex items-center gap-2",
                    isNew ? "text-destructive" : "text-foreground"
                  )}>
                    <BellRing className="h-5 w-5" />
                    {isNew ? "EMERGENCY SOS!" : "Alert"}
                  </CardTitle>
                  <span className={cn(
                    "px-2 py-1 text-xs font-medium rounded-full",
                    isNew && "bg-destructive text-destructive-foreground",
                    alert.status.toLowerCase() === 'resolved' && "bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100",
                    alert.status.toLowerCase() === 'reviewed' && "bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100"
                  )}>
                    {alert.status.toUpperCase()}
                  </span>
                </CardHeader>
                <CardContent className="space-y-2">
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <User className="h-4 w-4" /> Patient: <span className="font-medium text-foreground">{patientFullName}</span>
                  </p>
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <Clock className="h-4 w-4" /> Time: <span className="font-medium text-foreground">{new Date(alert.created_at).toLocaleString()}</span>
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Message: <span className="font-medium text-foreground">{alert.message}</span>
                  </p>
                  {isNew && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-4 bg-green-500 hover:bg-green-600 text-white"
                      onClick={() => handleMarkAsResolved(alert.id)}
                    >
                      <CheckCircle className="h-4 w-4 mr-2" /> Mark as Resolved
                    </Button>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default EmergencyAlertsDisplay;