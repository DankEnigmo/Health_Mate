"use client";

import React from "react";
import { useSupabase } from "@/components/SessionContextProvider";
import { useMedicationSchedule } from "@/hooks/use-medication-schedule";
import { useMedicationLogs } from "@/hooks/use-medication-logs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Pill, Clock, Calendar, CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { format, parseISO, startOfDay, subDays } from "date-fns";
import { MedicationLog } from "@/types/medication";

const PatientMedications = () => {
  const { profile } = useSupabase();
  const patientId = profile?.id || "";
  const { medications, loading: scheduleLoading } = useMedicationSchedule(patientId);
  const { logs, loading: logsLoading } = useMedicationLogs(patientId, 7);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'taken':
        return (
          <Badge variant="default" className="bg-green-500">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Taken
          </Badge>
        );
      case 'missed':
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Missed
          </Badge>
        );
      case 'skipped':
        return (
          <Badge variant="secondary">
            <AlertCircle className="h-3 w-3 mr-1" />
            Skipped
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            Pending
          </Badge>
        );
    }
  };

  // Group logs by date
  const getLogsByDate = () => {
    const logsByDate = new Map<string, MedicationLog[]>();
    
    // Create entries for the last 7 days
    for (let i = 0; i < 7; i++) {
      const date = subDays(new Date(), i);
      const dateKey = format(startOfDay(date), 'yyyy-MM-dd');
      logsByDate.set(dateKey, []);
    }

    // Fill in the logs
    logs.forEach(log => {
      const dateKey = format(parseISO(log.scheduled_time), 'yyyy-MM-dd');
      if (logsByDate.has(dateKey)) {
        logsByDate.get(dateKey)!.push(log);
      }
    });

    return Array.from(logsByDate.entries())
      .sort((a, b) => b[0].localeCompare(a[0])); // Sort by date descending
  };

  const getMedicationName = (medicationId: string) => {
    const medication = medications.find(m => m.id === medicationId);
    return medication?.name || 'Unknown medication';
  };

  const getMedicationDosage = (medicationId: string) => {
    const medication = medications.find(m => m.id === medicationId);
    return medication?.dosage || '';
  };

  if (scheduleLoading || logsLoading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-primary">Your Medications</h1>
        <p className="text-lg text-muted-foreground">
          View your medication schedule and history.
        </p>
        <div className="animate-pulse space-y-4">
          <div className="h-96 bg-muted rounded-lg"></div>
        </div>
      </div>
    );
  }

  const logsByDate = getLogsByDate();

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary">Your Medications</h1>
      <p className="text-lg text-muted-foreground">
        View your medication schedule and history.
      </p>

      {/* Active Medication Schedules */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5" />
            Active Medication Schedule
          </CardTitle>
          <CardDescription>
            Your current medication prescriptions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {medications.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Pill className="mx-auto h-12 w-12 mb-4 opacity-50" />
              <p>No active medications</p>
            </div>
          ) : (
            <div className="space-y-4">
              {medications.map((medication) => (
                <Card key={medication.id} className="border-l-4 border-l-primary">
                  <CardContent className="pt-6">
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="text-lg font-semibold">{medication.name}</h3>
                          <p className="text-sm text-muted-foreground">{medication.dosage}</p>
                        </div>
                        <Badge variant="outline">{medication.frequency.replace('_', ' ')}</Badge>
                      </div>

                      {medication.instructions && (
                        <div className="p-2 bg-muted rounded-md">
                          <p className="text-sm text-muted-foreground">{medication.instructions}</p>
                        </div>
                      )}

                      <div className="flex flex-wrap gap-2">
                        {medication.times.map((time) => (
                          <Badge key={time} variant="secondary" className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {time}
                          </Badge>
                        ))}
                      </div>

                      <div className="text-xs text-muted-foreground">
                        Started: {format(parseISO(medication.start_date), 'MMM d, yyyy')}
                        {medication.end_date && ` • Ends: ${format(parseISO(medication.end_date), 'MMM d, yyyy')}`}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 7-Day Medication History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            7-Day Medication History
          </CardTitle>
          <CardDescription>
            Your medication adherence over the past week
          </CardDescription>
        </CardHeader>
        <CardContent>
          {logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Calendar className="mx-auto h-12 w-12 mb-4 opacity-50" />
              <p>No medication history available</p>
            </div>
          ) : (
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-6">
                {logsByDate.map(([dateKey, dateLogs]) => {
                  const dateObj = parseISO(dateKey);
                  const isToday = format(new Date(), 'yyyy-MM-dd') === dateKey;
                  
                  return (
                    <div key={dateKey} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-lg">
                          {isToday ? 'Today' : format(dateObj, 'EEEE')}
                        </h3>
                        <span className="text-sm text-muted-foreground">
                          {format(dateObj, 'MMM d, yyyy')}
                        </span>
                      </div>
                      
                      {dateLogs.length === 0 ? (
                        <div className="pl-4 border-l-2 border-muted">
                          <p className="text-sm text-muted-foreground py-2">
                            No medications scheduled
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-2 pl-4 border-l-2 border-muted">
                          {dateLogs
                            .sort((a, b) => a.scheduled_time.localeCompare(b.scheduled_time))
                            .map((log) => (
                              <div
                                key={log.id}
                                className="flex items-center justify-between p-3 rounded-md bg-muted/50 hover:bg-muted transition-colors"
                              >
                                <div className="flex items-center gap-3 flex-1">
                                  <Pill className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                  <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium truncate">
                                      {getMedicationName(log.medication_id)}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                      {getMedicationDosage(log.medication_id)}
                                    </p>
                                  </div>
                                  <div className="flex items-center gap-2 text-xs text-muted-foreground flex-shrink-0">
                                    <Clock className="h-3 w-3" />
                                    <span>{format(parseISO(log.scheduled_time), 'h:mm a')}</span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                                  {getStatusBadge(log.status)}
                                  {log.status === 'taken' && log.taken_time && (
                                    <span className="text-xs text-muted-foreground">
                                      at {format(parseISO(log.taken_time), 'h:mm a')}
                                    </span>
                                  )}
                                </div>
                              </div>
                            ))}
                        </div>
                      )}
                      
                      {dateKey !== logsByDate[logsByDate.length - 1][0] && (
                        <Separator className="mt-4" />
                      )}
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PatientMedications;
