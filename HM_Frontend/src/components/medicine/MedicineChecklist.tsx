import { useMedicationLogs } from '@/hooks/use-medication-logs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Pill, Clock, CheckCircle2, XCircle, Calendar } from 'lucide-react';
import { format, parseISO } from 'date-fns';
import { MedicationLog } from '@/types/medication';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';

interface MedicineChecklistProps {
  patientId: string;
  date?: Date;
}

export function MedicineChecklist({ patientId, date = new Date() }: MedicineChecklistProps) {
  const { todaysMedications, logs, loading, error, logMedication } = useMedicationLogs(patientId, 7);

  const handleCheckboxChange = async (medicationId: string, scheduledTime: string, isChecked: boolean) => {
    if (isChecked) {
      try {
        await logMedication(medicationId, scheduledTime);
      } catch (err) {
        console.error('Failed to log medication:', err);
      }
    }
  };

  const getStatusForTime = (medicationId: string, time: string): MedicationLog | undefined => {
    const todayDateStr = format(date, 'yyyy-MM-dd');
    const scheduledTime = `${todayDateStr}T${time}:00`;
    
    return logs.find(
      log => 
        log.medication_id === medicationId && 
        log.scheduled_time.startsWith(scheduledTime.substring(0, 16)) // Match up to minutes
    );
  };

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

  // Get past 7 days of logs grouped by date
  const getHistoryByDate = () => {
    const historyMap = new Map<string, MedicationLog[]>();
    
    logs.forEach(log => {
      const dateKey = format(parseISO(log.scheduled_time), 'yyyy-MM-dd');
      if (!historyMap.has(dateKey)) {
        historyMap.set(dateKey, []);
      }
      historyMap.get(dateKey)!.push(log);
    });

    return Array.from(historyMap.entries())
      .sort((a, b) => b[0].localeCompare(a[0])) // Sort by date descending
      .slice(0, 7); // Last 7 days
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Today's Medications</CardTitle>
          <CardDescription>Loading medications...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Today's Medications</CardTitle>
          <CardDescription className="text-destructive">
            Error loading medications: {error}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const historyByDate = getHistoryByDate();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Today's Medications</CardTitle>
        <CardDescription>
          {format(date, 'EEEE, MMMM d, yyyy')}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {todaysMedications.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Pill className="mx-auto h-12 w-12 mb-4 opacity-50" />
            <p>No medications scheduled for today</p>
          </div>
        ) : (
          <div className="space-y-4">
            {todaysMedications.map((medication) => (
              <Card key={medication.id} className="border-l-4 border-l-primary">
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="text-lg font-semibold">{medication.name}</h3>
                        <p className="text-sm text-muted-foreground">{medication.dosage}</p>
                      </div>
                    </div>

                    {medication.instructions && (
                      <div className="p-2 bg-muted rounded-md">
                        <p className="text-sm text-muted-foreground">{medication.instructions}</p>
                      </div>
                    )}

                    <div className="space-y-2">
                      {medication.times.map((time) => {
                        const todayDateStr = format(date, 'yyyy-MM-dd');
                        const scheduledTime = `${todayDateStr}T${time}:00`;
                        const logEntry = getStatusForTime(medication.id, time);
                        const isTaken = logEntry?.status === 'taken';

                        return (
                          <div
                            key={time}
                            className={`flex items-center justify-between p-3 rounded-md border ${
                              isTaken ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800' : 'bg-background'
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              <Checkbox
                                id={`${medication.id}-${time}`}
                                checked={isTaken}
                                onCheckedChange={(checked) =>
                                  handleCheckboxChange(medication.id, scheduledTime, checked as boolean)
                                }
                                disabled={isTaken}
                              />
                              <label
                                htmlFor={`${medication.id}-${time}`}
                                className={`flex items-center gap-2 cursor-pointer ${
                                  isTaken ? 'line-through text-muted-foreground' : ''
                                }`}
                              >
                                <Clock className="h-4 w-4" />
                                <span className="font-medium">{time}</span>
                              </label>
                            </div>
                            <div className="flex items-center gap-2">
                              {logEntry && getStatusBadge(logEntry.status)}
                              {isTaken && logEntry?.taken_time && (
                                <span className="text-xs text-muted-foreground">
                                  {format(parseISO(logEntry.taken_time), 'h:mm a')}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {historyByDate.length > 0 && (
          <>
            <Separator className="my-6" />
            <div>
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Medication History (Last 7 Days)
              </h3>
              <ScrollArea className="h-[300px] pr-4">
                <div className="space-y-4">
                  {historyByDate.map(([dateKey, dateLogs]) => (
                    <div key={dateKey} className="space-y-2">
                      <h4 className="font-medium text-sm text-muted-foreground">
                        {format(parseISO(dateKey), 'EEEE, MMM d')}
                      </h4>
                      <div className="space-y-2 pl-4 border-l-2 border-muted">
                        {dateLogs.map((log) => {
                          const medication = todaysMedications.find(m => m.id === log.medication_id);
                          return (
                            <div
                              key={log.id}
                              className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                            >
                              <div className="flex items-center gap-2">
                                <Pill className="h-4 w-4 text-muted-foreground" />
                                <span className="text-sm">
                                  {medication?.name || 'Unknown medication'}
                                </span>
                                <span className="text-xs text-muted-foreground">
                                  {format(parseISO(log.scheduled_time), 'h:mm a')}
                                </span>
                              </div>
                              {getStatusBadge(log.status)}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
