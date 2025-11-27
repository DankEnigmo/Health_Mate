import { useMedicationLogs } from '@/hooks/use-medication-logs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Pill, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle,
  Calendar
} from 'lucide-react';
import { format, parseISO, startOfDay, subDays } from 'date-fns';
import { MedicationLog } from '@/types/medication';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

interface MedicationAdherenceProps {
  patientId: string;
  days?: number;
}

export function MedicationAdherence({ patientId, days = 30 }: MedicationAdherenceProps) {
  const { adherenceStats, logs, loading, error } = useMedicationLogs(patientId, days);

  // Calculate adherence percentage
  const adherencePercentage = adherenceStats.adherenceRate;
  const isLowAdherence = adherencePercentage < 80;

  // Get missed medications
  const missedMedications = logs.filter(log => log.status === 'missed');

  // Group logs by date for detailed history
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
      .sort((a, b) => b[0].localeCompare(a[0])); // Sort by date descending
  };

  // Calculate daily adherence for chart
  const getDailyAdherence = () => {
    const dailyData: { date: string; adherence: number; taken: number; total: number }[] = [];
    const historyByDate = getHistoryByDate();

    historyByDate.forEach(([dateKey, dateLogs]) => {
      const total = dateLogs.length;
      const taken = dateLogs.filter(log => log.status === 'taken').length;
      const adherence = total > 0 ? (taken / total) * 100 : 0;

      dailyData.push({
        date: dateKey,
        adherence,
        taken,
        total,
      });
    });

    return dailyData;
  };

  const dailyAdherence = getDailyAdherence();

  // Get adherence color based on percentage
  const getAdherenceColor = (percentage: number) => {
    if (percentage >= 90) return 'text-green-600 dark:text-green-400';
    if (percentage >= 80) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getAdherenceIcon = (percentage: number) => {
    if (percentage >= 80) {
      return <TrendingUp className="h-5 w-5 text-green-600 dark:text-green-400" />;
    }
    return <TrendingDown className="h-5 w-5 text-red-600 dark:text-red-400" />;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Medication Adherence</CardTitle>
          <CardDescription>Loading adherence data...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Medication Adherence</CardTitle>
          <CardDescription className="text-destructive">
            Error loading adherence data: {error}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Low Adherence Warning */}
      {isLowAdherence && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Low Medication Adherence</AlertTitle>
          <AlertDescription>
            The patient's medication adherence is below 80%. Please review missed medications and
            consider interventions to improve adherence.
          </AlertDescription>
        </Alert>
      )}

      {/* Adherence Overview Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5" />
            Medication Adherence
          </CardTitle>
          <CardDescription>
            Last {days} days of medication adherence data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Adherence Percentage */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">Overall Adherence</span>
                  {getAdherenceIcon(adherencePercentage)}
                </div>
                <span className={`text-3xl font-bold ${getAdherenceColor(adherencePercentage)}`}>
                  {adherencePercentage.toFixed(1)}%
                </span>
              </div>
              <Progress 
                value={adherencePercentage} 
                className="h-3"
              />
            </div>

            {/* Statistics Grid */}
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Scheduled</p>
                <p className="text-2xl font-bold">{adherenceStats.totalScheduled}</p>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                  <p className="text-sm text-muted-foreground">Taken</p>
                </div>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {adherenceStats.totalTaken}
                </p>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-1">
                  <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                  <p className="text-sm text-muted-foreground">Missed</p>
                </div>
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {adherenceStats.totalMissed}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Missed Medications */}
      {missedMedications.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
              Missed Medications
            </CardTitle>
            <CardDescription>
              {missedMedications.length} medication{missedMedications.length !== 1 ? 's' : ''} missed in the last {days} days
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[200px] pr-4">
              <div className="space-y-2">
                {missedMedications.map((log) => (
                  <div
                    key={log.id}
                    className="flex items-center justify-between p-3 rounded-md border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950"
                  >
                    <div className="flex items-center gap-2">
                      <Pill className="h-4 w-4 text-red-600 dark:text-red-400" />
                      <div>
                        <p className="text-sm font-medium">Medication ID: {log.medication_id.substring(0, 8)}...</p>
                        <p className="text-xs text-muted-foreground">
                          {format(parseISO(log.scheduled_time), 'MMM d, yyyy • h:mm a')}
                        </p>
                      </div>
                    </div>
                    <Badge variant="destructive">
                      <XCircle className="h-3 w-3 mr-1" />
                      Missed
                    </Badge>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* Daily Adherence Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Daily Adherence History
          </CardTitle>
          <CardDescription>
            Adherence percentage by day
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px] pr-4">
            <div className="space-y-3">
              {dailyAdherence.map((day) => (
                <div key={day.date} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">
                      {format(parseISO(day.date), 'EEE, MMM d')}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {day.taken}/{day.total}
                      </span>
                      <span className={`text-sm font-bold ${getAdherenceColor(day.adherence)}`}>
                        {day.adherence.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <Progress 
                    value={day.adherence} 
                    className="h-2"
                  />
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Detailed History */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Medication History</CardTitle>
          <CardDescription>
            Complete log of all medications for the last {days} days
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-4">
              {getHistoryByDate().map(([dateKey, dateLogs]) => (
                <div key={dateKey} className="space-y-2">
                  <Separator />
                  <h4 className="font-medium text-sm text-muted-foreground pt-2">
                    {format(parseISO(dateKey), 'EEEE, MMMM d, yyyy')}
                  </h4>
                  <div className="space-y-2 pl-4 border-l-2 border-muted">
                    {dateLogs.map((log) => {
                      const isTaken = log.status === 'taken';
                      const isMissed = log.status === 'missed';
                      
                      return (
                        <div
                          key={log.id}
                          className={`flex items-center justify-between p-3 rounded-md border ${
                            isTaken 
                              ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800'
                              : isMissed
                              ? 'bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800'
                              : 'bg-background'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <Pill className="h-4 w-4 text-muted-foreground" />
                            <div>
                              <p className="text-sm font-medium">
                                Medication ID: {log.medication_id.substring(0, 8)}...
                              </p>
                              <p className="text-xs text-muted-foreground">
                                Scheduled: {format(parseISO(log.scheduled_time), 'h:mm a')}
                                {log.taken_time && (
                                  <> • Taken: {format(parseISO(log.taken_time), 'h:mm a')}</>
                                )}
                              </p>
                            </div>
                          </div>
                          <Badge 
                            variant={isTaken ? 'default' : isMissed ? 'destructive' : 'outline'}
                            className={isTaken ? 'bg-green-500' : ''}
                          >
                            {isTaken && <CheckCircle2 className="h-3 w-3 mr-1" />}
                            {isMissed && <XCircle className="h-3 w-3 mr-1" />}
                            {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
