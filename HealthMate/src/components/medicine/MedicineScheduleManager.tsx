import { useState } from 'react';
import { useMedicationSchedule } from '@/hooks/use-medication-schedule';
import { MedicationSchedule } from '@/types/medication';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Plus, Pill, Clock, Calendar, Edit, Trash2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { MedicineScheduleDialog } from './MedicineScheduleDialog';
import { format } from 'date-fns';

interface MedicineScheduleManagerProps {
  patientId: string;
  caregiverId: string;
}

export function MedicineScheduleManager({ patientId, caregiverId }: MedicineScheduleManagerProps) {
  const { medications, loading, error, createMedication, updateMedication, deleteMedication } =
    useMedicationSchedule(patientId);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedMedication, setSelectedMedication] = useState<MedicationSchedule | undefined>();

  const handleCreate = () => {
    setSelectedMedication(undefined);
    setIsDialogOpen(true);
  };

  const handleEdit = (medication: MedicationSchedule) => {
    setSelectedMedication(medication);
    setIsDialogOpen(true);
  };

  const handleDelete = async (medId: string) => {
    if (confirm('Are you sure you want to delete this medication schedule?')) {
      await deleteMedication(medId);
    }
  };

  const handleSave = async (
    medData: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'>
  ) => {
    if (selectedMedication) {
      await updateMedication({
        ...selectedMedication,
        ...medData,
      });
    } else {
      await createMedication(medData);
    }
  };

  const getFrequencyLabel = (frequency: string): string => {
    const labels: Record<string, string> = {
      daily: 'Daily',
      twice_daily: 'Twice Daily',
      three_times_daily: 'Three Times Daily',
      weekly: 'Weekly',
      as_needed: 'As Needed',
    };
    return labels[frequency] || frequency;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Medication Schedules</CardTitle>
          <CardDescription>Loading medication schedules...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Medication Schedules</CardTitle>
          <CardDescription className="text-destructive">
            Error loading medication schedules: {error}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Medication Schedules</CardTitle>
              <CardDescription>
                Manage medication schedules for your patient
              </CardDescription>
            </div>
            <Button onClick={handleCreate}>
              <Plus className="mr-2 h-4 w-4" />
              Add Medication
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {medications.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Pill className="mx-auto h-12 w-12 mb-4 opacity-50" />
              <p>No medication schedules yet</p>
              <p className="text-sm mt-2">Click "Add Medication" to create a schedule</p>
            </div>
          ) : (
            <div className="space-y-4">
              {medications.map((medication) => (
                <Card key={medication.id} className="border-l-4 border-l-primary">
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="text-lg font-semibold">{medication.name}</h3>
                          <Badge variant="secondary">{medication.dosage}</Badge>
                          <Badge variant="outline">{getFrequencyLabel(medication.frequency)}</Badge>
                        </div>

                        <div className="space-y-2 text-sm text-muted-foreground">
                          {medication.frequency !== 'as_needed' && medication.times.length > 0 && (
                            <div className="flex items-center gap-2">
                              <Clock className="h-4 w-4" />
                              <span>
                                Times: {medication.times.join(', ')}
                              </span>
                            </div>
                          )}

                          {medication.frequency === 'as_needed' && (
                            <div className="flex items-center gap-2">
                              <Clock className="h-4 w-4" />
                              <span className="italic">
                                Take as needed (no scheduled times)
                              </span>
                            </div>
                          )}

                          <div className="flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            <span>
                              Start: {format(new Date(medication.start_date), 'MMM d, yyyy')}
                              {medication.end_date &&
                                ` - End: ${format(new Date(medication.end_date), 'MMM d, yyyy')}`}
                            </span>
                          </div>

                          {medication.instructions && (
                            <div className="mt-2 p-2 bg-muted rounded-md">
                              <p className="text-sm">{medication.instructions}</p>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex gap-2 ml-4">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleEdit(medication)}
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDelete(medication.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <MedicineScheduleDialog
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        medication={selectedMedication}
        patientId={patientId}
        caregiverId={caregiverId}
        onSave={handleSave}
      />
    </>
  );
}
