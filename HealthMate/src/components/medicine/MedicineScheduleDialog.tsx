import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  MedicationSchedule,
  MedicationScheduleFormData,
  MedicationFrequency,
} from '@/types/medication';
import { format } from 'date-fns';
import { Plus, X } from 'lucide-react';

interface MedicineScheduleDialogProps {
  isOpen: boolean;
  onClose: () => void;
  medication?: MedicationSchedule;
  patientId: string;
  caregiverId: string;
  onSave: (
    medData: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'>
  ) => Promise<void>;
}

export function MedicineScheduleDialog({
  isOpen,
  onClose,
  medication,
  patientId,
  caregiverId,
  onSave,
}: MedicineScheduleDialogProps) {
  const [formData, setFormData] = useState<MedicationScheduleFormData>({
    name: '',
    dosage: '',
    frequency: 'daily',
    times: ['09:00'],
    instructions: '',
    start_date: new Date(),
    end_date: undefined,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (medication) {
      setFormData({
        name: medication.name,
        dosage: medication.dosage,
        frequency: medication.frequency,
        times: medication.times,
        instructions: medication.instructions || '',
        start_date: new Date(medication.start_date),
        end_date: medication.end_date ? new Date(medication.end_date) : undefined,
      });
    } else {
      setFormData({
        name: '',
        dosage: '',
        frequency: 'daily',
        times: ['09:00'],
        instructions: '',
        start_date: new Date(),
        end_date: undefined,
      });
    }
  }, [medication, isOpen]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Medication name is required';
    }

    if (!formData.dosage.trim()) {
      newErrors.dosage = 'Dosage is required';
    }

    // Times are only required if frequency is not "as_needed"
    if (formData.frequency !== 'as_needed') {
      if (formData.times.length === 0) {
        newErrors.times = 'At least one time is required';
      }

      for (const time of formData.times) {
        if (!/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/.test(time)) {
          newErrors.times = 'All times must be in HH:mm format';
          break;
        }
      }
    }

    if (!formData.start_date) {
      newErrors.start_date = 'Start date is required';
    }

    if (formData.end_date && formData.start_date && formData.end_date < formData.start_date) {
      newErrors.end_date = 'End date must be after start date';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      const medData: Omit<MedicationSchedule, 'id' | 'created_at' | 'updated_at'> = {
        patient_id: patientId,
        caregiver_id: caregiverId,
        name: formData.name.trim(),
        dosage: formData.dosage.trim(),
        frequency: formData.frequency,
        // For "as_needed", use empty array or a placeholder time
        times: formData.frequency === 'as_needed' ? [] : formData.times,
        instructions: formData.instructions?.trim() || undefined,
        start_date: format(formData.start_date, 'yyyy-MM-dd'),
        end_date: formData.end_date ? format(formData.end_date, 'yyyy-MM-dd') : undefined,
        is_active: true,
      };

      await onSave(medData);
      handleClose();
    } catch (error) {
      console.error('Error saving medication schedule:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setFormData({
      name: '',
      dosage: '',
      frequency: 'daily',
      times: ['09:00'],
      instructions: '',
      start_date: new Date(),
      end_date: undefined,
    });
    setErrors({});
    onClose();
  };

  const addTime = () => {
    setFormData({
      ...formData,
      times: [...formData.times, '09:00'],
    });
  };

  const removeTime = (index: number) => {
    setFormData({
      ...formData,
      times: formData.times.filter((_, i) => i !== index),
    });
  };

  const updateTime = (index: number, value: string) => {
    const newTimes = [...formData.times];
    newTimes[index] = value;
    setFormData({
      ...formData,
      times: newTimes,
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {medication ? 'Edit Medication Schedule' : 'Create Medication Schedule'}
          </DialogTitle>
          <DialogDescription>
            {medication
              ? 'Update the medication schedule details below.'
              : 'Fill in the details for the new medication schedule.'}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="name">
                Medication Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="e.g., Aspirin, Metformin"
              />
              {errors.name && <p className="text-sm text-destructive">{errors.name}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="dosage">
                Dosage <span className="text-destructive">*</span>
              </Label>
              <Input
                id="dosage"
                value={formData.dosage}
                onChange={(e) => setFormData({ ...formData, dosage: e.target.value })}
                placeholder="e.g., 10mg, 2 tablets"
              />
              {errors.dosage && <p className="text-sm text-destructive">{errors.dosage}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="frequency">
                Frequency <span className="text-destructive">*</span>
              </Label>
              <Select
                value={formData.frequency}
                onValueChange={(value: MedicationFrequency) =>
                  setFormData({ ...formData, frequency: value })
                }
              >
                <SelectTrigger id="frequency">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="twice_daily">Twice Daily</SelectItem>
                  <SelectItem value="three_times_daily">Three Times Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="as_needed">As Needed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {formData.frequency !== 'as_needed' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>
                    Times <span className="text-destructive">*</span>
                  </Label>
                  <Button type="button" variant="outline" size="sm" onClick={addTime}>
                    <Plus className="h-4 w-4 mr-1" />
                    Add Time
                  </Button>
                </div>
                <div className="space-y-2">
                  {formData.times.map((time, index) => (
                    <div key={index} className="flex gap-2">
                      <Input
                        type="time"
                        value={time}
                        onChange={(e) => updateTime(index, e.target.value)}
                        className="flex-1"
                      />
                      {formData.times.length > 1 && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          onClick={() => removeTime(index)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
                {errors.times && <p className="text-sm text-destructive">{errors.times}</p>}
              </div>
            )}

            {formData.frequency === 'as_needed' && (
              <div className="p-4 bg-muted rounded-md">
                <p className="text-sm text-muted-foreground">
                  <strong>As Needed:</strong> No specific times required. Patient can take this medication when needed.
                </p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="start_date">
                  Start Date <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="start_date"
                  type="date"
                  value={format(formData.start_date, 'yyyy-MM-dd')}
                  onChange={(e) =>
                    setFormData({ ...formData, start_date: new Date(e.target.value) })
                  }
                />
                {errors.start_date && (
                  <p className="text-sm text-destructive">{errors.start_date}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="end_date">End Date (Optional)</Label>
                <Input
                  id="end_date"
                  type="date"
                  value={formData.end_date ? format(formData.end_date, 'yyyy-MM-dd') : ''}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      end_date: e.target.value ? new Date(e.target.value) : undefined,
                    })
                  }
                />
                {errors.end_date && <p className="text-sm text-destructive">{errors.end_date}</p>}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="instructions">Instructions</Label>
              <Textarea
                id="instructions"
                value={formData.instructions}
                onChange={(e) => setFormData({ ...formData, instructions: e.target.value })}
                placeholder="e.g., Take with food, Avoid alcohol"
                rows={3}
              />
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Saving...' : medication ? 'Update' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
