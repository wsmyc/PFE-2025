
import React from 'react';
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DialogFooter } from "@/components/ui/dialog";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { format } from "date-fns";
import { fr } from "date-fns/locale";
import { CalendarIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Textarea } from "@/components/ui/textarea";

interface ReservationFormData {
  customerName?: string;
  phone?: string;
  people?: number;
  date?: Date;
  time?: string;
  notes?: string;
}

interface ReservationFormProps {
  formData: ReservationFormData;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
  handleDateChange: (date: Date | undefined) => void;
  handleTimeChange: (value: string) => void;
  handlePeopleChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  submitLabel: string;
}

const timeSlots = [
  "12:00", "12:30", "13:00", "13:30", 
  "19:00", "19:30", "20:00", "20:30", "21:00", "21:30"
];

const ReservationForm = ({
  formData,
  handleInputChange,
  handleDateChange,
  handleTimeChange,
  handlePeopleChange,
  onSubmit,
  onCancel,
  submitLabel
}: ReservationFormProps) => {
  return (
    <div className="grid gap-4 py-4">
      <div className="grid gap-2">
        <Label htmlFor="customerName">Nom du Client</Label>
        <Input
          id="customerName"
          name="customerName"
          value={formData.customerName || ""}
          onChange={handleInputChange}
          className="input-field"
        />
      </div>
      
      <div className="grid gap-2">
        <Label htmlFor="phone">Téléphone</Label>
        <Input
          id="phone"
          name="phone"
          value={formData.phone || ""}
          onChange={handleInputChange}
          className="input-field"
        />
      </div>
      
      <div className="grid gap-2">
        <Label htmlFor="people">Nombre de Personnes</Label>
        <Select 
          value={formData.people?.toString() || "default"} 
          onValueChange={handlePeopleChange}
        >
          <SelectTrigger className="input-field">
            <SelectValue placeholder="Sélectionner" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default" disabled>Sélectionner</SelectItem>
            {[1, 2, 3, 4, 5, 6, 7, 8, 10, 12].map(num => (
              <SelectItem key={num} value={num.toString()}>{num}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      
      <div className="grid gap-2">
        <Label>Date</Label>
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              className={cn(
                "w-full justify-start text-left font-normal",
                !formData.date && "text-muted-foreground"
              )}
            >
              <CalendarIcon className="mr-2 h-4 w-4" />
              {formData.date ? (
                format(formData.date, "PPP", { locale: fr })
              ) : (
                "Sélectionner une date"
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0">
            <Calendar
              mode="single"
              selected={formData.date}
              onSelect={handleDateChange}
              initialFocus
              locale={fr}
            />
          </PopoverContent>
        </Popover>
      </div>
      
      <div className="grid gap-2">
        <Label htmlFor="time">Heure</Label>
        <Select 
          value={formData.time || "default"} 
          onValueChange={handleTimeChange}
        >
          <SelectTrigger className="input-field">
            <SelectValue placeholder="Sélectionner" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default" disabled>Sélectionner une heure</SelectItem>
            {timeSlots.map(time => (
              <SelectItem key={time} value={time}>{time}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      
      <div className="grid gap-2">
        <Label htmlFor="notes">Notes</Label>
        <Textarea
          id="notes"
          name="notes"
          value={formData.notes || ""}
          onChange={handleInputChange}
          placeholder="Allergies, préférences..."
          className="input-field"
        />
      </div>
      
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Annuler</Button>
        <Button className="bg-restaurant-primary hover:bg-restaurant-primary/90" onClick={onSubmit}>
          {submitLabel}
        </Button>
      </DialogFooter>
    </div>
  );
};

export default ReservationForm;
