
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { format, isToday, isTomorrow, addDays } from "date-fns";
import { fr } from "date-fns/locale";
import { CalendarIcon, Check, X, Search, AlertCircle, PlusCircle } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import ReservationForm from "@/components/reservations/ReservationForm";

interface Reservation {
  id: string;
  customerName: string;
  people: number;
  date: Date;
  time: string;
  status: "En attente" | "Confirmée" | "Annulée";
  notes?: string;
  phone: string;
}

const ReservationStatusBadge = ({ status }: { status: Reservation["status"] }) => {
  let badgeClass = "";
  
  switch (status) {
    case "En attente":
      badgeClass = "bg-yellow-500 hover:bg-yellow-600";
      break;
    case "Confirmée":
      badgeClass = "bg-green-500 hover:bg-green-600";
      break;
    case "Annulée":
      badgeClass = "bg-red-500 hover:bg-red-600";
      break;
  }
  
  return (
    <Badge className={`${badgeClass} text-white`}>{status}</Badge>
  );
};

const Reservations = () => {
  const [reservations, setReservations] = useState<Reservation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all-statuses");
  const [dateFilter, setDateFilter] = useState<Date | undefined>(undefined);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [formData, setFormData] = useState<Partial<Reservation>>({
    customerName: "",
    phone: "",
    people: 2,
    date: new Date(),
    time: "19:30",
    notes: "",
    status: "En attente"
  });

  useEffect(() => {
    // In a real application, this would fetch from Firebase
    const fetchReservations = async () => {
      setIsLoading(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock data with dates around today
        const today = new Date();
        const tomorrow = addDays(today, 1);
        const dayAfterTomorrow = addDays(today, 2);
        const yesterday = addDays(today, -1);
        
        const mockReservations: Reservation[] = [
          {
            id: "RSV-001",
            customerName: "Jean Dupont",
            people: 4,
            date: today,
            time: "19:30",
            status: "Confirmée",
            phone: "06 12 34 56 78"
          },
          {
            id: "RSV-002",
            customerName: "Marie Lefebvre",
            people: 2,
            date: today,
            time: "20:00",
            status: "En attente",
            notes: "Table près de la fenêtre",
            phone: "06 23 45 67 89"
          },
          {
            id: "RSV-003",
            customerName: "Philippe Martin",
            people: 6,
            date: tomorrow,
            time: "19:00",
            status: "Confirmée",
            phone: "06 34 56 78 90"
          },
          {
            id: "RSV-004",
            customerName: "Sophie Bernard",
            people: 3,
            date: tomorrow,
            time: "20:30",
            status: "En attente",
            phone: "06 45 67 89 01"
          },
          {
            id: "RSV-005",
            customerName: "Lucas Petit",
            people: 2,
            date: dayAfterTomorrow,
            time: "21:00",
            status: "Confirmée",
            notes: "Anniversaire",
            phone: "06 56 78 90 12"
          },
          {
            id: "RSV-006",
            customerName: "Emma Dubois",
            people: 8,
            date: yesterday,
            time: "19:30",
            status: "Annulée",
            phone: "06 67 89 01 23"
          },
          {
            id: "RSV-007",
            customerName: "Thomas Leroy",
            people: 5,
            date: today,
            time: "12:30",
            status: "Confirmée",
            notes: "Allergie aux fruits de mer",
            phone: "06 78 90 12 34"
          },
          {
            id: "RSV-008",
            customerName: "Camille Roux",
            people: 2,
            date: tomorrow,
            time: "13:00",
            status: "En attente",
            phone: "06 89 01 23 45"
          }
        ];
        
        setReservations(mockReservations);
      } catch (error) {
        console.error("Erreur lors du chargement des réservations:", error);
        toast.error("Impossible de charger les réservations");
      } finally {
        setIsLoading(false);
      }
    };

    fetchReservations();
  }, []);

  const updateReservationStatus = (reservationId: string, newStatus: Reservation["status"]) => {
    // In a real application, this would update Firebase
    setReservations(prev => 
      prev.map(reservation => 
        reservation.id === reservationId 
          ? { ...reservation, status: newStatus } 
          : reservation
      )
    );
    
    const statusMessage = 
      newStatus === "Confirmée" ? "confirmée" : 
      newStatus === "Annulée" ? "annulée" : "mise à jour";
    
    toast.success(`Réservation ${reservationId} ${statusMessage}`);
  };

  const filteredReservations = reservations.filter(reservation => {
    const matchesSearch = 
      reservation.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      reservation.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      reservation.phone.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = statusFilter === "all-statuses" || reservation.status === statusFilter;
    
    const matchesDate = !dateFilter || 
      (reservation.date.getDate() === dateFilter.getDate() &&
       reservation.date.getMonth() === dateFilter.getMonth() &&
       reservation.date.getFullYear() === dateFilter.getFullYear());
    
    return matchesSearch && matchesStatus && matchesDate;
  });

  const sortedReservations = [...filteredReservations].sort((a, b) => {
    // Sort by date first
    const dateComparison = a.date.getTime() - b.date.getTime();
    if (dateComparison !== 0) return dateComparison;
    
    // If same date, sort by time
    return a.time.localeCompare(b.time);
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleDateChange = (date: Date | undefined) => {
    setFormData({
      ...formData,
      date
    });
  };

  const handleTimeChange = (value: string) => {
    if (value === "default") return;
    setFormData({
      ...formData,
      time: value
    });
  };

  const handlePeopleChange = (value: string) => {
    if (value === "default") return;
    setFormData({
      ...formData,
      people: parseInt(value)
    });
  };

  const resetForm = () => {
    setFormData({
      customerName: "",
      phone: "",
      people: 2,
      date: new Date(),
      time: "19:30",
      notes: "",
      status: "En attente"
    });
  };

  const handleAddReservation = () => {
    if (!formData.customerName || !formData.phone || !formData.date || !formData.time || !formData.people) {
      toast.error("Veuillez remplir tous les champs obligatoires");
      return;
    }

    const newReservation: Reservation = {
      id: `RSV-${String(reservations.length + 1).padStart(3, '0')}`,
      customerName: formData.customerName,
      phone: formData.phone,
      people: formData.people,
      date: formData.date,
      time: formData.time,
      status: "En attente",
      notes: formData.notes
    };

    setReservations([...reservations, newReservation]);
    toast.success("Réservation ajoutée");
    setIsAddDialogOpen(false);
    resetForm();
  };

  const formatDateLabel = (date: Date) => {
    if (isToday(date)) return "Aujourd'hui";
    if (isTomorrow(date)) return "Demain";
    return format(date, "P", { locale: fr });
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="card-dashboard">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-xl font-semibold text-restaurant-primary">Réservations</CardTitle>
          <Button 
            className="bg-restaurant-primary hover:bg-restaurant-primary/90"
            onClick={() => setIsAddDialogOpen(true)}
          >
            <PlusCircle className="mr-2 h-4 w-4" />
            Nouvelle Réservation
          </Button>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
              <Input
                placeholder="Rechercher une réservation..."
                className="pl-10 input-field"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-full sm:w-40 input-field">
                  <SelectValue placeholder="Tous les statuts" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-statuses">Tous les statuts</SelectItem>
                  <SelectItem value="En attente">En attente</SelectItem>
                  <SelectItem value="Confirmée">Confirmée</SelectItem>
                  <SelectItem value="Annulée">Annulée</SelectItem>
                </SelectContent>
              </Select>

              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full sm:w-52 justify-start text-left font-normal input-field",
                      !dateFilter && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {dateFilter ? (
                      format(dateFilter, "P", { locale: fr })
                    ) : (
                      "Sélectionner une date"
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={dateFilter}
                    onSelect={setDateFilter}
                    locale={fr}
                  />
                </PopoverContent>
              </Popover>

              {dateFilter && (
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={() => setDateFilter(undefined)}
                  className="h-10 w-10"
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {isLoading ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-restaurant-accent"></div>
              <p className="mt-2 text-restaurant-primary">Chargement des réservations...</p>
            </div>
          ) : sortedReservations.length === 0 ? (
            <div className="text-center py-8 flex flex-col items-center">
              <AlertCircle className="h-12 w-12 text-gray-400 mb-2" />
              <h3 className="font-medium text-lg text-restaurant-primary">Aucune réservation trouvée</h3>
              <p className="text-gray-500 mt-1">Essayez de modifier vos filtres ou votre recherche</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Nom du Client</TableHead>
                    <TableHead>Téléphone</TableHead>
                    <TableHead>Personnes</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Heure</TableHead>
                    <TableHead>Statut</TableHead>
                    <TableHead>Notes</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedReservations.map((reservation) => (
                    <TableRow key={reservation.id}>
                      <TableCell className="font-medium">{reservation.id}</TableCell>
                      <TableCell>{reservation.customerName}</TableCell>
                      <TableCell>{reservation.phone}</TableCell>
                      <TableCell>{reservation.people}</TableCell>
                      <TableCell>{formatDateLabel(reservation.date)}</TableCell>
                      <TableCell>{reservation.time}</TableCell>
                      <TableCell>
                        <ReservationStatusBadge status={reservation.status} />
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xs truncate">
                          {reservation.notes || "-"}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex space-x-2">
                          <Button 
                            variant="outline" 
                            size="icon" 
                            onClick={() => updateReservationStatus(reservation.id, "Confirmée")}
                            disabled={reservation.status === "Confirmée"}
                            className="h-8 w-8 text-green-500 hover:text-green-600 hover:bg-green-50"
                          >
                            <Check className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="outline" 
                            size="icon" 
                            onClick={() => updateReservationStatus(reservation.id, "Annulée")}
                            disabled={reservation.status === "Annulée"}
                            className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-50"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add Reservation Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Nouvelle Réservation</DialogTitle>
          </DialogHeader>
          <ReservationForm
            formData={formData}
            handleInputChange={handleInputChange}
            handleDateChange={handleDateChange}
            handleTimeChange={handleTimeChange}
            handlePeopleChange={handlePeopleChange}
            onSubmit={handleAddReservation}
            onCancel={() => {
              setIsAddDialogOpen(false);
              resetForm();
            }}
            submitLabel="Ajouter"
          />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Reservations;
