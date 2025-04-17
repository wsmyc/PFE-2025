
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Filter } from "lucide-react";

interface OrderFiltersProps {
  searchQuery: string;
  statusFilter: string;
  onSearchChange: (value: string) => void;
  onStatusChange: (value: string) => void;
}

const OrderFilters = ({
  searchQuery,
  statusFilter,
  onSearchChange,
  onStatusChange
}: OrderFiltersProps) => {
  return (
    <div className="flex flex-col sm:flex-row gap-4 mb-6">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
        <Input
          placeholder="Rechercher une commande..."
          className="pl-10 input-field"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>
      <div className="flex items-center gap-2 w-full sm:w-64">
        <Filter size={18} className="text-gray-400" />
        <Select value={statusFilter} onValueChange={onStatusChange}>
          <SelectTrigger className="input-field">
            <SelectValue placeholder="Tous les statuts" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Tous les statuts</SelectItem>
            <SelectItem value="En attente">En attente</SelectItem>
            <SelectItem value="En préparation">En préparation</SelectItem>
            <SelectItem value="Prêt">Prêt</SelectItem>
            <SelectItem value="Livré">Livré</SelectItem>
            <SelectItem value="Annulé">Annulé</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};

export default OrderFilters;
