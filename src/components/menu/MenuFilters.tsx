
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface MenuFiltersProps {
  searchQuery: string;
  categoryFilter: string;
  onSearchChange: (value: string) => void;
  onCategoryChange: (value: string) => void;
}

const MenuFilters = ({
  searchQuery,
  categoryFilter,
  onSearchChange,
  onCategoryChange
}: MenuFiltersProps) => {
  return (
    <div className="flex flex-col sm:flex-row gap-4 mb-6">
      <div className="relative flex-1">
        <Input
          placeholder="Rechercher un article..."
          className="input-field"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>
      <div className="w-full sm:w-64">
        <Select value={categoryFilter} onValueChange={onCategoryChange}>
          <SelectTrigger className="input-field">
            <SelectValue placeholder="Toutes les catégories" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all-categories">Toutes les catégories</SelectItem>
            <SelectItem value="Entrées">Entrées</SelectItem>
            <SelectItem value="Burgers & Sandwichs">Burgers & Sandwichs</SelectItem>
            <SelectItem value="Plats Traditionnels">Plats Traditionnels</SelectItem>
            <SelectItem value="Options Végétariennes">Options Végétariennes</SelectItem>
            <SelectItem value="Accompagnements">Accompagnements</SelectItem>
            <SelectItem value="Boissons Chaudes">Boissons Chaudes</SelectItem>
            <SelectItem value="Boissons Froides">Boissons Froides</SelectItem>
            <SelectItem value="Desserts">Desserts</SelectItem>
            <SelectItem value="Plats de Viande">Plats de Viande</SelectItem>
            <SelectItem value="Poissons & Fruits de Mer">Poissons & Fruits de Mer</SelectItem>
            <SelectItem value="Pizzas & Tartes">Pizzas & Tartes</SelectItem>
            <SelectItem value="Pâtes">Pâtes</SelectItem>
            <SelectItem value="Salades">Salades</SelectItem>
            <SelectItem value="Plats Rapides">Plats Rapides</SelectItem>
            <SelectItem value="Végétarien">Végétarien</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};

export default MenuFilters;
