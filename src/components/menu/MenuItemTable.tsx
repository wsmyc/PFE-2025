
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Pencil, Trash2, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface MenuItem {
  id: string;
  name: string;
  description: string;
  price: number;
  category: 
    | "Entrées"
    | "Burgers & Sandwichs"
    | "Plats Traditionnels"
    | "Options Végétariennes"
    | "Accompagnements"
    | "Boissons Chaudes"
    | "Boissons Froides"
    | "Desserts"
    | "Plats de Viande"
    | "Poissons & Fruits de Mer"
    | "Pizzas & Tartes"
    | "Pâtes"
    | "Salades"
    | "Plats Rapides"
    | "Végétarien";
  ingredients?: string;
}

interface MenuItemTableProps {
  items: MenuItem[];
  onEdit: (item: MenuItem) => void;
  onDelete: (itemId: string) => void;
}

const MenuItemTable = ({ items, onEdit, onDelete }: MenuItemTableProps) => {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Nom de l'Article</TableHead>
            <TableHead>Description</TableHead>
            <TableHead>Prix</TableHead>
            <TableHead>Catégorie</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow key={item.id}>
              <TableCell className="font-medium">{item.name}</TableCell>
              <TableCell className="max-w-xs truncate">
                <div className="flex items-center">
                  <span className="truncate">{item.description}</span>
                  {item.ingredients && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7 ml-1">
                            <Info className="h-4 w-4 text-muted-foreground" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="font-semibold">Ingrédients:</p>
                          <p>{item.ingredients}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
              </TableCell>
              <TableCell>{item.price.toFixed(2)}€</TableCell>
              <TableCell>
                <Badge variant="outline" className="bg-restaurant-primary/10 text-restaurant-primary border-restaurant-primary/20">
                  {item.category}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex space-x-2">
                  <Button 
                    variant="outline" 
                    size="icon" 
                    onClick={() => onEdit(item)}
                    className="h-8 w-8"
                  >
                    <Pencil className="h-4 w-4" />
                  </Button>
                  <Button 
                    variant="outline" 
                    size="icon" 
                    onClick={() => onDelete(item.id)}
                    className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-50"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default MenuItemTable;
