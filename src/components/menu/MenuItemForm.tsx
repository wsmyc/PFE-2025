
import React from 'react';
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DialogFooter } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";

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

interface MenuItemFormProps {
  formData: Partial<MenuItem>;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
  handleCategoryChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  submitLabel: string;
}

const MenuItemForm = ({
  formData,
  handleInputChange,
  handleCategoryChange,
  onSubmit,
  onCancel,
  submitLabel
}: MenuItemFormProps) => {
  return (
    <div className="grid gap-4 py-4">
      <div className="grid gap-2">
        <Label htmlFor="name">Nom de l'Article</Label>
        <Input
          id="name"
          name="name"
          value={formData.name || ""}
          onChange={handleInputChange}
          className="input-field"
        />
      </div>
      <div className="grid gap-2">
        <Label htmlFor="description">Description</Label>
        <Input
          id="description"
          name="description"
          value={formData.description || ""}
          onChange={handleInputChange}
          className="input-field"
        />
      </div>
      <div className="grid gap-2">
        <Label htmlFor="ingredients">Ingrédients</Label>
        <Textarea
          id="ingredients"
          name="ingredients"
          value={formData.ingredients || ""}
          onChange={handleInputChange}
          className="input-field"
          rows={3}
        />
      </div>
      <div className="grid gap-2">
        <Label htmlFor="price">Prix (€)</Label>
        <Input
          id="price"
          name="price"
          type="number"
          step="0.01"
          value={formData.price !== undefined ? formData.price : ""}
          onChange={handleInputChange}
          className="input-field"
        />
      </div>
      <div className="grid gap-2">
        <Label htmlFor="category">Catégorie</Label>
        <Select 
          value={formData.category || ""} 
          onValueChange={handleCategoryChange}
        >
          <SelectTrigger className="input-field">
            <SelectValue placeholder="Sélectionner une catégorie" />
          </SelectTrigger>
          <SelectContent>
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
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Annuler</Button>
        <Button className="bg-restaurant-primary hover:bg-restaurant-primary/90" onClick={onSubmit}>
          {submitLabel}
        </Button>
      </DialogFooter>
    </div>
  );
};

export default MenuItemForm;
