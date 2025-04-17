import { 
  PlusCircle, 
  AlertCircle, 
  Utensils, 
  Menu as BurgerIcon,
  ChefHat, 
  Leaf, 
  CupSoda, 
  Cake 
} from "lucide-react";
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import MenuItemForm from "@/components/menu/MenuItemForm";
import MenuItemTable from "@/components/menu/MenuItemTable";
import MenuFilters from "@/components/menu/MenuFilters";

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

const Menu = () => {
  const [menuItems, setMenuItems] = useState<MenuItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [formData, setFormData] = useState<Partial<MenuItem>>({
    name: "",
    description: "",
    price: 0,
    category: "Plats Traditionnels",
    ingredients: ""
  });
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [currentItemId, setCurrentItemId] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all-categories");

  useEffect(() => {
    const fetchMenuItems = async () => {
      setIsLoading(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const mockMenuItems: MenuItem[] = [
          // Entrées
          {
            id: "101",
            name: "Soupe à l'Oignon",
            description: "Soupe traditionnelle aux oignons caramélisés, gratinée au fromage comté",
            price: 7.50,
            category: "Entrées",
            ingredients: "Oignons, bouillon de bœuf, comté, pain baguette"
          },
          {
            id: "102",
            name: "Salade Niçoise",
            description: "Salade méditerranéenne avec thon, olives, œufs, et haricots verts",
            price: 9.00,
            category: "Entrées",
            ingredients: "Thon, olives noires, œufs, tomates, vinaigrette"
          },
          {
            id: "103",
            name: "Tartare de Saumon",
            description: "Saumon frais coupé au couteau, citron, aneth, et capres",
            price: 12.00,
            category: "Entrées",
            ingredients: "Saumon, aneth, citron, capres, toast"
          },
          {
            id: "104",
            name: "Quiche Lorraine",
            description: "Tarte salée avec lardons, œufs, et crème fraîche",
            price: 8.50,
            category: "Entrées",
            ingredients: "Lardons, œufs, crème, pâte brisée"
          },
          {
            id: "105",
            name: "Terrine de Campagne",
            description: "Terrine de porc et foie de volaille, cornichons, et pain de seigle",
            price: 8.00,
            category: "Entrées",
            ingredients: "Porc, foie de volaille, épices"
          },
          {
            id: "106",
            name: "Soupe de Potiron",
            description: "Velouté de potiron avec une touche de crème fraîche",
            price: 7.00,
            category: "Entrées",
            ingredients: "Potiron, crème fraîche, noix de muscade"
          },
          {
            id: "107",
            name: "Bruschetta Tomate-Mozzarella",
            description: "Pain grillé garni de tomates fraîches et mozzarella",
            price: 8.50,
            category: "Entrées",
            ingredients: "Tomates, mozzarella, basilic, huile d'olive"
          },
          {
            id: "108",
            name: "Carpaccio de Bœuf",
            description: "Tranches fines de bœuf marinées à l'huile d'olive et parmesan",
            price: 14.00,
            category: "Entrées",
            ingredients: "Bœuf, parmesan, roquette, huile d'olive"
          },
          {
            id: "109",
            name: "Rillettes de Thon",
            description: "Rillettes maison à base de thon et crème fraîche",
            price: 9.50,
            category: "Entrées",
            ingredients: "Thon, crème fraîche, citron, ciboulette"
          },
          {
            id: "110",
            name: "Soupe de Poisson",
            description: "Soupe méditerranéenne aux poissons et rouille",
            price: 10.00,
            category: "Entrées",
            ingredients: "Poissons, tomates, safran, rouille"
          },
          {
            id: "111",
            name: "Salade de Chèvre Chaud",
            description: "Salade verte avec fromage de chèvre grillé sur toast",
            price: 10.50,
            category: "Entrées",
            ingredients: "Chèvre, salade, noix, vinaigrette"
          },
          {
            id: "112",
            name: "Oeufs Mimosa",
            description: "Oeufs durs garnis de mayonnaise et jaune d'oeuf",
            price: 7.50,
            category: "Entrées",
            ingredients: "Oeufs, mayonnaise, ciboulette"
          },
          {
            id: "113",
            name: "Feuilleté aux Champignons",
            description: "Feuilleté croustillant garni de champignons à la crème",
            price: 9.00,
            category: "Entrées",
            ingredients: "Champignons, crème, pâte feuilletée"
          },
          {
            id: "114",
            name: "Avocat Crevettes",
            description: "Demi-avocat garni de crevettes et sauce cocktail",
            price: 12.50,
            category: "Entrées",
            ingredients: "Avocat, crevettes, sauce cocktail, citron"
          },
          {
            id: "115",
            name: "Salade César",
            description: "Salade romaine, croûtons, parmesan et sauce césar",
            price: 11.00,
            category: "Entrées",
            ingredients: "Poulet, parmesan, croûtons, sauce césar"
          },
          
          // Burgers & Sandwichs
          {
            id: "201",
            name: "Burger Bistrot",
            description: "Pain brioché, steak haché charolais, oignons confits, roquefort",
            price: 15.00,
            category: "Burgers & Sandwichs",
            ingredients: "Bœuf, roquefort, oignons, salade"
          },
          {
            id: "202",
            name: "Croque-Monsieur",
            description: "Classique jambon et fromage gratiné",
            price: 10.00,
            category: "Burgers & Sandwichs",
            ingredients: "Jambon, emmental, pain de mie, béchamel"
          },
          {
            id: "203",
            name: "Panini Poulet-Ratatouille",
            description: "Poulet grillé, ratatouille maison, et fromage de chèvre",
            price: 12.50,
            category: "Burgers & Sandwichs",
            ingredients: "Poulet, courgettes, aubergines, tomates"
          },
          {
            id: "204",
            name: "Burger Végétarien",
            description: "Steak de légumes, avocat, et sauce yaourt",
            price: 13.50,
            category: "Burgers & Sandwichs",
            ingredients: "Légumes, avocat, yaourt, pain brioché"
          },
          {
            id: "205",
            name: "Baguette Jambon-Beurre",
            description: "Baguette traditionnelle avec jambon et beurre",
            price: 8.00,
            category: "Burgers & Sandwichs",
            ingredients: "Jambon, beurre, baguette"
          },
          {
            id: "206",
            name: "Burger BBQ",
            description: "Steak haché, sauce barbecue, oignons frits",
            price: 16.00,
            category: "Burgers & Sandwichs",
            ingredients: "Bœuf, sauce BBQ, oignons, cheddar"
          },
          {
            id: "207",
            name: "Panini Saumon-Avocat",
            description: "Saumon fumé, avocat, et fromage frais",
            price: 14.00,
            category: "Burgers & Sandwichs",
            ingredients: "Saumon, avocat, fromage frais, pain panini"
          },
          {
            id: "208",
            name: "Club Sandwich",
            description: "Poulet, bacon, laitue, tomate, et mayonnaise",
            price: 12.00,
            category: "Burgers & Sandwichs",
            ingredients: "Poulet, bacon, tomate, pain de mie"
          },
          {
            id: "209",
            name: "Wrap Poulet-Crudités",
            description: "Poulet grillé, crudités, et sauce fromagère",
            price: 11.50,
            category: "Burgers & Sandwichs",
            ingredients: "Poulet, crudités, tortilla, sauce"
          },
          {
            id: "210",
            name: "Burger Chevre-Miel",
            description: "Steak haché, fromage de chèvre, et miel",
            price: 15.50,
            category: "Burgers & Sandwichs",
            ingredients: "Bœuf, chèvre, miel, salade"
          },
          
          // Plats Traditionnels
          {
            id: "301",
            name: "Coq au Vin",
            description: "Cuisse de poulet mijotée au vin rouge (sans alcool), champignons",
            price: 16.00,
            category: "Plats Traditionnels",
            ingredients: "Poulet, carottes, oignons, bouillon"
          },
          {
            id: "302",
            name: "Boeuf Bourguignon",
            description: "Bœuf mijoté aux carottes et oignons, sauce riche",
            price: 17.50,
            category: "Plats Traditionnels",
            ingredients: "Bœuf, lardons, champignons, bouillon"
          },
          {
            id: "303",
            name: "Ratatouille Provençale",
            description: "Légumes du soleil rôtis au thym et à l'huile d'olive",
            price: 13.00,
            category: "Plats Traditionnels",
            ingredients: "Aubergines, courgettes, poivrons, tomates"
          },
          {
            id: "304",
            name: "Quiche aux Poireaux",
            description: "Tarte aux poireaux et fromage de chèvre",
            price: 11.00,
            category: "Plats Traditionnels",
            ingredients: "Poireaux, chèvre, œufs, crème"
          },
          {
            id: "305",
            name: "Daurade Grillée",
            description: "Filet de daurade, légumes grillés, et sauce vierge",
            price: 18.00,
            category: "Plats Traditionnels",
            ingredients: "Daurade, citron, courgettes, tomates cerises"
          },
          {
            id: "306",
            name: "Blanquette de Veau",
            description: "Veau mijoté à la crème et aux champignons",
            price: 19.00,
            category: "Plats Traditionnels",
            ingredients: "Veau, champignons, crème, carottes"
          },
          {
            id: "307",
            name: "Confit de Canard",
            description: "Cuisse de canard confite, pommes de terre sautées",
            price: 18.50,
            category: "Plats Traditionnels",
            ingredients: "Canard, pommes de terre, ail"
          },
          {
            id: "308",
            name: "Pot-au-Feu",
            description: "Bœuf et légumes mijotés, bouillon parfumé",
            price: 17.00,
            category: "Plats Traditionnels",
            ingredients: "Bœuf, carottes, poireaux, navets"
          },
          {
            id: "309",
            name: "Lapin à la Moutarde",
            description: "Lapin mijoté à la moutarde à l'ancienne",
            price: 16.50,
            category: "Plats Traditionnels",
            ingredients: "Lapin, moutarde, crème, échalotes"
          },
          {
            id: "310",
            name: "Andouillette Grillée",
            description: "Andouillette grillée, sauce moutarde",
            price: 15.00,
            category: "Plats Traditionnels",
            ingredients: "Andouillette, moutarde, pommes de terre"
          },
          {
            id: "311",
            name: "Filet Mignon",
            description: "Filet mignon de porc, sauce aux champignons",
            price: 20.00,
            category: "Plats Traditionnels",
            ingredients: "Porc, champignons, crème, échalotes"
          },
          {
            id: "312",
            name: "Saumon en Papillote",
            description: "Saumon cuit en papillote avec légumes",
            price: 19.50,
            category: "Plats Traditionnels",
            ingredients: "Saumon, courgettes, citron, aneth"
          },
          {
            id: "313",
            name: "Poulet Rôti",
            description: "Poulet rôti avec jus de cuisson",
            price: 15.50,
            category: "Plats Traditionnels",
            ingredients: "Poulet, thym, ail, jus de cuisson"
          },
          {
            id: "314",
            name: "Gigot d'Agneau",
            description: "Gigot d'agneau rôti, flageolets",
            price: 22.00,
            category: "Plats Traditionnels",
            ingredients: "Agneau, flageolets, romarin"
          },
          {
            id: "315",
            name: "Cassoulet",
            description: "Haricots blancs, saucisse, et confit de canard",
            price: 16.50,
            category: "Plats Traditionnels",
            ingredients: "Haricots, saucisse, canard, tomates"
          },
          
          // Options Végétariennes
          {
            id: "401",
            name: "Gratin de Légumes",
            description: "Courgettes, aubergines, et tomates gratinées au fromage",
            price: 12.00,
            category: "Options Végétariennes",
            ingredients: "Légumes de saison, béchamel, emmental"
          },
          {
            id: "402",
            name: "Galette de Lentilles",
            description: "Galette de lentilles corail, sauce yaourt-citron",
            price: 11.50,
            category: "Options Végétariennes",
            ingredients: "Lentilles, carottes, oignons, épices"
          },
          {
            id: "403",
            name: "Risotto aux Champignons",
            description: "Risotto crémeux aux champignons sauvages",
            price: 14.00,
            category: "Options Végétariennes",
            ingredients: "Riz, champignons, parmesan, crème"
          },
          {
            id: "404",
            name: "Tarte aux Legumes",
            description: "Tarte aux légumes de saison et fromage de chèvre",
            price: 12.50,
            category: "Options Végétariennes",
            ingredients: "Légumes, chèvre, pâte brisée"
          },
          {
            id: "405",
            name: "Falafel",
            description: "Boulettes de pois chiches, sauce tahini",
            price: 10.50,
            category: "Options Végétariennes",
            ingredients: "Pois chiches, tahini, persil"
          },
          {
            id: "406",
            name: "Lasagnes Végétariennes",
            description: "Lasagnes aux légumes et béchamel",
            price: 13.50,
            category: "Options Végétariennes",
            ingredients: "Courgettes, aubergines, tomates, béchamel"
          },
          {
            id: "407",
            name: "Curry de Légumes",
            description: "Curry doux aux légumes et lait de coco",
            price: 12.00,
            category: "Options Végétariennes",
            ingredients: "Légumes, lait de coco, curry"
          },
          {
            id: "408",
            name: "Pâtes aux Artichauts",
            description: "Pâtes fraîches aux artichauts et parmesan",
            price: 11.50,
            category: "Options Végétariennes",
            ingredients: "Pâtes, artichauts, parmesan, crème"
          },
          {
            id: "409",
            name: "Soupe de Lentilles",
            description: "Soupe épicée aux lentilles et légumes",
            price: 9.00,
            category: "Options Végétariennes",
            ingredients: "Lentilles, carottes, oignons, épices"
          },
          {
            id: "410",
            name: "Quinoa aux Legumes",
            description: "Quinoa, légumes grillés, et vinaigrette citronnée",
            price: 12.50,
            category: "Options Végétariennes",
            ingredients: "Quinoa, courgettes, poivrons, citron"
          },
          
          // Accompagnements
          {
            id: "501",
            name: "Frites Maison",
            description: "Coupées à la main, sel de Guérande",
            price: 5.00,
            category: "Accompagnements",
            ingredients: "Pommes de terre, huile d'arachide"
          },
          {
            id: "502",
            name: "Gratin Dauphinois",
            description: "Pommes de terre, crème fraîche, et noix de muscade",
            price: 6.50,
            category: "Accompagnements",
            ingredients: "Pommes de terre, crème, muscade"
          },
          {
            id: "503",
            name: "Légumes Rôtis",
            description: "Courgettes, carottes, et poivrons rôtis à l'huile d'olive",
            price: 5.50,
            category: "Accompagnements",
            ingredients: "Courgettes, carottes, poivrons, huile d'olive"
          },
          {
            id: "504",
            name: "Purée de Céleri-Rave",
            description: "Purée onctueuse au céleri-rave et crème fraîche",
            price: 4.50,
            category: "Accompagnements",
            ingredients: "Céleri-rave, crème, beurre"
          },
          {
            id: "505",
            name: "Quinoa aux Herbes",
            description: "Quinoa, persil, coriandre, et tomates séchées",
            price: 5.00,
            category: "Accompagnements",
            ingredients: "Quinoa, persil, coriandre, tomates"
          },
          {
            id: "506",
            name: "Riz Basmati",
            description: "Riz basmati parfumé au beurre",
            price: 4.00,
            category: "Accompagnements",
            ingredients: "Riz, beurre"
          },
          {
            id: "507",
            name: "Pommes de Terre Sautées",
            description: "Pommes de terre sautées à l'ail et au persil",
            price: 5.50,
            category: "Accompagnements",
            ingredients: "Pommes de terre, ail, persil"
          },
          {
            id: "508",
            name: "Haricots Verts",
            description: "Haricots verts à l'ail",
            price: 4.50,
            category: "Accompagnements",
            ingredients: "Haricots verts, ail, beurre"
          },
          {
            id: "509",
            name: "Polenta",
            description: "Polenta crémeuse au parmesan",
            price: 5.00,
            category: "Accompagnements",
            ingredients: "Polenta, parmesan, crème"
          },
          {
            id: "510",
            name: "Petits Pois Carottes",
            description: "Petits pois et carottes à la française",
            price: 4.50,
            category: "Accompagnements",
            ingredients: "Petits pois, carottes, oignons"
          },
          
          // Boissons Chaudes
          {
            id: "601",
            name: "Café Allongé",
            description: "Café filtre servi dans une grande tasse",
            price: 3.00,
            category: "Boissons Chaudes",
            ingredients: "Café arabica"
          },
          {
            id: "602",
            name: "Thé à la Menthe Fraîche",
            description: "Thé vert à la menthe fraîche et miel",
            price: 4.00,
            category: "Boissons Chaudes",
            ingredients: "Thé vert, menthe, miel"
          },
          {
            id: "603",
            name: "Chocolat Viennois",
            description: "Chocolat chaud épais avec chantilly et copeaux de chocolat",
            price: 5.50,
            category: "Boissons Chaudes",
            ingredients: "Chocolat noir, crème, sucre"
          },
          {
            id: "604",
            name: "Infusion Verveine-Citron",
            description: "Verveine et zeste de citron bio",
            price: 4.00,
            category: "Boissons Chaudes",
            ingredients: "Verveine, citron"
          },
          {
            id: "605",
            name: "Café Noisette",
            description: "Expresso avec une touche de lait mousseux",
            price: 3.50,
            category: "Boissons Chaudes",
            ingredients: "Café, lait"
          },
          {
            id: "606",
            name: "Cappuccino",
            description: "Expresso, lait mousseux, et cacao",
            price: 4.50,
            category: "Boissons Chaudes",
            ingredients: "Café, lait, cacao"
          },
          {
            id: "607",
            name: "Thé Noir",
            description: "Thé noir de Ceylan",
            price: 3.50,
            category: "Boissons Chaudes",
            ingredients: "Thé noir"
          },
          {
            id: "608",
            name: "Latte Macchiato",
            description: "Lait mousseux avec un trait d'expresso",
            price: 5.00,
            category: "Boissons Chaudes",
            ingredients: "Lait, café"
          },
          {
            id: "609",
            name: "Chocolat à l'Orange",
            description: "Chocolat chaud parfumé à l'orange",
            price: 5.50,
            category: "Boissons Chaudes",
            ingredients: "Chocolat, orange, crème"
          },
          {
            id: "610",
            name: "Thé Chai",
            description: "Thé noir épicé au lait",
            price: 4.50,
            category: "Boissons Chaudes",
            ingredients: "Thé, épices, lait"
          },
          
          // Boissons Froides
          {
            id: "701",
            name: "Jus d'Orange Pressé",
            description: "Pressé quotidiennement",
            price: 5.00,
            category: "Boissons Froides",
            ingredients: "Oranges fraîches"
          },
          {
            id: "702",
            name: "Limonade Maison",
            description: "Citron, sucre de canne, et eau gazeuse",
            price: 4.50,
            category: "Boissons Froides",
            ingredients: "Citron, sucre, eau"
          },
          {
            id: "703",
            name: "Eau Infusée (Citron/Gingembre)",
            description: "Eau fraîche infusée au citron et gingembre",
            price: 3.50,
            category: "Boissons Froides",
            ingredients: "Citron, gingembre, eau"
          },
          {
            id: "704",
            name: "Smoothie Tropical",
            description: "Mangue, ananas, et banane mixés avec yaourt nature",
            price: 6.00,
            category: "Boissons Froides",
            ingredients: "Mangue, ananas, banane, yaourt"
          },
          {
            id: "705",
            name: "Jus de Pomme",
            description: "Jus de pomme pressé à froid",
            price: 4.50,
            category: "Boissons Froides",
            ingredients: "Pommes"
          },
          {
            id: "706",
            name: "Coca-Cola",
            description: "Soda classique",
            price: 3.50,
            category: "Boissons Froides",
            ingredients: "Eau gazéifiée, sucre, arômes"
          },
          {
            id: "707",
            name: "Perrier",
            description: "Eau minérale gazeuse",
            price: 3.00,
            category: "Boissons Froides",
            ingredients: "Eau minérale"
          },
          {
            id: "708",
            name: "Ice Tea",
            description: "Thé glacé à la pêche",
            price: 4.00,
            category: "Boissons Froides",
            ingredients: "Thé, pêche, sucre"
          },
          {
            id: "709",
            name: "Jus de Carotte",
            description: "Jus de carotte frais",
            price: 5.00,
            category: "Boissons Froides",
            ingredients: "Carottes"
          },
          {
            id: "710",
            name: "Diabolo Menthe",
            description: "Limonade à la menthe",
            price: 4.00,
            category: "Boissons Froides",
            ingredients: "Limonade, sirop de menthe"
          },
          
          // Desserts
          {
            id: "801",
            name: "Crème Brûlée",
            description: "Crème vanille caramélisée à la torche",
            price: 6.50,
            category: "Desserts",
            ingredients: "Crème fraîche, vanille, sucre, œufs"
          },
          {
            id: "802",
            name: "Tarte Tatin",
            description: "Tarte renversée aux pommes caramélisées",
            price: 7.00,
            category: "Desserts",
            ingredients: "Pommes, sucre, pâte feuilletée"
          },
          {
            id: "803",
            name: "Mousse au Chocolat Noir",
            description: "Mousse 70% cacao, chantilly légère",
            price: 6.00,
            category: "Desserts",
            ingredients: "Chocolat noir, œufs, crème"
          },
          {
            id: "804",
            name: "Île Flottante",
            description: "Meringue légère sur crème anglaise vanillée",
            price: 6.50,
            category: "Desserts",
            ingredients: "Œufs, lait, vanille"
          },
          {
            id: "805",
            name: "Salade de Fruits Frais",
            description: "Fruits de saison coupés au couteau (melon, fraises, kiwi)",
            price: 5.50,
            category: "Desserts",
            ingredients: "Melon, fraises, kiwi"
          },
          {
            id: "806",
            name: "Sorbet Citron/Framboise",
            description: "Sorbet artisanal au choix",
            price: 5.00,
            category: "Desserts",
            ingredients: "Citron ou framboise, eau, sucre"
          },
          {
            id: "807",
            name: "Profiteroles",
            description: "Choux garnis de glace vanille et sauce chocolat",
            price: 7.50,
            category: "Desserts",
            ingredients: "Choux, glace, chocolat"
          },
          {
            id: "808",
            name: "Fondant au Chocolat",
            description: "Gâteau moelleux avec cœur coulant",
            price: 7.00,
            category: "Desserts",
            ingredients: "Chocolat, beurre, œufs"
          },
          {
            id: "809",
            name: "Tiramisu",
            description: "Dessert italien au café et mascarpone",
            price: 6.50,
            category: "Desserts",
            ingredients: "Mascarpone, café, biscuits"
          },
          {
            id: "810",
            name: "Clafoutis aux Cerises",
            description: "Clafoutis traditionnel aux cerises",
            price: 6.00,
            category: "Desserts",
            ingredients: "Cerises, lait, œufs, farine"
          }
        ];
        setMenuItems(mockMenuItems);
      } catch (error) {
        console.error("Erreur lors du chargement du menu:", error);
        toast.error("Impossible de charger les articles du menu");
      } finally {
        setIsLoading(false);
      }
    };

    fetchMenuItems();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    let parsedValue: string | number = value;
    
    if (name === "price") {
      parsedValue = parseFloat(value) || 0;
    }
    
    setFormData({
      ...formData,
      [name]: parsedValue
    });
  };

  const handleCategoryChange = (value: string) => {
    setFormData({
      ...formData,
      category: value as MenuItem["category"]
    });
  };

  const resetForm = () => {
    setFormData({
      name: "",
      description: "",
      price: 0,
      category: "Plats Traditionnels",
      ingredients: ""
    });
  };

  const handleAddItem = () => {
    if (!formData.name || !formData.description || formData.price === undefined || !formData.category) {
      toast.error("Veuillez remplir tous les champs");
      return;
    }

    const newItem: MenuItem = {
      id: `item-${Date.now()}`,
      name: formData.name,
      description: formData.description,
      price: formData.price,
      category: formData.category as MenuItem["category"],
      ingredients: formData.ingredients
    };

    setMenuItems([...menuItems, newItem]);
    toast.success("Article ajouté au menu");
    setIsAddDialogOpen(false);
    resetForm();
  };

  const handleEditItem = () => {
    if (!formData.name || !formData.description || formData.price === undefined || !formData.category) {
      toast.error("Veuillez remplir tous les champs");
      return;
    }

    setMenuItems(prevItems => 
      prevItems.map(item => 
        item.id === currentItemId 
          ? { 
              ...item, 
              name: formData.name!, 
              description: formData.description!, 
              price: formData.price!, 
              category: formData.category as MenuItem["category"],
              ingredients: formData.ingredients 
            } 
          : item
      )
    );
    
    toast.success("Article mis à jour");
    setIsEditDialogOpen(false);
    resetForm();
  };

  const handleDeleteItem = () => {
    setMenuItems(prevItems => prevItems.filter(item => item.id !== currentItemId));
    toast.success("Article supprimé");
    setIsDeleteDialogOpen(false);
  };

  const openEditDialog = (item: MenuItem) => {
    setCurrentItemId(item.id);
    setFormData({
      name: item.name,
      description: item.description,
      price: item.price,
      category: item.category,
      ingredients: item.ingredients
    });
    setIsEditDialogOpen(true);
  };

  const openDeleteDialog = (itemId: string) => {
    setCurrentItemId(itemId);
    setIsDeleteDialogOpen(true);
  };

  const filteredMenuItems = menuItems.filter(item => {
    const matchesSearch = 
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = categoryFilter === "all-categories" || item.category === categoryFilter;
    
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="card-dashboard">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-xl font-semibold text-restaurant-primary">Menu</CardTitle>
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-restaurant-primary hover:bg-restaurant-primary/90">
                <PlusCircle className="mr-2 h-4 w-4" />
                Ajouter un Article
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Ajouter un Nouvel Article</DialogTitle>
                <DialogDescription>
                  Ajoutez un nouvel article à votre menu. Cliquez sur sauvegarder lorsque vous avez terminé.
                </DialogDescription>
              </DialogHeader>
              <MenuItemForm
                formData={formData}
                handleInputChange={handleInputChange}
                handleCategoryChange={handleCategoryChange}
                onSubmit={handleAddItem}
                onCancel={() => setIsAddDialogOpen(false)}
                submitLabel="Ajouter"
              />
            </DialogContent>
          </Dialog>
        </CardHeader>
        <CardContent>
          <MenuFilters
            searchQuery={searchQuery}
            categoryFilter={categoryFilter}
            onSearchChange={setSearchQuery}
            onCategoryChange={setCategoryFilter}
          />

          {isLoading ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-restaurant-accent"></div>
              <p className="mt-2 text-restaurant-primary">Chargement du menu...</p>
            </div>
          ) : filteredMenuItems.length === 0 ? (
            <div className="text-center py-8 flex flex-col items-center">
              <AlertCircle className="h-12 w-12 text-gray-400 mb-2" />
              <h3 className="font-medium text-lg text-restaurant-primary">Aucun article trouvé</h3>
              <p className="text-gray-500 mt-1">Essayez de modifier vos filtres ou votre recherche</p>
            </div>
          ) : (
            <MenuItemTable
              items={filteredMenuItems}
              onEdit={openEditDialog}
              onDelete={openDeleteDialog}
            />
          )}
        </CardContent>
      </Card>

      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Modifier un Article</DialogTitle>
            <DialogDescription>
              Modifiez les détails de l'article de menu. Cliquez sur sauvegarder lorsque vous avez terminé.
            </DialogDescription>
          </DialogHeader>
          <MenuItemForm
            formData={formData}
            handleInputChange={handleInputChange}
            handleCategoryChange={handleCategoryChange}
            onSubmit={handleEditItem}
            onCancel={() => setIsEditDialogOpen(false)}
            submitLabel="Sauvegarder"
          />
        </DialogContent>
      </Dialog>

      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirmation de Suppression</DialogTitle>
            <DialogDescription>
              Êtes-vous sûr de vouloir supprimer cet article du menu ? Cette action ne peut pas être annulée.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end space-x-2 mt-4">
            <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)}>Annuler</Button>
            <Button variant="destructive" onClick={handleDeleteItem}>Supprimer</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Menu;
