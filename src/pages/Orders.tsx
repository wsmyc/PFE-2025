import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";
import { toast } from "sonner";
import OrderStatusBadge from "@/components/orders/OrderStatusBadge";
import OrderFilters from "@/components/orders/OrderFilters";
import OrdersTable from "@/components/orders/OrdersTable";

interface Order {
  id: string;
  customerName: string;
  items: string[];
  total: number;
  status: "En attente" | "En préparation" | "Prêt" | "Livré" | "Annulé";
  time: string;
}

const Orders = () => {
  const [orders, setOrders] = useState<Order[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchOrders = async () => {
      setIsLoading(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const mockedOrders: Order[] = [
          {
            id: "ORD-001",
            customerName: "Jean Dupont",
            items: ["Bœuf Bourguignon", "Crème Brûlée", "Vin Rouge"],
            total: 42.50,
            status: "En préparation",
            time: "14:30"
          },
          {
            id: "ORD-002",
            customerName: "Marie Lefebvre",
            items: ["Salade Niçoise", "Coq au Vin", "Eau Minérale"],
            total: 35.80,
            status: "En attente",
            time: "14:45"
          },
          {
            id: "ORD-003",
            customerName: "Philippe Martin",
            items: ["Quiche Lorraine", "Ratatouille", "Tarte Tatin", "Vin Blanc"],
            total: 53.20,
            status: "Prêt",
            time: "14:15"
          },
          {
            id: "ORD-004",
            customerName: "Sophie Bernard",
            items: ["Cassoulet", "Mousse au Chocolat"],
            total: 28.90,
            status: "Livré",
            time: "13:50"
          },
          {
            id: "ORD-005",
            customerName: "Lucas Petit",
            items: ["Escargots", "Magret de Canard", "Tarte aux Pommes"],
            total: 45.60,
            status: "Annulé",
            time: "14:00"
          },
          {
            id: "ORD-006",
            customerName: "Emma Dubois",
            items: ["Soupe à l'Oignon", "Steak Frites", "Crêpes Suzette"],
            total: 39.70,
            status: "En préparation",
            time: "15:00"
          },
          {
            id: "ORD-007",
            customerName: "Thomas Leroy",
            items: ["Foie Gras", "Blanquette de Veau", "Éclair au Chocolat"],
            total: 58.30,
            status: "En attente",
            time: "15:15"
          },
          {
            id: "ORD-008",
            customerName: "Camille Roux",
            items: ["Salade de Chèvre Chaud", "Bouillabaisse", "Tarte au Citron"],
            total: 47.90,
            status: "En préparation",
            time: "15:30"
          }
        ];
        setOrders(mockedOrders);
      } catch (error) {
        console.error("Erreur lors du chargement des commandes:", error);
        toast.error("Impossible de charger les commandes");
      } finally {
        setIsLoading(false);
      }
    };

    fetchOrders();
  }, []);

  const updateOrderStatus = (orderId: string, newStatus: Order["status"]) => {
    setOrders(prev => 
      prev.map(order => 
        order.id === orderId 
          ? { ...order, status: newStatus } 
          : order
      )
    );
    toast.success(`Commande ${orderId}: Statut mis à jour vers "${newStatus}"`);
  };

  const filteredOrders = orders.filter(order => {
    const matchesSearch = 
      order.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      order.id.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = statusFilter === "all" || order.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="card-dashboard">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-restaurant-primary">Commandes</CardTitle>
        </CardHeader>
        <CardContent>
          <OrderFilters
            searchQuery={searchQuery}
            statusFilter={statusFilter}
            onSearchChange={setSearchQuery}
            onStatusChange={setStatusFilter}
          />

          {isLoading ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-restaurant-accent"></div>
              <p className="mt-2 text-restaurant-primary">Chargement des commandes...</p>
            </div>
          ) : filteredOrders.length === 0 ? (
            <div className="text-center py-8 flex flex-col items-center">
              <AlertCircle className="h-12 w-12 text-gray-400 mb-2" />
              <h3 className="font-medium text-lg text-restaurant-primary">Aucune commande trouvée</h3>
              <p className="text-gray-500 mt-1">Essayez de modifier vos filtres ou votre recherche</p>
            </div>
          ) : (
            <OrdersTable
              orders={filteredOrders}
              onStatusChange={updateOrderStatus}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Orders;
