
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import OrderStatusBadge from "./OrderStatusBadge";

interface Order {
  id: string;
  customerName: string;
  items: string[];
  total: number;
  status: "En attente" | "En préparation" | "Prêt" | "Livré" | "Annulé";
  time: string;
}

interface OrdersTableProps {
  orders: Order[];
  onStatusChange: (orderId: string, newStatus: Order["status"]) => void;
}

const OrdersTable = ({ orders, onStatusChange }: OrdersTableProps) => {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Nom du Client</TableHead>
            <TableHead>Articles</TableHead>
            <TableHead>Total</TableHead>
            <TableHead>Statut</TableHead>
            <TableHead>Heure</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {orders.map((order) => (
            <TableRow key={order.id}>
              <TableCell className="font-medium">{order.id}</TableCell>
              <TableCell>{order.customerName}</TableCell>
              <TableCell>
                <div className="max-w-xs truncate">
                  {order.items.join(", ")}
                </div>
              </TableCell>
              <TableCell>{order.total.toFixed(2)}€</TableCell>
              <TableCell>
                <OrderStatusBadge status={order.status} />
              </TableCell>
              <TableCell>{order.time}</TableCell>
              <TableCell>
                <Select 
                  defaultValue={order.status} 
                  onValueChange={(value) => onStatusChange(order.id, value as Order["status"])}
                >
                  <SelectTrigger className="w-36 h-8 text-xs">
                    <SelectValue placeholder="Changer le statut" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="En attente">En attente</SelectItem>
                    <SelectItem value="En préparation">En préparation</SelectItem>
                    <SelectItem value="Prêt">Prêt</SelectItem>
                    <SelectItem value="Livré">Livré</SelectItem>
                    <SelectItem value="Annulé">Annulé</SelectItem>
                  </SelectContent>
                </Select>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default OrdersTable;
