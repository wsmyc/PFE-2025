
import { Badge } from "@/components/ui/badge";

type OrderStatus = "En attente" | "En préparation" | "Prêt" | "Livré" | "Annulé";

interface OrderStatusBadgeProps {
  status: OrderStatus;
}

const OrderStatusBadge = ({ status }: OrderStatusBadgeProps) => {
  let color = "";
  switch (status) {
    case "En attente":
      color = "bg-yellow-500 hover:bg-yellow-600";
      break;
    case "En préparation":
      color = "bg-blue-500 hover:bg-blue-600";
      break;
    case "Prêt":
      color = "bg-green-500 hover:bg-green-600";
      break;
    case "Livré":
      color = "bg-gray-500 hover:bg-gray-600";
      break;
    case "Annulé":
      color = "bg-red-500 hover:bg-red-600";
      break;
  }

  return (
    <Badge className={`${color} text-white`}>{status}</Badge>
  );
};

export default OrderStatusBadge;
