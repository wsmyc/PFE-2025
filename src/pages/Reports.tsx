import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { FileDown, FileText, Calendar, CircleDollarSign, PieChart as PieChartIcon } from "lucide-react";
import { toast } from "sonner";

// Sample data for reports
const weeklyRevenueData = [
  { name: 'Lun', revenue: 1380 },
  { name: 'Mar', revenue: 1520 },
  { name: 'Mer', revenue: 1650 },
  { name: 'Jeu', revenue: 1800 },
  { name: 'Ven', revenue: 2540 },
  { name: 'Sam', revenue: 2820 },
  { name: 'Dim', revenue: 1870 }
];

const monthlyRevenueData = [
  { name: 'Jan', revenue: 42000 },
  { name: 'Fév', revenue: 38000 },
  { name: 'Mar', revenue: 45000 },
  { name: 'Avr', revenue: 40000 },
  { name: 'Mai', revenue: 52000 },
  { name: 'Juin', revenue: 58000 },
  { name: 'Juil', revenue: 65000 },
  { name: 'Aoû', revenue: 68000 },
  { name: 'Sep', revenue: 56000 },
  { name: 'Oct', revenue: 49000 },
  { name: 'Nov', revenue: 47000 },
  { name: 'Déc', revenue: 55000 }
];

const salesByCategoryData = [
  { name: 'Entrées', value: 10 },
  { name: 'Burgers & Sandwichs', value: 15 },
  { name: 'Plats Traditionnels', value: 25 },
  { name: 'Options Végétariennes', value: 12 },
  { name: 'Accompagnements', value: 8 },
  { name: 'Boissons Chaudes', value: 7 },
  { name: 'Boissons Froides', value: 13 },
  { name: 'Desserts', value: 10 }
];

const COLORS = [
  '#245536', '#ba3400', '#db9051', '#e9b975', 
  '#5c8d76', '#f06e28', '#94c6ac', '#ffd166'
];

const popularDishesData = [
  { name: 'Bœuf Bourguignon', orders: 68 },
  { name: 'Coq au Vin', orders: 52 },
  { name: 'Crème Brûlée', orders: 45 },
  { name: 'Salade Niçoise', orders: 40 },
  { name: 'Soupe à l\'Oignon', orders: 38 }
];

const Reports = () => {
  const [reportType, setReportType] = useState("weekly");
  const [reportCategory, setReportCategory] = useState("revenue");
  
  const handleGeneratePDF = () => {
    toast.success("Génération du PDF en cours...");
    setTimeout(() => {
      toast.success("PDF généré avec succès");
    }, 2000);
  };

  const calculateTotalRevenue = (data: any[]) => {
    return data.reduce((sum, item) => sum + item.revenue, 0);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="card-dashboard">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-restaurant-primary">Rapports</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="flex-1">
              <Tabs defaultValue="revenue" onValueChange={setReportCategory}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="revenue">
                    <CircleDollarSign className="mr-2 h-4 w-4" />
                    Revenus
                  </TabsTrigger>
                  <TabsTrigger value="dishes">
                    <PieChartIcon className="mr-2 h-4 w-4" />
                    Plats
                  </TabsTrigger>
                  <TabsTrigger value="categories">
                    <FileText className="mr-2 h-4 w-4" />
                    Catégories
                  </TabsTrigger>
                </TabsList>
              
                <TabsContent value="revenue">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-medium flex items-center gap-2">
                        <CircleDollarSign className="h-5 w-5 text-restaurant-accent" />
                        {reportType === "weekly" 
                          ? "Revenus Hebdomadaires" 
                          : reportType === "monthly" 
                          ? "Revenus Mensuels" 
                          : "Revenus Annuels"}
                      </CardTitle>
                      <div className="text-sm text-muted-foreground">
                        Total: {calculateTotalRevenue(reportType === "weekly" ? weeklyRevenueData : monthlyRevenueData).toLocaleString()}€
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={reportType === "weekly" ? weeklyRevenueData : monthlyRevenueData}
                            margin={{ top: 10, right: 30, left: 0, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip 
                              formatter={(value) => [`${value}€`, 'Revenus']}
                              contentStyle={{
                                backgroundColor: 'white',
                                borderRadius: '0.5rem',
                                border: 'none',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                              }}
                            />
                            <Bar dataKey="revenue" fill="#245536" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="dishes">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-medium flex items-center gap-2">
                        <PieChartIcon className="h-5 w-5 text-restaurant-accent" />
                        Plats les Plus Populaires
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={popularDishesData}
                            margin={{ top: 10, right: 30, left: 80, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                            <XAxis type="number" />
                            <YAxis dataKey="name" type="category" />
                            <Tooltip 
                              formatter={(value) => [`${value}`, 'Commandes']}
                              contentStyle={{
                                backgroundColor: 'white',
                                borderRadius: '0.5rem',
                                border: 'none',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                              }}
                            />
                            <Bar dataKey="orders" fill="#ba3400" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="categories">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-medium flex items-center gap-2">
                        <FileText className="h-5 w-5 text-restaurant-accent" />
                        Ventes par Catégorie
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 flex items-center justify-center">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={salesByCategoryData}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            >
                              {salesByCategoryData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip 
                              formatter={(value) => [`${value}%`, 'Pourcentage']}
                              contentStyle={{
                                backgroundColor: 'white',
                                borderRadius: '0.5rem',
                                border: 'none',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                              }}
                            />
                            <Legend />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
            <div className="flex flex-row gap-4">
              <Select value={reportType} onValueChange={setReportType}>
                <SelectTrigger className="w-full sm:w-40 input-field">
                  <SelectValue placeholder="Période" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="weekly">Hebdomadaire</SelectItem>
                  <SelectItem value="monthly">Mensuel</SelectItem>
                  <SelectItem value="annual">Annuel</SelectItem>
                </SelectContent>
              </Select>
              
              <Button 
                className="bg-restaurant-primary hover:bg-restaurant-primary/90"
                onClick={handleGeneratePDF}
              >
                <FileDown className="mr-2 h-4 w-4" />
                PDF
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="card-dashboard">
        <CardHeader>
          <CardTitle className="text-lg font-medium flex items-center gap-2">
            <Calendar className="h-5 w-5 text-restaurant-accent" />
            Rapports Personnalisés
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <Button variant="outline" className="h-auto py-6 flex flex-col items-center gap-2" onClick={handleGeneratePDF}>
              <FileText className="h-10 w-10 text-restaurant-primary" />
              <span className="text-restaurant-primary font-medium">Rapport de Ventes</span>
              <span className="text-xs text-muted-foreground">Ventes détaillées par période</span>
            </Button>
            
            <Button variant="outline" className="h-auto py-6 flex flex-col items-center gap-2" onClick={handleGeneratePDF}>
              <CircleDollarSign className="h-10 w-10 text-restaurant-primary" />
              <span className="text-restaurant-primary font-medium">Rapport Financier</span>
              <span className="text-xs text-muted-foreground">Revenus et dépenses</span>
            </Button>
            
            <Button variant="outline" className="h-auto py-6 flex flex-col items-center gap-2" onClick={handleGeneratePDF}>
              <PieChartIcon className="h-10 w-10 text-restaurant-primary" />
              <span className="text-restaurant-primary font-medium">Analyse des Plats</span>
              <span className="text-xs text-muted-foreground">Popularité et rentabilité</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Reports;
