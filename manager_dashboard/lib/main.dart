import 'package:flutter/material.dart';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:excel/excel.dart' as excel;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const RestaurantManagerApp());
}

class RestaurantManagerApp extends StatelessWidget {
  const RestaurantManagerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFFBA3400),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF245536),
          secondary: const Color(0xFFDB9051),
        ),
        scaffoldBackgroundColor: const Color(0xFFE9B975),
        fontFamily: 'Roboto',
      ),
      home: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) =>
            snapshot.hasData ? const ManagerDashboard() : const LoginScreen(),
      ),
    );
  }
}

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  Future<void> _login() async {
    try {
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: _emailController.text,
        password: _passwordController.text,
      );
    } on FirebaseAuthException catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erreur: ${e.message}')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('Gestionnaire Restaurant',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
            const SizedBox(height: 40),
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Mot de passe',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _login,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF245536),
                foregroundColor: Colors.white,
              ),
              child: const Text('Connexion'),
            ),
          ],
        ),
      ),
    );
  }
}

class ManagerDashboard extends StatefulWidget {
  const ManagerDashboard({super.key});

  @override
  State<ManagerDashboard> createState() => _ManagerDashboardState();
}

class _ManagerDashboardState extends State<ManagerDashboard> {
  int _selectedIndex = 0;
  // Removed unused '_db' field
  final List<Widget> _tabs = [
    const ReservationsTab(),
    const MenuManagementTab(),
    const StaffManagementTab(),
    const FinancialReportsTab(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tableau de Bord'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () => FirebaseAuth.instance.signOut(),
          ),
        ],
      ),
      body: _tabs[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        selectedItemColor: const Color(0xFFBA3400),
        unselectedItemColor: const Color(0xFF245536),
        onTap: (index) => setState(() => _selectedIndex = index),
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.event),
            label: 'Réservations',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.restaurant_menu),
            label: 'Menu',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.people),
            label: 'Personnel',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.assessment),
            label: 'Rapports',
          ),
        ],
      ),
    );
  }
}

class ReservationsTab extends StatelessWidget {
  const ReservationsTab({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<DatabaseEvent>(
      stream: FirebaseDatabase.instance.ref('reservations').onValue,
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        final reservations = Map<String, dynamic>.from(
            snapshot.data!.snapshot.value as Map? ?? {});

        return ListView.builder(
          itemCount: reservations.length,
          itemBuilder: (context, index) {
            final key = reservations.keys.elementAt(index);
            final res = reservations[key];
            return Card(
              margin: const EdgeInsets.all(8),
              child: ListTile(
                title: Text('Table ${res['table']} - ${res['time']}'),
                subtitle: Text('${res['date']} - ${res['status']}'),
                trailing: res['status'] == 'à venir'
                    ? ElevatedButton(
                        onPressed: () => FirebaseDatabase.instance
                            .ref('reservations/$key/status')
                            .set('confirmée'),
                        child: const Text('Confirmer'),
                      )
                    : null,
              ),
            );
          },
        );
      },
    );
  }
}

class MenuManagementTab extends StatefulWidget {
  const MenuManagementTab({super.key});

  @override
  State<MenuManagementTab> createState() => _MenuManagementTabState();
}

class _MenuManagementTabState extends State<MenuManagementTab> {
  final DatabaseReference _menuRef =
      FirebaseDatabase.instance.ref('menu_items');

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<DatabaseEvent>(
      stream: _menuRef.onValue,
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        final menuItems = Map<String, dynamic>.from(
            snapshot.data!.snapshot.value as Map? ?? {});

        return ListView.builder(
          itemCount: menuItems.length,
          itemBuilder: (context, index) {
            final key = menuItems.keys.elementAt(index);
            final item = menuItems[key];
            return Card(
              margin: const EdgeInsets.all(8),
              child: ListTile(
                title: Text(item['name']),
                subtitle: Text('Stock: ${item['stock']}'),
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.remove),
                      onPressed: () => _updateStock(key, item['stock'] - 1),
                    ),
                    IconButton(
                      icon: const Icon(Icons.add),
                      onPressed: () => _updateStock(key, item['stock'] + 1),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  void _updateStock(String itemId, int newStock) {
    _menuRef.child('$itemId/stock').set(newStock);
  }
}

class StaffManagementTab extends StatelessWidget {
  const StaffManagementTab({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<DatabaseEvent>(
      stream: FirebaseDatabase.instance.ref('employees').onValue,
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        final employees = Map<String, dynamic>.from(
            snapshot.data!.snapshot.value as Map? ?? {});

        return ListView.builder(
          itemCount: employees.length,
          itemBuilder: (context, index) {
            final key = employees.keys.elementAt(index);
            final emp = employees[key];
            return Card(
              margin: const EdgeInsets.all(8),
              child: ListTile(
                title: Text(emp['name']),
                subtitle: Text(emp['position']),
                trailing: Text('${emp['hours']}h/semaine'),
              ),
            );
          },
        );
      },
    );
  }
}

class FinancialReportsTab extends StatelessWidget {
  const FinancialReportsTab({super.key});

  Future<void> _generateExcelReport() async {
    final salesSnapshot = await FirebaseDatabase.instance.ref('sales').get();
    final excelFile = excel.Excel.createExcel();
    final sheet = excelFile['Ventes'];

    sheet.appendRow(['Date', 'Montant', 'Méthode de Paiement']);

    (salesSnapshot.value as Map).forEach((key, value) {
      sheet.appendRow([
        value['date'],
        value['amount'],
        value['payment_method'],
      ]);
    });

    final bytes = excelFile.save();
    // Example usage: Save the file locally or upload it to a server
    // For demonstration, we'll print the length of the bytes
    print('Excel file generated with ${bytes?.length} bytes.');
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ElevatedButton(
        onPressed: _generateExcelReport,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF245536),
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        ),
        child: const Text(
          'Générer Rapport Excel',
          style: TextStyle(fontSize: 18),
        ),
      ),
    );
  }
}
