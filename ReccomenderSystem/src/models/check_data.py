import os
import pandas as pd

def verify_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Vérifier notes.csv
    notes_path = os.path.join(base_dir, 'data', 'notes.csv')
    notes = pd.read_csv(notes_path)
    
    # Vérifier utilisateurs.csv
    users_path = os.path.join(base_dir, 'data', 'utilisateurs.csv')
    users = pd.read_csv(users_path)
    
    # Vérifier menu.csv
    menu_path = os.path.join(base_dir, 'data', 'menu.csv')
    menu = pd.read_csv(menu_path)
    
    print("=== Vérification des IDs ===")
    
    # Cohérence utilisateurs
    unique_users_notes = notes['id_utilisateur'].unique()
    unique_users_list = users['id_utilisateur'].unique()
    print(f"Utilisateurs dans notes.csv: {len(unique_users_notes)}")
    print(f"Utilisateurs dans utilisateurs.csv: {len(unique_users_list)}")
    
    # Cohérence articles
    unique_items_notes = notes['id_article'].unique()
    unique_items_menu = menu['id_article'].unique()
    print(f"\nArticles dans notes.csv: {len(unique_items_notes)}")
    print(f"Articles dans menu.csv: {len(unique_items_menu)}")
    
    # Vérifier les IDs manquants
    missing_in_menu = set(unique_items_notes) - set(unique_items_menu)
    if missing_in_menu:
        print(f"\nATTENTION: IDs articles manquants dans menu.csv: {missing_in_menu}")
    
    print("\n=== Vérification terminée ===")

if __name__ == "__main__":
    verify_data()