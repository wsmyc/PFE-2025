import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
import os

fake = Faker('fr_FR')
np.random.seed(42)

# Charger les données
menu = pd.read_csv('data/menu.csv')
users = pd.read_csv('data/utilisateurs.csv')

# Créer mapping ID articles
item_id_map = {row['id_article']: idx for idx, row in menu.iterrows()}

# Paramètres de génération
restrictions = {
    'Végétarien': ['bœuf', 'poulet', 'saumon', 'thon', 'porc', 'jambon', 'lardons'],
    'Végétalien': ['œufs', 'lait', 'crème', 'fromage', 'beurre', 'yaourt'],
    'Sans Gluten': ['blé', 'pain', 'pâte', 'farine', 'seigle'],
    'Halal': ['porc', 'lardons', 'alcool']
}

ratings = []

for _, user in tqdm(users.iterrows(), total=len(users)):
    user_id = user['id_utilisateur']
    regime = user['préférence_alimentaire']
    
    # Filtrer articles compatibles
    compatible = []
    for _, item in menu.iterrows():
        ingredients = item['ingrédients'].lower()
        if regime == 'Aucune' or not any(ing in ingredients for ing in restrictions.get(regime, [])):
            compatible.append(item['id_article'])
    
    # Générer 10-25 notes par utilisateur
    for item_id in np.random.choice(compatible, size=np.random.randint(10, 25), replace=False):
        # Générer note réaliste
        base_note = 4.0 if menu.loc[item_id_map[item_id], 'catégorie'] == user['catégorie_préférée'] else 3.0
        note = np.clip(np.random.normal(base_note, 0.5), 1.0, 5.0)
        
        ratings.append({
            'id_utilisateur': user_id,
            'id_article': item_id,
            'note': round(note, 1),
            'date': fake.date_between(start_date='-1y', end_date='today').isoformat()
        })

# Sauvegarder
pd.DataFrame(ratings).to_csv('data/notes.csv', index=False)
print("Notes générées ✓")