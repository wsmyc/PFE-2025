import pandas as pd
import numpy as np

np.random.seed(42)
num_users = 100

users = []
for user_id in range(1, num_users + 1):
    users.append({
        'id_utilisateur': user_id,
        'âge': np.random.randint(18, 65),
        'préférence_alimentaire': np.random.choice(
            ["Aucune", "Végétarien", "Végétalien", "Sans Gluten", "Halal"],
            p=[0.7, 0.2, 0.05, 0.03, 0.02]
        ),
        'catégorie_préférée': np.random.choice(
            ["Entrées", "Burgers & Sandwichs", "Plats Traditionnels", 
             "Accompagnements", "Boissons Chaudes", "Boissons Froides", "Desserts"],
            p=[0.15, 0.25, 0.2, 0.1, 0.1, 0.1, 0.1]
        )
    })

pd.DataFrame(users).to_csv('data/utilisateurs.csv', index=False, encoding='utf-8-sig')
print("Utilisateurs générés ✓")