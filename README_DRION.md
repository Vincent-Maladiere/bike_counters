### External_data_sorted

external_data_cleaned : cette variable permet de stocker un external_data nettoyé, afin de pouvoir itérer facilement dessus.

On utilise la fonction dropna(axis=1, how=all) pour supprimer toutes les colonnes dont l'intégralité des valeurs sont des Nan. On sait grâce à l'instruction select_dtypes que toutes les colonnes, à l'exception de la colonne date, sont constituées de valeurs numériques.

Cette méthode donne le même résultat que le code suivant, qui permettait d'éliminer les colonnes dont la somme des valeurs absolues étaient égales à 0 :

_________________________________________________________________________________________________________
# Let's first separate numerical columns from categorical ones
numerical_features = external_data.select_dtypes(include=np.number)
categorical_features = external_data.select_dtypes(exclude=np.number)

print("Colonnes numériques:\n", numerical_features)
print("Colonnes catégorielles:\n", categorical_features)

# Calculer la somme des valeurs absolues pour chaque colonne numérique
somme_abs_colonnes = numerical_features.apply(lambda x: x.abs().sum())

# Filtrer les colonnes dont la somme des valeurs absolues est égale à zéro
colonnes_somme_zero = somme_abs_colonnes[somme_abs_colonnes == 0].index

print("Noms des colonnes numériques dont la somme des valeurs absolues est égale à zéro:", colonnes_somme_zero)

external_data_cleaned_1 = external_data.drop(columns=colonnes_somme_zero)
_________________________________________________________________________________________________________

Nous sélectionnons ensuite uniquement les valeurs d'intérêts pour l'entraînement, stockées dans la variable external_data_train. Nous faisons de même pour l'échantillon test avec external_data_test