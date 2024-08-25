# RappelConso - Chatbot & Dashboard

## Introduction
Bienvenue sur **RappelConso - Chatbot & Dashboard**. Cette application permet de consulter et d'analyser les rappels de produits alimentaires en France, en utilisant les données fournies par la base de données officielle RappelConso.

L'application se divise en plusieurs sections accessibles via la barre latérale de navigation.

## Utilisation de l'application

### 1. Page Principale
La page principale présente un tableau de bord synthétisant les rappels de produits récents. Elle se compose des éléments suivants :
- **Filtres Avancés** : Utilisez les filtres pour affiner votre recherche par sous-catégories, risques et périodes de temps.
- **Nombre Total de Rappels** : Un indicateur du nombre total de rappels correspondant aux critères sélectionnés.
- **Graphiques Top 5** : Deux graphiques affichent les 5 sous-catégories de produits les plus rappelées et les 5 principaux risques.
- **Liste des Derniers Rappels** : Une liste paginée des rappels les plus récents, incluant le nom du produit, la date de rappel, la marque, le motif du rappel, et un lien pour voir l'affichette du rappel.

### 2. Visualisation
Cette section permet de visualiser les rappels de produits sous forme de graphiques interactifs :
- **Sous-catégories** : Un diagramme en camembert montrant la répartition des rappels par sous-catégorie de produit.
- **Décision de Rappel** : Un diagramme en camembert illustrant la nature juridique des rappels.
- **Nombre de Rappels par Mois** : Un histogramme montrant l'évolution du nombre de rappels mois par mois.

### 3. Détails
La page des détails permet de consulter un tableau complet des rappels de produits correspondant aux critères sélectionnés. Vous pouvez :
- **Voir les Détails** : Explorer chaque rappel avec toutes les informations disponibles.
- **Télécharger les Données** : Télécharger les données filtrées au format CSV pour une analyse plus approfondie.

### 4. Chatbot
La page Chatbot vous permet de poser des questions spécifiques concernant les rappels de produits. Vous pouvez :
- **Poser une Question** : Entrez une question relative aux rappels de produits (ex. "Quels produits ont été rappelés pour cause de Listeria ?").
- **Obtenir des Réponses** : Le chatbot vous répondra en fonction des données disponibles, en fournissant des informations claires et précises.

## Mise à Jour des Données
Les données utilisées par cette application sont récupérées dynamiquement à partir de la base de données RappelConso via une requête API. Voici comment les données sont mises à jour :

- **Chargement des Données** : À chaque lancement de l'application, les données sont récupérées à partir de l'API RappelConso. Les données sont ensuite nettoyées, notamment en ce qui concerne les dates, et stockées pour être utilisées dans l'application.
- **Mise à Jour Automatique** : Si l'application est déployée sur un serveur (comme Streamlit Cloud), les données seront automatiquement mises à jour à chaque redémarrage de l'application.
- **Cache des Données** : Pour des performances optimales, les données sont mises en cache, ce qui permet d'éviter des appels API répétés lors de l'utilisation continue de l'application.

## Configuration de l'API Gemini
Pour interagir avec l'API Gemini et permettre au chatbot de répondre aux questions des utilisateurs, une clé API doit être configurée dans l'application. Cette clé est stockée en toute sécurité dans les secrets de l'application et utilisée pour initier les sessions de chat avec le modèle Gemini.

## Conclusion
Cette application est conçue pour être un outil intuitif et puissant pour le suivi des rappels de produits alimentaires en France. Elle offre une combinaison unique d'analyse de données et d'interaction via chatbot, permettant aux utilisateurs de rester informés des derniers rappels de manière interactive et visuelle.
