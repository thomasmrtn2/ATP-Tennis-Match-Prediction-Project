# Prédiction de Résultats de Matchs ATP avec Machine Learning


## 📖 Description

Ce projet vise à prédire l'issue des matchs de tennis masculin du circuit ATP en utilisant des techniques de Machine Learning. Il met en œuvre un pipeline complet, allant du chargement des données historiques à l'évaluation d'un modèle optimisé, en passant par une ingénierie de caractéristiques approfondie et une gestion rigoureuse des aspects temporels pour éviter les fuites de données.

L'objectif principal est de démontrer une approche méthodique pour aborder un problème de prédiction temporelle complexe, en mettant l'accent sur :
*   L'extraction et la création de caractéristiques pertinentes (statistiques de jeu, forme récente, H2H, classement).
*   La gestion correcte des données temporelles pour l'entraînement et l'évaluation (utilisation de données passées uniquement).
*   L'optimisation des hyperparamètres via la recherche bayésienne (`BayesSearchCV`).
*   L'évaluation robuste du modèle avec `TimeSeriesSplit`.

## 🎯 Problème

Prédire l'issue d'un match de tennis est un défi en raison de la multitude de facteurs influençant la performance (état de forme, surface, historique des confrontations, conditions de jeu, etc.). Ce projet tente de modéliser ces influences à partir de données historiques pour construire un prédicteur fiable.

## 📊 Données

Les données utilisées proviennent du dépôt GitHub très complet de **Jeff Sackmann** : [tennis_atp](https://github.com/JeffSackmann/tennis_atp).
Ce dépôt contient des informations détaillées sur :
*   Les résultats des matchs (simple et double) depuis 1968.
*   Les classements ATP historiques.
*   Des informations sur les joueurs (date de naissance, taille, main dominante, etc.).

Pour ce projet, nous nous concentrons sur les matchs de simple de l'ATP entre **2000 et 2024**.

## 🛠️ Méthodologie

Le projet suit un pipeline structuré (implémenté en grande partie dans `atp_prediction.py`) :

1.  **Chargement des Données :** Chargement des fichiers CSV de matchs, classements et joueurs pour la période sélectionnée.

2.  **Prétraitement Initial :**
    *   Conversion des dates au format datetime.
    *   Nettoyage initial des données joueurs (imputation simple pour taille, date de naissance, etc.).
    *   Tri des données par date pour respecter la chronologie.

3.  **Gestion de la Cible :** La fonction `flip_dataset` permute aléatoirement (50% de chance) les colonnes `winner_*` et `loser_*` et crée une cible binaire `winner` (0 si joueur 1 gagne, 1 si joueur 2 gagne). Ceci évite que le modèle apprenne simplement que "le joueur 1 gagne toujours" et crée un problème de classification binaire équilibré.

4.  **Ingénierie de Caractéristiques (Feature Engineering) :** Création de nombreuses caractéristiques pour capturer différentes facettes du jeu, **en s'assurant de n'utiliser que des informations antérieures au match actuel pour éviter la fuite de données** :
    *   **Forme Récente :** Nombre de jeux joués récemment (`get_Number_of_games_recent`).
    *   **Taux de Victoire :** Taux de victoire globaux, par surface, et sur les N derniers matchs, ajustés avec une approche bayésienne pour plus de robustesse (`create_win_rate`, `win_rate_adjusted`).
    *   **Confrontations Directes (H2H) :** Bilan des rencontres précédentes entre les deux joueurs (`get_h2h`).
    *   **Statistiques de Service :** Pourcentages de 1ère balle, points gagnés sur 1ère/2ème balle, aces/doubles fautes par match (calculés sur l'historique, `get_serve_statistics`).
    *   **Statistiques de Points de Break :** Pourcentage de balles de break sauvées/converties, nombre de balles de break jouées/subies par match (calculés sur l'historique, `get_break_points_statistics`).
    *   **Classement :** Evolution des points ATP sur une période donnée (`get_players_rank_stats`).

5.  **Imputation des Données Manquantes :**
    *   Imputation par la médiane pour les statistiques de service/break calculées (`ServeStatisticsImputer`).
    *   Imputation plus sophistiquée par **KNNImputer** (après standardisation) pour les rangs et certaines statistiques clés, intégrée **correctement** dans un `ColumnTransformer` pour éviter la fuite de données lors de la validation croisée et sur les données de test.

6.  **Prétraitement Final (via `ColumnTransformer`) :**
    *   Suppression des colonnes brutes inutiles.
    *   Encodage One-Hot des variables catégorielles (surface, tournoi, etc.).
    *   Standardisation (`StandardScaler`) des caractéristiques numériques restantes.

7.  **Modélisation :**
    *   Comparaison initiale de plusieurs modèles (RandomForest, AdaBoost, GradientBoosting).
    *   Sélection de `GradientBoostingClassifier` comme modèle principal.

8.  **Optimisation des Hyperparamètres :**
    *   Utilisation de `BayesSearchCV` (bibliothèque `scikit-optimize`) pour trouver les meilleurs hyperparamètres du pipeline complet (incluant la sélection de caractéristiques `SelectKBest` et les paramètres du `GradientBoostingClassifier`).
    *   Validation croisée avec `TimeSeriesSplit` pour respecter la dépendance temporelle des données.

9.  **Évaluation :**
    *   Score de validation croisée (Accuracy) obtenu lors de la recherche bayésienne.
    *   Évaluation finale sur un jeu de test indépendant (données futures par rapport à l'entraînement) pour mesurer la performance de généralisation.

## 🚀 Technologies Utilisées

*   **Python 3.x**
*   **Pandas:** Manipulation et analyse de données.
*   **NumPy:** Calcul numérique.
*   **Scikit-learn:** Pipelines, prétraitement, modèles ML, métriques, validation croisée (`TimeSeriesSplit`), `ColumnTransformer`, `KNNImputer`, `StandardScaler`, `SelectKBest`.
*   **Scikit-optimize (`skopt`):** Optimisation bayésienne (`BayesSearchCV`).
*   **Matplotlib / Seaborn:** Visualisation (utilisé pendant l'exploration, non visible dans le script final).
*   **Re:** Expressions régulières (pour parser le score).

## 📈Résultats

* Modèle Optimal : GradientBoostingClassifier (intégré dans un pipeline Scikit-learn complet incluant prétraitement, imputation KNN et sélection de caractéristiques).
* Meilleur Score CV (Accuracy) : **0.67**
(Score moyen obtenu par validation croisée temporelle lors de l'optimisation Bayesienne)
* Accuracy sur le Test Set (2022-2024) : **0.65**
(Performance finale évaluée sur des matchs non vus pendant l'entraînement ou l'optimisation)

Ces résultats montrent l'incertitude présente au tennis. Montrant que le jeu conserve une part fondamentale d'imprévisibilité, où la forme du jour, la dynamique du match et la force mentale peuvent déjouer les pronostics, et c'est précisément ce qui fait toute la beauté et le suspense de ce sport
## 💡Améliorations Possibles

* Intégrer des systèmes de notation plus dynamiques (type ELO rating) pour mieux capturer la forme relative des joueurs.
* Ajouter des caractéristiques liées à la fatigue (temps passé sur le court récemment, enchaînement des matchs, décalage horaire/voyages).
* Explorer des techniques de Deep Learning (ex: réseaux récurrents ou transformers si les séquences de matchs sont considérées) si la complexité et le volume de données le justifient.
* Affiner l'analyse des erreurs : identifier les types de matchs (surface, niveau de tournoi, H2H spécifique) où le modèle échoue le plus pour guider l'amélioration des caractéristiques.
* Tester des fenêtres temporelles différentes pour le calcul des statistiques roulantes.

## Auteur

Thomas Martin - https://github.com/thomasmrtn2 


## Remerciements

Un grand merci à Jeff Sackmann pour la collecte exhaustive et la mise à disposition publique des données de tennis via son dépôt tennis_atp (https://github.com/JeffSackmann/tennis_atp).




