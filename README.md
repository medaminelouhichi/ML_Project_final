# Projet ML — TON_IoT Network : détection d'anomalies + prédiction du type

Ordre d'exécution : 01 -> 02..05 (indépendants entre eux) -> 06. Placer `train_test_network.csv` dans `data/`.
Dépendances : pandas numpy scikit-learn xgboost imbalanced-learn seaborn matplotlib pyarrow joblib tabulate.
Chaque notebook importe `src/ml_utils.py` (chemins relatifs au projet, indépendants du dossier courant).

| Notebook | Membre | Contenu |
|---|---|---|
| 01_prep | 1 | qualité (`-`=NaN), EDA, feature engineering, déduplication, outliers (log1p), split 60/20/20, sélection de variables, étude de l'équilibrage |
| 02_anomaly_random_forest | 2 | RF binaire : baseline, équilibrage (CV), RandomizedSearch, courbe d'apprentissage, calibration seuil + probabilités, test, importances, erreurs par type |
| 03_anomaly_svm | 3 | SVM linéaire, RBF, Nystroem+linéaire ; mêmes étapes |
| 04_type_random_forest_regressor | 4 | RF Regressor sur code ordinal, pondération, tuning, seuils de décodage optimisés |
| 05_type_xgboost_regressor | 5 | XGBoost Regressor, early stopping, mêmes étapes |
| 06_comparison | tous | tableau comparatif final (`reports/model_comparison.csv`), graphiques, limites |

Les notebooks sont fournis déjà exécutés. Durée totale de ré-exécution : ~1 h sur 1 CPU (02 ≈ 25 min, 04/05 ≈ 10-15 min).
