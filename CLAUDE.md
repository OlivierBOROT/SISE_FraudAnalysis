# Recommandations pour la Pipeline de Classification Déséquilibrée

## Contexte
- Dataset: ~4M lignes, ~0.6% de fraudes (23,346 fraudes / 3,865,122 légitimes)
- Stratégie: Undersampling (10%) + SMOTE (20%) → ~292k échantillons finaux
- Objectif: Monter plusieurs modèles rapidement sur données rééquilibrées

## 1. Architecture de Pipeline Recommandée

### Option A: Pipeline Modulaire avec imblearn (RECOMMANDÉ)
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# Pipeline complète réutilisable
def create_model_pipeline(model, name="model"):
    return ImbPipeline([
        ('scaler', StandardScaler()),  # Optionnel selon le modèle
        ('classifier', model)
    ])

# Exemples de modèles à tester
models = {
    'rf': RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1),
    'xgb': XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                         scale_pos_weight=4, n_jobs=-1),
    'lgbm': LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                           class_weight='balanced', n_jobs=-1),
    'gbc': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                       learning_rate=0.1),
}
```

### Option B: Pipeline avec Sampling Intégré
```python
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Pipeline avec sampling + modèle
def create_full_pipeline(model):
    return Pipeline([
        ('rus', RandomUnderSampler(sampling_strategy=0.10)),
        ('smote', SMOTE(sampling_strategy=0.25)),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
```

## 2. Modèles Recommandés pour Données Rééquilibrées

### Modèles Rapides et Efficaces
1. **LightGBM** (MEILLEUR CHOIX)
   - Très rapide sur ~300k échantillons
   - Gère bien le déséquilibre avec `class_weight='balanced'`
   - Paramètres: `n_estimators=200-500, max_depth=7-10, learning_rate=0.05-0.1`

2. **XGBoost**
   - Excellent pour la détection de fraude
   - `scale_pos_weight` pour ajuster le déséquilibre résiduel (20%)
   - Paramètres: `n_estimators=200-400, max_depth=5-8`

3. **Random Forest**
   - Parallélisable (n_jobs=-1)
   - Robuste, moins de tuning nécessaire
   - Paramètres: `n_estimators=200-300, max_depth=15-20`

4. **CatBoost**
   - Très bon sur données catégorielles
   - `auto_class_weights='Balanced'`
   - Plus lent mais performant

### Modèles Complémentaires pour Stacking
5. **Logistic Regression** (avec L2 régularisation)
   - Rapide, bon comme baseline
   - Utile comme meta-model

6. **Gradient Boosting Classifier** (sklearn)
   - Moins rapide que LightGBM mais différent algorithme
   - Bon pour diversifier l'ensemble

## 3. Stratégie d'Hyperparameter Tuning

### Approche Efficace avec RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Exemple pour LightGBM
param_distributions = {
    'n_estimators': randint(200, 600),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.15),
    'num_leaves': randint(20, 100),
    'min_child_samples': randint(20, 100),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}

model = LGBMClassifier(class_weight='balanced', n_jobs=-1)

search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=20,  # 20 combinaisons au lieu de toutes
    cv=3,       # 3 folds pour aller plus vite
    scoring='f1',  # ou votre f1_fraud_scorer
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_final, y_final)
```

### Grilles Ciblées pour Chaque Modèle
```python
# XGBoost
xgb_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.15],
    'scale_pos_weight': [3, 4, 5],  # Ajuster selon ratio résiduel
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Random Forest
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# LightGBM (le plus rapide)
lgbm_params = {
    'n_estimators': [300, 400, 500],
    'max_depth': [7, 10, 12],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'min_child_samples': [20, 50, 100]
}
```

## 4. Stratégie de Stacking sur Données Rééquilibrées

### Approche 1: Stacking Classique avec CV
```python
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import StackingClassifier

# Modèles de base (entrainés sur X_final, y_final)
base_models = [
    ('lgbm', LGBMClassifier(n_estimators=400, max_depth=10,
                            class_weight='balanced', n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=7,
                          scale_pos_weight=4, n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)),
]

# Meta-model (simple)
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Stacking avec CV automatique
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba',  # Utilise les probabilités
    n_jobs=-1
)

# Entrainement sur données rééquilibrées
stacking_clf.fit(X_final, y_final)

# Prédiction sur test complet
y_pred = stacking_clf.predict(X_test)
y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
```

### Approche 2: Stacking Manuel avec OOF
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_oof_predictions(models, X, y, n_splits=5):
    """Génère les prédictions out-of-fold pour le stacking"""
    oof_preds = np.zeros((len(X), len(models)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (name, model) in enumerate(models):
        oof_pred = cross_val_predict(
            model, X, y,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]
        oof_preds[:, i] = oof_pred
        print(f"OOF for {name} completed")

    return oof_preds

# Générer les OOF predictions
oof_train = get_oof_predictions(base_models, X_final, y_final)

# Entrainer le meta-model
meta_model.fit(oof_train, y_final)

# Pour le test: entrainer tous les base models sur données complètes
test_preds = np.zeros((len(X_test), len(base_models)))
for i, (name, model) in enumerate(base_models):
    model.fit(X_final, y_final)
    test_preds[:, i] = model.predict_proba(X_test)[:, 1]

# Prédiction finale
final_pred = meta_model.predict(test_preds)
final_pred_proba = meta_model.predict_proba(test_preds)[:, 1]
```

## 5. Métriques d'Évaluation Recommandées

```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

def evaluate_fraud_model(y_true, y_pred, y_pred_proba):
    """Évaluation complète pour détection de fraude"""
    print("=== Métriques de Classification ===")
    print(f"F1 Score (Fraud): {f1_score(y_true, y_pred):.4f}")
    print(f"Precision (Fraud): {precision_score(y_true, y_pred):.4f}")
    print(f"Recall (Fraud): {recall_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_pred_proba):.4f}")

    print("\n=== Matrice de Confusion ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Légit', 'Fraude']))

    return {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba)
    }
```

## 6. Workflow Complet Recommandé

```python
# 1. Préparer les données (déjà fait)
# X_final, y_final = données rééquilibrées (~292k)
# X_test, y_test = données test complètes

# 2. Définir les modèles avec leurs meilleurs hyperparamètres
models_dict = {
    'lgbm': LGBMClassifier(
        n_estimators=400, max_depth=10, learning_rate=0.1,
        num_leaves=50, min_child_samples=50,
        class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'xgb': XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        scale_pos_weight=4, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    ),
    'rf': RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'gbc': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
}

# 3. Entrainer chaque modèle individuellement et évaluer
results = {}
for name, model in models_dict.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")

    model.fit(X_final, y_final)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results[name] = evaluate_fraud_model(y_test, y_pred, y_pred_proba)

# 4. Stacking des meilleurs modèles
base_models = [(name, model) for name, model in models_dict.items()]
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

print(f"\n{'='*50}")
print("Training Stacking Model...")
print(f"{'='*50}")

stacking_clf.fit(X_final, y_final)
y_pred_stack = stacking_clf.predict(X_test)
y_pred_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

results['stacking'] = evaluate_fraud_model(y_test, y_pred_stack, y_pred_proba_stack)

# 5. Comparer les résultats
import pandas as pd
results_df = pd.DataFrame(results).T
print("\n=== Comparaison des Modèles ===")
print(results_df.sort_values('f1', ascending=False))
```

## 7. Techniques Avancées à Explorer

### A. Techniques de Sampling Alternatives
```python
# TomekLinks: Nettoie les points mal classés à la frontière
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(
    sampling_strategy=0.25,
    tomek=TomekLinks(sampling_strategy='majority')
)
X_clean, y_clean = smote_tomek.fit_resample(X_resampled, y_resampled)

# NearMiss: Undersampling intelligent basé sur la distance
from imblearn.under_sampling import NearMiss

nearmiss = NearMiss(version=2, n_neighbors=3)
X_nm, y_nm = nearmiss.fit_resample(X_train, y_train)
```

### B. Calibration des Probabilités
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrer les probabilités du modèle
calibrated_clf = CalibratedClassifierCV(
    best_model,
    method='isotonic',  # ou 'sigmoid'
    cv=5
)
calibrated_clf.fit(X_final, y_final)
```

### C. Threshold Tuning
```python
from sklearn.metrics import precision_recall_curve

# Trouver le meilleur seuil pour maximiser F1
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal threshold: {best_threshold:.4f}")
y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)
```

## 8. Conseils pour Optimiser le Temps de Calcul

1. **Utiliser `n_jobs=-1`** sur tous les modèles parallélisables
2. **RandomizedSearchCV** plutôt que GridSearchCV (20-30 itérations suffisent)
3. **Réduire le nombre de CV folds** (3 au lieu de 5) pour le tuning
4. **LightGBM > XGBoost > RandomForest** en termes de vitesse
5. **Sauvegarder les modèles entraînés** avec pickle/joblib
6. **Early stopping** pour les boosting models:
```python
lgbm = LGBMClassifier(n_estimators=1000, early_stopping_rounds=50)
lgbm.fit(X_final, y_final, eval_set=[(X_val, y_val)], verbose=False)
```

## 9. Structure de Code Recommandée

```
SISE_FraudAnalysis/
├── notebooks/
│   ├── 5_models_undersampling.ipynb
│   └── 6_stacking_pipeline.ipynb (NOUVEAU)
├── src/
│   ├── preprocessing.py (fonctions de sampling)
│   ├── models.py (définitions des modèles)
│   ├── evaluation.py (métriques)
│   └── stacking.py (pipelines de stacking)
├── models/ (modèles sauvegardés)
│   ├── lgbm_best.pkl
│   ├── xgb_best.pkl
│   └── stacking_final.pkl
└── results/
    └── model_comparison.csv
```

## 10. Checklist d'Implémentation

- [ ] Tester LightGBM en premier (le plus rapide)
- [ ] Tester XGBoost avec `scale_pos_weight`
- [ ] Tester Random Forest avec `class_weight='balanced'`
- [ ] Comparer RandomizedSearchCV vs GridSearchCV sur un modèle
- [ ] Implémenter le stacking avec StackingClassifier
- [ ] Évaluer avec F1, Precision, Recall, ROC-AUC, PR-AUC
- [ ] Tester threshold tuning pour optimiser F1
- [ ] Explorer SMOTETomek si le temps le permet
- [ ] Sauvegarder les meilleurs modèles
- [ ] Documenter les résultats dans un notebook dédié

## Notes Importantes

- **Ne pas sur-optimiser** sur le set de validation rééchantillonné
- **Toujours évaluer sur le test set complet** (distribution réelle)
- **Le F1-score sur la classe minoritaire** est votre métrique principale
- **Le stacking apporte généralement +1-3% de F1** par rapport au meilleur modèle seul
- **La calibration des probabilités** peut être cruciale si vous utilisez un seuil de décision

---
*Document créé le 2026-01-12 par Claude Code*
