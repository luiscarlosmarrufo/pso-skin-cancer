"""
FIX RÁPIDO: Clasificadores con Manejo de Desbalance
Ejecuta esto usando los features ya guardados
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ==========================================
# CARGAR DATOS
# ==========================================
print("📂 Cargando datos guardados...")

MODELS_DIR = Path("../results/exp3_20251016_124249/models")
FIGURES_DIR = MODELS_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

Xtr_feat = np.load(MODELS_DIR / "Xtr_feat.npy")
Xte_feat = np.load(MODELS_DIR / "Xte_feat.npy")
y_train = np.load(MODELS_DIR / "y_train.npy")
y_test = np.load(MODELS_DIR / "y_test.npy")
selected_idx = np.load(MODELS_DIR / "selected_features.npy")

Xtr_sel = Xtr_feat[:, selected_idx]
Xte_sel = Xte_feat[:, selected_idx]

print(f"✅ Train: {Xtr_sel.shape}, Test: {Xte_sel.shape}")

# ==========================================
# FUNCIÓN DE EVALUACIÓN COMPLETA
# ==========================================
def evaluate_model(X_train, y_train, X_test, y_test, clf, name, use_sample_weight=False):
    """Entrena y evalúa con métricas balanceadas"""
    print(f"\n{'='*60}")
    print(f"⏳ Entrenando: {name}")
    print(f"{'='*60}")
    
    # Pipeline
    pipe = make_pipeline(StandardScaler(), clf)
    
    # Entrenar (con sample weights si es KNN)
    if use_sample_weight:
        sample_weights = compute_sample_weight('balanced', y_train)
        # Entrenar solo el clasificador final con weights
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test_scaled)
    else:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    
    # Métricas
    acc = metrics.accuracy_score(y_test, y_pred)
    balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    sen = metrics.recall_score(y_test, y_pred, pos_label=1)
    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    pre = metrics.precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, pos_label=1)
    
    # Report completo
    print(f"\n📊 RESULTADOS:")
    print(f"   Accuracy:          {acc:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f} ⭐")
    print(f"   Sensitivity:       {sen:.4f} (Recall Malignant)")
    print(f"   Specificity:       {spe:.4f} (Recall Benign)")
    print(f"   Precision:         {pre:.4f}")
    print(f"   F1-Score:          {f1:.4f}")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"              Benign  Malignant")
    print(f"   Benign      {tn:5d}    {fp:5d}")
    print(f"   Malignant   {fn:5d}    {tp:5d}")
    print(f"\n   False Negatives: {fn} ⚠️ (casos malignos perdidos)")
    print(f"   False Positives: {fp} (falsos alarmas)")
    
    # Interpretación clínica
    if sen < 0.7:
        print(f"\n   ⚠️ SENSITIVITY BAJA: Solo detecta {sen*100:.1f}% de cáncer")
    elif sen < 0.9:
        print(f"\n   ⚡ SENSITIVITY MODERADA: Detecta {sen*100:.1f}% de cáncer")
    else:
        print(f"\n   ✅ SENSITIVITY ALTA: Detecta {sen*100:.1f}% de cáncer")
    
    # Visualización
    plt.figure(figsize=(10, 4))
    
    # Subplot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f"{name}\nConfusion Matrix", fontweight='bold')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Subplot 2: Métricas
    plt.subplot(1, 2, 2)
    metrics_data = {
        'Accuracy': acc,
        'Balanced\nAccuracy': balanced_acc,
        'Sensitivity': sen,
        'Specificity': spe,
        'F1-Score': f1
    }
    colors = ['#3498db' if v >= 0.8 else '#e74c3c' if v < 0.5 else '#f39c12' 
              for v in metrics_data.values()]
    plt.barh(list(metrics_data.keys()), list(metrics_data.values()), color=colors)
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title(f"{name}\nPerformance Metrics", fontweight='bold')
    for i, (k, v) in enumerate(metrics_data.items()):
        plt.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"detailed_{name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "precision": float(pre),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "false_negatives": int(fn),
        "false_positives": int(fp)
    }
    
    return results, pipe if not use_sample_weight else (scaler, clf)

# ==========================================
# CLASIFICADORES CON MANEJO DE DESBALANCE
# ==========================================

results_all = {}

print("\n" + "="*60)
print("🔧 CLASIFICADORES CON CLASS BALANCING")
print("="*60)

# 1. SVM con class_weight='balanced'
print("\n1️⃣ SVM RBF Balanceado")
svm_balanced = SVC(
    kernel="rbf", 
    class_weight='balanced',  # ← Automáticamente ajusta pesos
    random_state=42,
    probability=True  # Para ROC curves
)
results_all['SVM_Balanced'], _ = evaluate_model(
    Xtr_sel, y_train, Xte_sel, y_test, 
    svm_balanced, "SVM RBF Balanced"
)

# 2. SVM Linear Balanceado (más rápido)
print("\n2️⃣ SVM Linear Balanceado")
from sklearn.svm import LinearSVC
svm_linear = LinearSVC(
    class_weight='balanced',
    random_state=42,
    max_iter=2000
)
results_all['SVM_Linear_Balanced'], _ = evaluate_model(
    Xtr_sel, y_train, Xte_sel, y_test,
    svm_linear, "SVM Linear Balanced"
)

# 3. Random Forest Balanceado
print("\n3️⃣ Random Forest Balanceado")
from sklearn.ensemble import RandomForestClassifier
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
results_all['RandomForest_Balanced'], _ = evaluate_model(
    Xtr_sel, y_train, Xte_sel, y_test,
    rf_balanced, "Random Forest Balanced"
)

# 4. Subspace KNN con Sample Weights
print("\n4️⃣ Subspace KNN con Sample Weights")
# KNN en Bagging no soporta sample_weight directamente
# Mejor opción: usar otro ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_balanced = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
    n_estimators=50,
    random_state=42,
    algorithm='SAMME'
)
results_all['AdaBoost_Balanced'], _ = evaluate_model(
    Xtr_sel, y_train, Xte_sel, y_test,
    ada_balanced, "AdaBoost Balanced"
)

# ==========================================
# COMPARACIÓN FINAL
# ==========================================
print("\n" + "="*60)
print("📊 COMPARACIÓN DE TODOS LOS MODELOS")
print("="*60)

comparison_df = []
for name, results in results_all.items():
    comparison_df.append({
        'Model': name,
        'Accuracy': results['accuracy'],
        'Balanced Acc': results['balanced_accuracy'],
        'Sensitivity': results['sensitivity'],
        'Specificity': results['specificity'],
        'F1': results['f1'],
        'False Neg': results['false_negatives']
    })

import pandas as pd
df = pd.DataFrame(comparison_df)
df = df.sort_values('Balanced Acc', ascending=False)  # ← Corregido: usa 'Balanced Acc'

print("\n" + df.to_string(index=False))

# Encontrar mejor modelo
best_model = df.iloc[0]['Model']
print(f"\n🏆 MEJOR MODELO: {best_model}")
print(f"   Balanced Accuracy: {df.iloc[0]['Balanced Acc']:.4f}")
print(f"   Sensitivity: {df.iloc[0]['Sensitivity']:.4f}")
print(f"   False Negatives: {int(df.iloc[0]['False Neg'])}")

# Guardar resultados
with open(MODELS_DIR.parent / "results_balanced.json", "w") as f:
    json.dump({
        "all_models": results_all,
        "best_model": best_model,
        "comparison_table": df.to_dict('records')
    }, f, indent=2)

print(f"\n✅ Resultados guardados en: {MODELS_DIR.parent / 'results_balanced.json'}")
print(f"📊 Gráficas detalladas en: {FIGURES_DIR}")