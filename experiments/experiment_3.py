"""
==========================================
Experimento 3 (Paper): Xception -> PSO -> Subspace KNN
Basado en: Shah et al. (2024) - J. Imaging 10, 332
==========================================
"""
import os, random, json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.xception import preprocess_input

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import pyswarms as ps

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================
# Crear estructura de directorios
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / f"exp3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for dir_path in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"📁 Resultados se guardarán en: {RESULTS_DIR}")

# ==========================================
# CONFIGURACIÓN (según Tabla 4 del paper)
# ==========================================
CONFIG = {
    # Datos
    "DATA_DIR": "../../data/data_isic_bin",
    "IMG_SIZE": 224,
    "BATCH": 32,
    "VAL_SPLIT": 0.2,
    "SEED": 42,
    
    # PSO (Tabla 4 del paper)
    "PSO_ITERS": 100,
    "PSO_C1": 2.5,
    "PSO_C2": 2.5,
    "PSO_W": 0.7,
    "N_PARTICLES": 30,
    "TARGET_DIM": 508,
    
    # Subspace KNN
    "N_ESTIMATORS": 30,
    "K_IN_BASE": 5,
    "SUBSPACE_FRAC": 0.5,
}

# Guardar configuración
with open(RESULTS_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# Semillas para reproducibilidad
np.random.seed(CONFIG["SEED"])
random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ GPU configurada: {len(gpus)} dispositivo(s)")
    except Exception as e:
        print(f"⚠️ Error configurando GPU: {e}")

# ==========================================
# 1. CARGA Y AUGMENTACIÓN DE DATOS
# ==========================================
print("\n" + "="*60)
print("1. CARGANDO Y AUGMENTANDO DATOS")
print("="*60)

def augment_image(image, label):
    """Data augmentation según Sección 3.1.3 del paper"""
    # Random rotation (-180 to 180 degrees)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness (simula Gaussian blur effect)
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast (simula sharpening effect)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    return image, label

def load_ds_to_numpy(data_dir, img_size=224, batch=32, val_split=0.2, seed=42, augment=True):
    """Carga dataset con augmentation opcional"""
    common = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch,
        shuffle=True,
        seed=seed,
        validation_split=val_split
    )
    
    ds_train = image_dataset_from_directory(subset="training", **common)
    ds_val = image_dataset_from_directory(subset="validation", **common)
    
    # Preprocesado Xception
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.map(lambda x, y: (preprocess_input(x), y))
    
    # Augmentation solo en training
    if augment:
        ds_train = ds_train.map(augment_image, num_parallel_calls=AUTOTUNE)
    
    ds_train = ds_train.cache().prefetch(AUTOTUNE)
    ds_val = ds_val.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)
    
    # A numpy
    def ds_to_numpy(dset):
        X_list, y_list = [], []
        for Xb, yb in dset:
            X_list.append(Xb.numpy())
            y_list.append(yb.numpy())
        return np.concatenate(X_list, 0), np.concatenate(y_list, 0)
    
    X_train, y_train = ds_to_numpy(ds_train)
    X_test, y_test = ds_to_numpy(ds_val)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_ds_to_numpy(
    CONFIG["DATA_DIR"], 
    CONFIG["IMG_SIZE"], 
    CONFIG["BATCH"],
    CONFIG["VAL_SPLIT"],
    CONFIG["SEED"]
)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Balance Train - Benign: {(y_train==0).sum()}, Malignant: {(y_train==1).sum()}")
print(f"   Balance Test  - Benign: {(y_test==0).sum()}, Malignant: {(y_test==1).sum()}")

# ==========================================
# 2. EXTRACCIÓN DE FEATURES CON XCEPTION
# ==========================================
print("\n" + "="*60)
print("2. EXTRAYENDO FEATURES CON XCEPTION")
print("="*60)

def build_feature_extractor(img_size=224):
    """Xception feature extractor según Sección 3.3 del paper"""
    base = Xception(
        weights="imagenet", 
        include_top=False,
        input_tensor=Input(shape=(img_size, img_size, 3))
    )
    
    # Congelar capas base
    for layer in base.layers:
        layer.trainable = False
    
    # Arquitectura del paper: GAP -> Dense(1024) -> Dropout(0.5)
    x = GlobalAveragePooling2D(name="gap")(base.output)
    feat = Dense(1024, activation="relu", name="feat1024")(x)
    feat = Dropout(0.5, name="dropout")(feat)
    
    return Model(inputs=base.input, outputs=feat)

feat_model = build_feature_extractor(CONFIG["IMG_SIZE"])
print(feat_model.summary())

# Extraer features
print("\n⏳ Extrayendo features...")
Xtr_feat = feat_model.predict(X_train, batch_size=CONFIG["BATCH"], verbose=1)
Xte_feat = feat_model.predict(X_test, batch_size=CONFIG["BATCH"], verbose=1)

print(f"✅ Features extraídas: {Xtr_feat.shape}")

# Guardar features
np.save(MODELS_DIR / "Xtr_feat.npy", Xtr_feat)
np.save(MODELS_DIR / "Xte_feat.npy", Xte_feat)
np.save(MODELS_DIR / "y_train.npy", y_train)
np.save(MODELS_DIR / "y_test.npy", y_test)

# ==========================================
# 3. PSO PARA SELECCIÓN DE FEATURES
# ==========================================
print("\n" + "="*60)
print("3. OPTIMIZANDO FEATURES CON PSO")
print("="*60)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["SEED"])
feat_dim = Xtr_feat.shape[1]

# Historial de fitness para gráfica
fitness_history = []

def pso_fitness(mask):
    """Función fitness según Sección 3.4 del paper"""
    n_particles = mask.shape[0]
    losses = np.zeros(n_particles, dtype=np.float32)
    
    for i in range(n_particles):
        sel = np.where(mask[i] > 0.5)[0]
        
        if sel.size == 0:
            losses[i] = 1.0
            continue
        
        # KNN k=5 como fitness function
        pipe = make_pipeline(
            StandardScaler(), 
            KNeighborsClassifier(n_neighbors=CONFIG["K_IN_BASE"])
        )
        
        # Cross-validation accuracy
        acc = cross_val_score(
            pipe, Xtr_feat[:, sel], y_train, 
            cv=skf, scoring="accuracy", n_jobs=-1
        ).mean()
        
        # Penalty por desviación del target
        penalty = abs(sel.size - CONFIG["TARGET_DIM"]) / CONFIG["TARGET_DIM"]
        losses[i] = (1.0 - acc) + 0.05 * penalty  # Penalty reducido
    
    # Guardar mejor fitness de esta iteración
    fitness_history.append(losses.min())
    
    return losses

# Configurar PSO
options = {
    "c1": CONFIG["PSO_C1"], 
    "c2": CONFIG["PSO_C2"], 
    "w": CONFIG["PSO_W"], 
    "k": 10, 
    "p": 2
}

print(f"\n⏳ Ejecutando PSO ({CONFIG['PSO_ITERS']} iteraciones, {CONFIG['N_PARTICLES']} partículas)...")
print("⚠️ ADVERTENCIA: Esto puede tomar 2-4 horas con tu hardware")

optimizer = ps.discrete.BinaryPSO(
    n_particles=CONFIG["N_PARTICLES"], 
    dimensions=feat_dim, 
    options=options
)

best_cost, best_pos = optimizer.optimize(pso_fitness, iters=CONFIG["PSO_ITERS"], verbose=True)

selected_idx = np.where(best_pos > 0.5)[0]
print(f"\n✅ PSO completado!")
print(f"   Features seleccionadas: {selected_idx.size} (objetivo: {CONFIG['TARGET_DIM']})")
print(f"   Mejor costo: {best_cost:.4f}")

# Guardar selección
np.save(MODELS_DIR / "selected_features.npy", selected_idx)
with open(MODELS_DIR / "pso_results.pkl", "wb") as f:
    pickle.dump({"best_cost": best_cost, "best_pos": best_pos, "fitness_history": fitness_history}, f)

# Gráfica de evolución PSO
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, linewidth=2)
plt.xlabel("Iteración", fontsize=12)
plt.ylabel("Mejor Fitness", fontsize=12)
plt.title("Evolución del PSO", fontsize=14, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "pso_evolution.png", dpi=300)
plt.close()

# Aplicar selección
Xtr_sel = Xtr_feat[:, selected_idx]
Xte_sel = Xte_feat[:, selected_idx]

# ==========================================
# 4. CLASIFICACIÓN CON SUBSPACE KNN Y SVM
# ==========================================
print("\n" + "="*60)
print("4. ENTRENANDO CLASIFICADORES")
print("="*60)

def train_and_evaluate(X_train, y_train, X_test, y_test, clf, name):
    """Entrena y evalúa un clasificador"""
    print(f"\n⏳ Entrenando {name}...")
    
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    
    # Predicciones
    y_pred = pipe.predict(X_test)
    
    # Métricas
    acc = metrics.accuracy_score(y_test, y_pred)
    sen = metrics.recall_score(y_test, y_pred, pos_label=1)
    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    pre = metrics.precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, pos_label=1)
    
    results = {
        "accuracy": acc,
        "sensitivity": sen,
        "specificity": spe,
        "precision": pre,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }
    
    print(f"✅ {name} - Acc: {acc:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}, F1: {f1:.4f}")
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cm_{name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    return results, pipe

# Subspace KNN (mejor del paper)
base_knn = KNeighborsClassifier(n_neighbors=CONFIG["K_IN_BASE"])
# ✅ NUEVO (compatible con scikit-learn >= 1.2)
subspace_knn = BaggingClassifier(
    estimator=base_knn,  # ← CAMBIO AQUÍ
    n_estimators=CONFIG["N_ESTIMATORS"],
    max_features=CONFIG["SUBSPACE_FRAC"],
    bootstrap=False,
    bootstrap_features=True,
    random_state=CONFIG["SEED"],
    n_jobs=-1
)

results_subspace, model_subspace = train_and_evaluate(
    Xtr_sel, y_train, Xte_sel, y_test, 
    subspace_knn, "Subspace KNN"
)

# SVM RBF (comparación)
svm_rbf = SVC(kernel="rbf", random_state=CONFIG["SEED"])
results_svm, model_svm = train_and_evaluate(
    Xtr_sel, y_train, Xte_sel, y_test,
    svm_rbf, "SVM RBF"
)

# ==========================================
# 5. GUARDAR RESULTADOS FINALES
# ==========================================
print("\n" + "="*60)
print("5. GUARDANDO RESULTADOS")
print("="*60)

final_results = {
    "config": CONFIG,
    "data_info": {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_balance": {"benign": int((y_train==0).sum()), "malignant": int((y_train==1).sum())},
        "test_balance": {"benign": int((y_test==0).sum()), "malignant": int((y_test==1).sum())}
    },
    "pso": {
        "features_original": int(feat_dim),
        "features_selected": int(len(selected_idx)),
        "best_cost": float(best_cost)
    },
    "classifiers": {
        "Subspace_KNN": results_subspace,
        "SVM_RBF": results_svm
    }
}

with open(RESULTS_DIR / "final_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

# Guardar modelos
'''
with open(MODELS_DIR / "subspace_knn.pkl", "wb") as f:
    pickle.dump(model_subspace, f)
with open(MODELS_DIR / "svm_rbf.pkl", "wb") as f:
    pickle.dump(model_svm, f)
'''

print(f"\n✅ EXPERIMENTO COMPLETADO")
print(f"📁 Resultados guardados en: {RESULTS_DIR}")
print(f"\n🎯 MEJOR RESULTADO (Subspace KNN):")
print(f"   Accuracy:    {results_subspace['accuracy']:.4f}")
print(f"   Sensitivity: {results_subspace['sensitivity']:.4f}")
print(f"   Specificity: {results_subspace['specificity']:.4f}")
print(f"   F1-Score:    {results_subspace['f1']:.4f}")