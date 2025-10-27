# Replicación experimental del pipeline Xception–PSO para clasificación de melanoma (ISIC 2018)


Este repositorio contiene la implementación del estudio presentado en el artículo asociado, cuyo objetivo es evaluar el impacto del algoritmo bioinspirado Particle Swarm Optimization (PSO) combinado con la red Xception como extractor de características, considerando dos escenarios: sin balanceo de clases y con balanceo aplicado en el clasificador final.

---

## 1. Estructura del repositorio

```
PSO-SKIN-CANCER
│
├── data/                 # No incluida en el repositorio (se encuentra en una ruta externa)
├── experiments/
│   ├── experiment_3.py   # Pipeline base: Xception + PSO + clasificadores SIN balanceo de clases
│   ├── sol1.py           # Pipeline con características seleccionadas por PSO y balanceo mediante class_weight
│   └── results/          # Resultados generados automáticamente
│
├── notebooks/
│   └── figures/          # Figuras utilizadas en el paper y análisis
│
├── results/              # Carpeta con métricas, gráficos y modelos generados
├── data_cleaning.py      # Script inicial para extracción y limpieza desde un directorio externo
└── README.md
```

---

## 2. Datos externos al repositorio

Este proyecto utiliza el dataset **ISIC 2018**, el cual no se incluye en el repositorio debido a su tamaño.
El usuario debe descargar manualmente los datos desde:
[https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)

Los datos deben ubicarse **fuera del repositorio**, por ejemplo:

```
../../data/ISIC2018_Task3_Training_Input/
../../data/ISIC2018_Task3_Training_GroundTruth/
```

El script `data_cleaning.py` (versión preliminar) extrae y organiza las imágenes en formato binario (`benign` vs `malignant`), generando una carpeta local denominada `data_isic_bin/`. La versión final de limpieza utilizada para los experimentos fue ejecutada externamente.

---

## 3. Descripción de los experimentos

| Archivo           | Descripción                                                                                                                                                   | Balanceo de clases |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `experiment_3.py` | Implementa el pipeline base descrito en el estudio original: extracción con Xception, selección de características mediante PSO y clasificación sin balanceo. | No implementado    |
| `sol1.py`         | Utiliza las características seleccionadas por PSO y aplica balanceo de clases en los clasificadores (por ejemplo, `class_weight=balanced` en SVM).            | Sí implementado    |

Esta división corresponde a lo discutido en el artículo:

* **experiment_3.py** reproduce la configuración *sin balanceo*, utilizada para mostrar el impacto del desbalanceo en la detección de melanoma.
* **sol1.py** evalúa la misma selección de características, pero aplicando balanceo en el clasificador final, demostrando la mejora en sensibilidad clínica.

---

## 4. Reproducibilidad

El repositorio sigue principios de trazabilidad científica:

* Código separado por etapas.
* Resultados almacenados en carpetas con sello de tiempo.
* Configuraciones de cada experimento documentadas en archivos JSON.
* Figuras generadas automáticamente para soportar las conclusiones del estudio.

---

## 5. Ejecución


Ejecución del pipeline sin balanceo:

```bash
cd experiments
python experiment_3.py
```

Ejecución del pipeline con balanceo:

```bash
cd experiments
python sol1.py
```

---

## 6. Referencias

1. Shah et al., *Explainable AI-Based Skin Cancer Detection Using CNN, Particle Swarm Optimization and Machine Learning*, Journal of Imaging, 2024.
2. ISIC 2018 Challenge Dataset. [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
3. Repositorio del proyecto: [https://github.com/luiscarlosmarrufo/pso-skin-cancer](https://github.com/luiscarlosmarrufo/pso-skin-cancer)
