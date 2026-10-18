# Actividad 5: MLOps, Experiment Tracking con MLflow y Comparación de Modelos

**Programa:** Máster en Inteligencia Artificial  
**Asignatura:** Gestión de Proyectos en IA  
**Actividad:** Actividad 5 — Entrenamiento, Optimización, Comparación de Modelos y Registro con MLflow  

---

## 📋 Descripción del Proyecto

Este repositorio contiene la solución completa de la **Actividad 5**, orientada a implementar un pipeline de Machine Learning estructurado bajo principios de **MLOps** y gobernanza de proyectos de inteligencia artificial.

Se aborda el problema de predicción de abandono de clientes (**Customer Churn**) utilizando el dataset oficial de **Telco Customer Churn (IBM)**. El flujo implementa:
1. **Auditoría, limpieza y reglas de negocio** sobre datos crudos.
2. **Pipelines reproducibles** de preprocesamiento para mitigar fuga de información (*data leakage*).
3. **Optimización sistemática de hiperparámetros** mediante validación cruzada estratificada (Stratified 5-Fold Grid Search).
4. **Comparación rigurosa de 3 algoritmos representativos**:
   - **Regresión Logística** (baseline lineal interpretable con regularización L1/L2).
   - **Random Forest Classifier** (ensamble no lineal por bagging).
   - **XGBoost Classifier** (ensamble secuencial por gradient boosting).
5. **Seguimiento integral de experimentos en MLflow** (parámetros, métricas estandarizadas, firmas de modelos y artefactos visuales).
6. **Reporte técnico exhaustivo y análisis de deuda técnica**.

---

## 🗂️ Estructura del Repositorio

La distribución de carpetas y archivos cumple estrictamente la especificación requerida:

```text
Actividad5/
│── datos/
│   ├── datos_ini/                              # Datos originales crudos
│   │   └── Telco-Customer-Churn.csv
│   └── datos_limp/                             # Datos transformados y limpios
│       └── Telco-Customer-Churn-Clean.csv
│── fuentes/
│   ├── entrena.ipynb                           # Notebook interactivo (Google Colab / Jupyter)
│   ├── datos_prep.py                           # Módulo de funciones de limpieza y preprocesamiento
│   └── train.py                                # Script de entrenamiento, tuning y registro en MLflow
│── evidencias/                                 # Gráficas, tablas y capturas de MLflow generadas
│   ├── cm_Regresion_Logistica.png
│   ├── cm_Random_Forest.png
│   ├── cm_XGBoost.png
│   ├── roc_Regresion_Logistica.png
│   ├── roc_Random_Forest.png
│   ├── roc_XGBoost.png
│   ├── pr_Regresion_Logistica.png
│   ├── pr_Random_Forest.png
│   ├── pr_XGBoost.png
│   ├── feature_importance_Regresion_Logistica.png
│   ├── feature_importance_Random_Forest.png
│   ├── feature_importance_XGBoost.png
│   ├── comparativa_curvas_roc.png
│   ├── comparativa_metricas_globales.png
│   ├── comparacion_modelos_metricas.csv
│   └── mlflow_dashboard_ui.png
│── .gitignore                                  # Exclusión de .venv, bases de datos y temporales
│── requirements.txt                            # Dependencias fijadas con versiones probadas
│── README.md                                   # Guía de reproducción, arquitectura y decisiones
│── CHANGELOG.md                                # Reporte técnico final y registro de versiones
└── Reporte_Tecnico_Actividad5.pdf              # Informe ejecutivo formal en PDF
```

---

## 🚀 Guía de Reproducción Rápida

### 1. Clonar el repositorio y acceder a la carpeta

```bash
git clone <URL_DEL_REPOSITORIO>
cd Actividad5
```

### 2. Crear y activar el entorno virtual

En **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En **Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el módulo de preparación de datos

Este paso audita los datos originales, aplica las reglas de negocio e imputaciones y exporta el conjunto depurado a `datos/datos_limp/`:

```bash
python fuentes/datos_prep.py
```

### 5. Ejecutar el pipeline de entrenamiento y registro en MLflow

Entrena los 3 modelos, ejecuta Grid Search con 5-Fold Stratified CV, evalúa métricas en el conjunto de prueba independiente (20%), genera las gráficas en `evidencias/` y registra todo en la base de datos de MLflow:

```bash
python fuentes/train.py
```

### 6. Explorar los resultados en el Dashboard interactivo de MLflow

Para abrir la interfaz web de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Luego abra en su navegador: `http://localhost:5000`.

### 7. Ejecutar en Jupyter Notebook / Google Colab

Puede abrir y ejecutar paso a paso el notebook documentado en:
- `fuentes/entrena.ipynb`

---

## 📊 Resumen de Resultados Obtenidos

Evaluación sobre el conjunto de prueba independiente ($N = 1,409$ clientes, 20% estratificado):

| Modelo | CV ROC-AUC (Media $\pm$ Desv) | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | $0.8464 \pm 0.0128$ | 0.8415 | **80.55%** | 65.72% | **55.88%** | **0.6040** | 0.6548 |
| **Random Forest** | $0.8442 \pm 0.0115$ | 0.8410 | 79.06% | **67.40%** | 40.91% | 0.5092 | 0.6433 |
| **XGBoost** | $\mathbf{0.8488 \pm 0.0117}$ | $\mathbf{0.8464}$ | 79.91% | 65.96% | 50.27% | 0.5706 | $\mathbf{0.6575}$ |

### Hallazgos Principales:
1. **XGBoost** obtuvo la mayor capacidad de discriminación probabilística global (**ROC-AUC = 0.8464** y **PR-AUC = 0.6575**), ideal para priorizar clientes en listas de retención comercial.
2. **Regresión Logística** demostró ser un baseline excepcionalmente competitivo, alcanzando el mayor **F1-Score (0.6040)** y mayor **Recall (55.88%)** bajo el umbral convencional de 0.50, con ventajas en explicabilidad y costo computacional nulo.
3. **Random Forest** exhibió un sesgo más conservador hacia la clase mayoritaria (menor recall, mayor precisión).

---

## ⚙️ Decisiones de Diseño y Arquitectura

1. **Prevención de Data Leakage:** El escalado (`StandardScaler`) y la codificación de variables categóricas (`OneHotEncoder`) se integran dentro de un `Pipeline` de scikit-learn acoplado al `ColumnTransformer`, asegurando que cada fold de validación cruzada y el conjunto de prueba se transformen de forma independiente.
2. **Regla de Negocio para `TotalCharges`:** Se detectaron 11 clientes con espacios en blanco en `TotalCharges`. Al cruzar con la variable `tenure`, se constató que todos tenían $0$ meses de antigüedad. La imputación lógica de negocio es fijar $0.0$, evitando eliminar filas o distorsionar con promedios.
3. **Validación Cruzada Estratificada:** Dada la asimetría de clases (26.5% abandono), se utilizó `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` para garantizar proporciones constantes de clases en cada pliegue.
4. **Backend SQLite para MLflow:** Compatible con MLflow 3.x, permite concurrencia local confiable, búsquedas SQL y compatibilidad con el Model Registry.

---

## 📖 Documentación Detallada

Para consultar el reporte técnico completo, diccionario de variables, análisis detallado de deuda técnica y gobernanza MLOps, diríjase a:
- [CHANGELOG.md](file:///c:/Users/juani/Desktop/MasterIA/Curso5.%20Gestion%20de%20Proyectos/Actividad5/CHANGELOG.md)
- [Reporte_Tecnico_Actividad5.pdf](file:///c:/Users/juani/Desktop/MasterIA/Curso5.%20Gestion%20de%20Proyectos/Actividad5/Reporte_Tecnico_Actividad5.pdf)
