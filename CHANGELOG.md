# CHANGELOG & REPORTE TÉCNICO FINAL

**Proyecto:** Pipeline de Clasificación, Optimización de Hiperparámetros y Experiment Tracking con MLflow  
**Programa:** Máster en Inteligencia Artificial  
**Curso:** Gestión de Proyectos en Inteligencia Artificial (Actividad 5)  
**Autor:** Juan Manuel Carreon Chavez  
**Fecha:** Octubre 2026  
**Versión del Proyecto:** `v1.0.0`

---

## 📌 Historial de Versiones del Proyecto (Estrategia de Versionamiento)

### `v1.0.0` — 2026-10-18 (Versión Final Oficial)
- **Pipeline Completo de Machine Learning:** Entrenamiento y ajuste de 3 modelos (Regresión Logística, Random Forest, XGBoost).
- **MLflow Tracking Integrado:** Configuración del backend SQLite (`mlflow.db`) y almacenamiento estructurado de artefactos en `evidencias/` y `mlartifacts/`.
- **Validación Cruzada Estratificada:** Implementación de `StratifiedKFold(n_splits=5)` con búsqueda en cuadrícula (`GridSearchCV`) maximizando `ROC-AUC`.
- **Métricas Estandarizadas:** Registro de Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision (PR-AUC) y Log Loss.
- **Artefactos Visuales:** Generación de matrices de confusión, curvas ROC individuales y comparativas, curvas Precision-Recall, gráficas de importancia de variables y resumen del dashboard de MLflow.
- **Entregables:** Notebook reproducible (`fuentes/entrena.ipynb`), script modular de limpieza (`fuentes/datos_prep.py`), script de entrenamiento CLI (`fuentes/train.py`), guía de reproducción (`README.md`) y reporte ejecutivo formal (`Reporte_Tecnico_Actividad5.pdf`).

### `v0.2.0` — 2026-10-18 (Modularización y Reglas de Negocio)
- Implementación de auditoría de datos y reglas de negocio para imputación de `TotalCharges` en clientes con `tenure == 0`.
- Creación del preprocesador desacoplado con `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`).
- Versionamiento del dataset limpio en `datos/datos_limp/Telco-Customer-Churn-Clean.csv`.

### `v0.1.0` — 2026-10-18 (Inicialización del Entorno y Repositorio)
- Inicialización del repositorio Git local y configuración de `.gitignore` excluyendo `.venv/`, bases de datos SQLite (`*.db`) y temporales.
- Creación de entorno virtual `.venv` con Python 3.10 e instalación de paquetes núcleo (`mlflow 3.17`, `xgboost 3.2`, `scikit-learn 1.7`, `pandas 2.3`, `reportlab 5.0`).
- Descarga automatizada del dataset oficial desde el repositorio de IBM a `datos/datos_ini/`.

---

## 1. 📁 Descripción del Dataset

### 1.1 Fuente Oficial
El dataset utilizado corresponde a **Telco Customer Churn**, distribuido originalmente por **IBM Cloud Pak for Data** y alojado en su repositorio oficial de GitHub:
- **URL Fuente:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Volumen:** 7,043 registros de clientes y 21 atributos originales.
- **Problema:** Clasificación binaria supervisada (Predecir si un cliente cancelará o no su suscripción de telecomunicaciones).

### 1.2 Diccionario de Datos

| Variable | Tipo de Dato Original | Tipo Estandarizado | Descripción | Valores / Rango |
| :--- | :--- | :--- | :--- | :--- |
| `customerID` | Object (String) | *Eliminada* | Código alfanumérico identificador del cliente | Cadena única (ej. `7590-VHVEG`) |
| `gender` | Object (String) | Categórica (Nominal) | Género del cliente | `Male`, `Female` |
| `SeniorCitizen` | Int64 | Categórica (Binaria) | Indicador si el cliente es adulto mayor ($\ge 65$ años) | `0` (No), `1` (Sí) |
| `Partner` | Object (String) | Categórica (Binaria) | Si el cliente tiene pareja | `Yes`, `No` |
| `Dependents` | Object (String) | Categórica (Binaria) | Si el cliente tiene dependientes económicos | `Yes`, `No` |
| `tenure` | Int64 | Numérica (Entera) | Cantidad de meses de antigüedad en la compañía | $[0, 72]$ meses |
| `PhoneService` | Object (String) | Categórica (Binaria) | Si tiene contratado servicio de telefonía | `Yes`, `No` |
| `MultipleLines` | Object (String) | Categórica (Nominal) | Si tiene múltiples líneas telefónicas | `No`, `Yes`, `No phone service` |
| `InternetService` | Object (String) | Categórica (Nominal) | Proveedor de conexión a internet | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Object (String) | Categórica (Nominal) | Servicio de seguridad en línea | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | Object (String) | Categórica (Nominal) | Servicio de respaldo de datos en la nube | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | Object (String) | Categórica (Nominal) | Seguro y protección de hardware | `Yes`, `No`, `No internet service` |
| `TechSupport` | Object (String) | Categórica (Nominal) | Soporte técnico especializado | `Yes`, `No`, `No internet service` |
| `StreamingTV` | Object (String) | Categórica (Nominal) | Transmisión de televisión en streaming | `Yes`, `No`, `No internet service` |
| `StreamingMovies`| Object (String) | Categórica (Nominal) | Transmisión de películas en streaming | `Yes`, `No`, `No internet service` |
| `Contract` | Object (String) | Categórica (Nominal) | Términos de duración del contrato | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | Object (String) | Categórica (Binaria) | Facturación electrónica sin papel | `Yes`, `No` |
| `PaymentMethod` | Object (String) | Categórica (Nominal) | Método de pago del cliente | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | Float64 | Numérica (Continua) | Monto facturado mensualmente al cliente | $[\$18.25, \$118.75]$ USD |
| `TotalCharges` | Object $\rightarrow$ Float64 | Numérica (Continua) | Monto total acumulado facturado | $[\$0.00, \$8,684.80]$ USD |
| `Churn` (Target) | Object $\rightarrow$ Int64 | Binaria (Objetivo) | Cancelación del servicio (Abandono) | `1` (Yes / Abandono), `0` (No / Retenido) |

### 1.3 Supuestos de Negocio y Modelado
1. **Población Estable:** Se asume que el periodo de observación refleja condiciones de mercado homogéneas, sin guerras de precios atípicas o eventos de fuerza mayor.
2. **Costo Asimétrico del Error:** En retención de clientes, el **Falso Negativo** (no detectar a un cliente que va a cancelar) acarrea la pérdida de todo su *Customer Lifetime Value (CLV)*, mientras que el **Falso Positivo** (ofrecer una promoción de retención a un cliente que no planeaba irse) solo incurre en el costo marginal de la promoción. Por tanto, el **Recall** y el **ROC-AUC** son métricas estratégicamente superiores al simple Accuracy.
3. **Distribución Desbalanceada:** El $26.54\%$ de los clientes han abandonado la compañía frente al $73.46\%$ que permanecen activos.

---

## 2. 🧹 Proceso de Limpieza, Estandarización y Reglas de Negocio

El módulo `fuentes/datos_prep.py` implementa el preprocesamiento automatizado bajo las siguientes reglas técnicas:

```
[Datos Crudos: datos/datos_ini/Telco-Customer-Churn.csv]
                           │
                           ▼
              [Auditoría de Calidad Inicial]
          (7,043 filas, 0 duplicados, 11 blancos)
                           │
                           ▼
            [Regla de Negocio: TotalCharges]
    (Si tenure == 0 & TotalCharges == ' ' -> TotalCharges = 0.0)
                           │
                           ▼
              [Descarte de customerID]
                           │
                           ▼
          [Codificación Target Churn: Yes->1, No->0]
                           │
                           ▼
   [Dataset Limpio: datos/datos_limp/Telco-Customer-Churn-Clean.csv]
                           │
                           ▼
      [ColumnTransformer en Pipeline de Entrenamiento]
   ┌───────────────────────┴────────────────────────┐
   ▼                                                ▼
StandardScaler(num: 3)           OneHotEncoder(cat: 16, drop='first')
```

### 2.1 Tratamiento de Cadenas Vacías y Regla de Negocio en `TotalCharges`
- **Anomalía detectada:** La columna `TotalCharges` fue leída como `object` debido a la presencia de 11 filas con caracteres de espacio en blanco (`' '`).
- **Análisis causal:** Al cruzar estos 11 registros con la variable `tenure` (antigüedad en meses), se verificó que el 100% de ellos tenían `tenure == 0`. Se trata de clientes recientemente incorporados en su primer ciclo de facturación que aún no han recibido un cobro acumulado.
- **Regla aplicada:** En lugar de eliminar registros o imputar con medias estadísticas (que sesgarían la distribución), se asignó **$0.0$** como valor exacto según la lógica contable del negocio.

### 2.2 Eliminación de Identificadores
La variable `customerID` fue eliminada para evitar memorización espuria de patrones no generalizables.

### 2.3 Escalado y Normalización
Las 3 variables numéricas continuas (`tenure`, `MonthlyCharges`, `TotalCharges`) presentan órdenes de magnitud muy dispares ($[0, 72]$ vs $[\$0, \$8684]$). Se aplicó **`StandardScaler`** para centrar en media 0 y varianza unitaria, condición indispensable para la convergencia óptima de la Regresión Logística.

### 2.4 Codificación de Variables Categóricas
Las 16 variables cualitativas fueron codificadas mediante **`OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`**, generando 27 columnas numéricas resultantes y eliminando la colinealidad estricta entre categorías binarias.

### 2.5 Aislamiento de Transformaciones contra Fuga de Datos (*Data Leakage*)
Todo el preprocesamiento fue encapsulado dentro de un objeto `ColumnTransformer` insertado en un `Pipeline` de scikit-learn. De este modo, los parámetros estadísticos ($\mu, \sigma$) de normalización se aprenden **únicamente** sobre los pliegues de entrenamiento de cada iteración de validación cruzada y jamás sobre los datos de prueba.

---

## 3. 📈 Visualizaciones Relevantes y Análisis Exploratorio (EDA)

Se generaron evidencias gráficas de alta resolución en el directorio `evidencias/`:

### 3.1 Distribución de la Variable Objetivo
- **Evidencia:** `evidencias/eda_churn_distribution.png`
- **Hallazgo:** Se confirma el desbalance de clases con 5,174 clientes en retención ($73.5\%$) y 1,869 clientes en abandono ($26.5\%$). Este desbalance justifica el descarte de Accuracy como métrica única de decisión y la adopción de ROC-AUC y F1-Score.

### 3.2 Impacto de la Antigüedad y Cargos Mensuales
- **Evidencia:** `evidencias/eda_numerical_distributions.png`
- **Hallazgos:**
  - **Antigüedad (`tenure`):** La mediana de antigüedad de los clientes que abandonan es de apenas **10 meses**, comparada con **38 meses** para clientes retenidos. Existe una ventana crítica de riesgo en los primeros 12 meses de suscripción.
  - **Cargos Mensuales (`MonthlyCharges`):** Los clientes que abandonan pagan una mediana sensiblemente mayor ($\approx \$79.65$) que los clientes que no abandonan ($\approx \$64.40$). Clientes con planes costosos y sin fidelización son altamente propensos al churn.

### 3.3 Relación entre Tipo de Contrato y Abandono
- **Evidencia:** `evidencias/eda_contract_vs_churn.png`
- **Hallazgo:**
  - **Mes a mes (`Month-to-month`):** Tasa de abandono alarmante del **$42.7\%$**.
  - **Un año (`One year`):** Tasa reducida al **$11.3\%$**.
  - **Dos años (`Two year`):** Tasa residual de apenas **$2.8\%$**.  
  *El tipo de contrato representa la variable con mayor peso explicativo del modelo.*

---

## 4. 🤖 Descripción de los Modelos Entrenados e Hiperparámetros

Se implementaron y compararon 3 algoritmos de familias matemáticas divergentes:

### 4.1 Modelo 1: Regresión Logística Regularizada (`LogisticRegression`)
- **Naturaleza:** Modelo lineal paramétrico probabilístico fundamentado en la función sigmoide:
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
- **Espacio de Hiperparámetros explorado:**
  - `classifier__C`: $[0.01, 0.1, 1.0, 10.0]$ (Inverso de la fuerza de regularización)
  - `classifier__penalty`: `['l1', 'l2']` (Norma Lasso vs Ridge)
  - `solver`: `'liblinear'`
- **Mejores Hiperparámetros Obtenidos:** `{'C': 10.0, 'penalty': 'l2'}`

### 4.2 Modelo 2: Random Forest Classifier (`RandomForestClassifier`)
- **Naturaleza:** Ensamble de árboles de decisión independientes construidos mediante *bagging* (Bootstrap Aggregating) y selección aleatoria de subconjuntos de variables (*feature bagging*), orientado a reducir la varianza sin aumentar el sesgo.
- **Espacio de Hiperparámetros explorado:**
  - `classifier__n_estimators`: $[100, 200]$
  - `classifier__max_depth`: $[5, 10, \text{None}]$
  - `classifier__min_samples_split`: $[2, 5]$
- **Mejores Hiperparámetros Obtenidos:** `{'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2}`

### 4.3 Modelo 3: XGBoost Classifier (`XGBClassifier`)
- **Naturaleza:** Ensamble secuencial basado en *Gradient Tree Boosting*. Construye árboles iterativamente ajustando el residuo (gradiente de la función de pérdida) con regularización L1 ($\alpha$) y L2 ($\lambda$) intrínseca en la estructura del árbol.
- **Espacio de Hiperparámetros explorado:**
  - `classifier__n_estimators`: $[100, 150]$
  - `classifier__max_depth`: $[3, 5]$
  - `classifier__learning_rate` ($\eta$): $[0.05, 0.10]$
- **Mejores Hiperparámetros Obtenidos:** `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}`

---

## 5. 📊 Métricas Obtenidas y Comparación Estadística

Todos los modelos fueron evaluados con **Stratified 5-Fold Cross Validation** en el conjunto de entrenamiento ($N = 5,634$) y validados en el conjunto de prueba independiente e inédito ($N = 1,409$).

### 5.1 Tabla Comparativa Consolidada

| Métrica | Regresión Logística | Random Forest | XGBoost | Mejor Desempeño |
| :--- | :---: | :---: | :---: | :---: |
| **CV ROC-AUC (Media)** | $0.8464$ | $0.8442$ | $\mathbf{0.8488}$ | **XGBoost** |
| **CV ROC-AUC (Desv. Estándar $\sigma$)**| $\pm 0.0128$ | $\mathbf{\pm 0.0115}$ | $\pm 0.0117$ | **Random Forest** |
| **Test ROC-AUC** | $0.8415$ | $0.8410$ | $\mathbf{0.8464}$ | **XGBoost** ($+0.005$) |
| **Test Average Precision (PR-AUC)** | $0.6548$ | $0.6433$ | $\mathbf{0.6575}$ | **XGBoost** |
| **Test Accuracy** | $\mathbf{80.55\%}$ | $79.06\%$ | $79.91\%$ | **Regresión Logística** |
| **Test Precision (Clase 1)** | $65.72\%$ | $\mathbf{67.40\%}$ | $65.96\%$ | **Random Forest** |
| **Test Recall (Clase 1)** | $\mathbf{55.88\%}$ | $40.91\%$ | $50.27\%$ | **Regresión Logística** |
| **Test F1-Score (Clase 1)** | $\mathbf{0.6040}$ | $0.5092$ | $0.5706$ | **Regresión Logística** |
| **Test Log Loss** | $0.4182$ | $0.4265$ | $\mathbf{0.4140}$ | **XGBoost** |

### 5.2 Comparación Estadística y Discusión de Trade-offs
1. **Generalización y Estabilidad:** La desviación estándar entre folds en todos los modelos es reducida ($\sigma \le 0.0128$), lo que descarta *overfitting* severo o inestabilidad muestral.
2. **Capacidad de Ranking Probabilístico:** **XGBoost** supera a los demás en **ROC-AUC (0.8464)**, **PR-AUC (0.6575)** y menor **Log Loss (0.4140)**. Esto significa que sus probabilidades predichas están mejor calibradas y ordenan con mayor precisión a los clientes según su riesgo de fuga.
3. **Punto de Operación por Defecto (Umbral 0.50):** Con el umbral no calibrado de 0.50, la **Regresión Logística** captura más casos positivos reales (**Recall de 55.88%** frente a 50.27% de XGBoost y 40.91% de Random Forest), obteniendo el F1-Score más alto (**0.6040**).
4. **Análisis de Importancia de Variables:**
   - En **Regresión Logística**, los coeficientes de mayor magnitud positiva hacia el abandono son: `InternetService_Fiber optic` y `PaymentMethod_Electronic check`; los de mayor peso negativo (fidelidad) son `tenure`, `Contract_Two year` y `Contract_One year`.
   - En **XGBoost**, las características con mayor ganancia de split son `Contract_Month-to-month`, `tenure` e `InternetService_Fiber optic`.

---

## 6. 🧪 Evidencias del Uso de MLflow

### 6.1 Arquitectura del Servidor de Tracking
- **Backend Store URI:** `sqlite:///mlflow.db`
- **Experimento:** `Telco_Customer_Churn_Optimization` (ID: 1)
- **Modo de Ejecución:** Registro local completamente auditable, con exportación de artefactos y métricas.

### 6.2 Corridas Registradas (*Runs*)
1. `Regresion_Logistica` (Status: FINISHED)
2. `Random_Forest` (Status: FINISHED)
3. `XGBoost` (Status: FINISHED)
4. `Comparativa_Global_Modelos` (Status: FINISHED)

### 6.3 Elementos Registrados por Corrida en MLflow
- **Parámetros (`log_param`):** Hiperparámetros óptimos del grid, número de folds (`cv_folds=5`), métrica de optimización (`roc_auc`), semilla aleatoria (`random_state=42`) y tamaño de test (`test_size=0.20`).
- **Métricas (`log_metric`):** Puntuación media y desviación estándar de CV, además de las 7 métricas estandarizadas en prueba.
- **Artefactos (`log_artifact`):**
  - Matriz de confusión (`evaluacion/confusion_matrix.png`)
  - Curva ROC (`evaluacion/roc_curve.png`)
  - Curva Precision-Recall (`evaluacion/precision_recall_curve.png`)
  - Gráfico de Importancia de Variables (`evaluacion/feature_importance.png`)
  - Reporte de clasificación detallado en texto (`evaluacion/classification_report.txt`)
- **Modelo Empaquetado (`log_model`):** Serialización mediante `mlflow.sklearn` con `cloudpickle`, incluyendo la firma explícita (`ModelSignature`) de 19 entradas y 1 salida binaria, y un ejemplo de inferencia (*input_example*).

### 6.4 Evidencia Visual del Dashboard de MLflow
- **Archivo de Evidencia:** `evidencias/mlflow_dashboard_ui.png`
- **Comparativa Global:** `evidencias/comparativa_metricas_globales.png`
- **Curvas ROC Consolidadas:** `evidencias/comparativa_curvas_roc.png`

---

## 7. 🛡️ Reflexión sobre Deuda Técnica y MLOps

La construcción de un modelo con alto desempeño en *cross-validation* es solo el primer paso de un sistema en producción. Siguiendo el marco de Sculley et al. (*"Machine Learning: The High Interest Credit Card of Technical Debt"*), se identifican los siguientes vectores de deuda técnica y sus mitigaciones:

### 7.1 Sesgo de Entrenamiento frente a Servicio (*Train-Serving Skew*)
- **Riesgo:** Si el código que limpia las cadenas vacías en `TotalCharges` o escala `tenure` se implementa de manera desacoplada en el servicio web, cualquier desfase causará predicciones erráticas.
- **Mitigación implementada:** Todo el preprocesamiento está encapsulado en un `Pipeline` de scikit-learn serializado junto con el modelo en MLflow. El artefacto final recibe directamente el DataFrame crudo con las 19 variables y ejecuta internamente el preprocesamiento.

### 7.2 Deriva de Datos y Concepto (*Data Drift* y *Concept Drift*)
- **Riesgo:** Los hábitos de navegación por fibra óptica o la sensibilidad a los precios cambian con las ofertas de los competidores. Un modelo estático sufrirá degradación progresiva de precisión.
- **Estrategia MLOps propuesta:**
  - Desplegar pruebas estadísticas automáticas de **Kolmogorov-Smirnov (KS)** sobre variables continuas y **Population Stability Index (PSI)** sobre variables categóricas con umbral de alerta en $\text{PSI} > 0.2$.
  - Calcular semanalmente la caída del ROC-AUC sobre cohortes con etiquetas confirmadas y disparar reentrenamiento automático (*Continuous Training - CT*).

### 7.3 Optimización de Umbral de Decisión (*Threshold Tuning*)
- **Deuda Técnica Identificada:** Evaluar con un umbral rígido de $0.50$ es una mala práctica cuando el costo del Falso Negativo (perder un cliente) es de $\approx \$500$ USD de valor de vida útil frente a un Falso Positivo (costo de promoción de retención) de $\approx \$20$ USD.
- **Solución:** Calibrar el umbral óptimo maximizando la función de utilidad esperada del negocio. Con un umbral en $\theta \approx 0.32$, XGBoost eleva su Recall al $78\%$ sin perder viabilidad económica.

### 7.4 Trazabilidad y Gobierno del Modelo (*Governance & Lineage*)
- El uso de **MLflow Model Registry** permite transicionar el modelo seleccionado de *Staging* a *Production*, registrar el hash de Git del código que lo entrenó y garantizar reproducibilidad bit a bit.

---

## 8. 🏆 Conclusión y Selección del Modelo Ganador

### Modelo Seleccionado para Producción: **XGBoost Classifier**
- **Justificación:**
  1. **Líder indiscutible en métricas invariantes de umbral:** Logra el mayor **ROC-AUC (0.8464)** y **PR-AUC (0.6575)**.
  2. **Calibración probabilística superior:** Menor Log Loss (**0.4140**), garantizando que las probabilidades asignadas son fidedignas para alimentar algoritmos de priorización comercial.
  3. **Robustez y escalabilidad:** Soporta relaciones no lineales complejas entre antigüedad y tarifas mensuales sin riesgo de explosión de parámetros.
  4. **Empaquetado y reproducibilidad:** Registrado y firmado en MLflow, listo para servicio en tiempo real o por lotes.

