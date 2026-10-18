"""
train.py
--------
Script de entrenamiento, optimización de hiperparámetros y registro de experimentos en MLflow.
Compara múltiples modelos:
1. Regresión Logística (baseline lineal regularizado)
2. Random Forest Classifier (ensamble de árboles por bagging)
3. XGBoost Classifier (gradient boosting de alto rendimiento)

Registra:
- Parámetros del grid y mejores hiperparámetros
- Métricas estandarizadas de CV y conjunto de prueba (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss)
- Artefactos: matrices de confusión, curvas ROC, curvas PR, importancia de variables y modelos serializados
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Importar módulo de preprocesamiento local
from datos_prep import (
    cargar_datos, limpiar_datos, separar_x_y,
    obtener_definicion_columnas, crear_preprocesador, guardar_datos
)


os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def configurar_mlflow(db_name: str = "mlflow.db", experiment_name: str = "Telco_Customer_Churn_Optimization"):
    """
    Configura el tracking URI de MLflow utilizando backend SQLite (estándar recomendado en MLflow 3.x)
    con respaldo para almacenamiento local de artefactos.
    """
    abs_db_path = os.path.abspath(db_name).replace(os.sep, '/')
    tracking_uri = f"sqlite:///{abs_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking URI configurado en: {tracking_uri}")
    print(f"[MLflow] Experimento activo: {experiment_name}")
    return tracking_uri


def evaluar_modelo(pipeline, X_test, y_test):
    """
    Calcula el conjunto estándar de métricas de clasificación en test.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob))
    }
    return metricas, y_pred, y_prob


def generar_artefactos_visuales(nombre_modelo, y_test, y_pred, y_prob, pipeline, X_train, out_dir="evidencias"):
    """
    Genera y guarda en disco los artefactos visuales:
    - Matriz de confusión
    - Curva ROC
    - Curva Precision-Recall
    - Importancia de características (o coeficientes)
    - Reporte de clasificación en texto
    """
    os.makedirs(out_dir, exist_ok=True)
    archivos_artefactos = {}

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'], ax=ax)
    ax.set_title(f'Matriz de Confusión - {nombre_modelo}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"cm_{nombre_modelo}.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    archivos_artefactos["confusion_matrix"] = cm_path

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (Recall)')
    ax.set_title(f'Curva ROC - {nombre_modelo}', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"roc_{nombre_modelo}.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    archivos_artefactos["roc_curve"] = roc_path

    # 3. Curva Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color='#2ca02c', lw=2, label=f'PR curve (AP = {pr_auc_val:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Curva Precision-Recall - {nombre_modelo}', fontsize=12, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"pr_{nombre_modelo}.png")
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)
    archivos_artefactos["pr_curve"] = pr_path

    # 4. Reporte de Clasificación en Texto
    cr_text = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], digits=4)
    cr_path = os.path.join(out_dir, f"classification_report_{nombre_modelo}.txt")
    with open(cr_path, 'w', encoding='utf-8') as f:
        f.write(cr_text)
    archivos_artefactos["classification_report"] = cr_path

    # 5. Importancia de Variables / Coeficientes
    try:
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()

        importancias = None
        titulo_imp = ""
        if hasattr(classifier, 'feature_importances_'):
            importancias = classifier.feature_importances_
            titulo_imp = f'Importancia de Variables (Gini/Gain) - {nombre_modelo}'
        elif hasattr(classifier, 'coef_'):
            importancias = np.abs(classifier.coef_[0])
            titulo_imp = f'Magnitud de Coeficientes (|Beta|) - {nombre_modelo}'

        if importancias is not None and len(importancias) == len(feature_names):
            df_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importancias
            }).sort_values('importance', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=df_imp, x='importance', y='feature', palette='viridis', ax=ax)
            ax.set_title(titulo_imp, fontsize=11, fontweight='bold')
            ax.set_xlabel('Importancia')
            ax.set_ylabel('Característica')
            plt.tight_layout()
            imp_path = os.path.join(out_dir, f"feature_importance_{nombre_modelo}.png")
            fig.savefig(imp_path, dpi=300)
            plt.close(fig)
            archivos_artefactos["feature_importance"] = imp_path
    except Exception as e:
        print(f"Nota: No se pudo extraer nombres de variables para {nombre_modelo}: {e}")

    return archivos_artefactos


def ejecutar_experimentos(ruta_datos_limpios="datos/datos_limp/Telco-Customer-Churn-Clean.csv", out_dir="evidencias"):
    """
    Función principal que entrena y registra los 3 algoritmos con Grid Search y MLflow.
    """
    print("=" * 70)
    print("INICIANDO FLUJO DE MLOPS: ENTRENAMIENTO, TUNING Y REGISTRO EN MLFLOW")
    print("=" * 70)

    # 1. Cargar y preparar datos limpios
    df = pd.read_csv(ruta_datos_limpios)
    X, y = separar_x_y(df, col_target='Churn')
    cols_num, cols_cat = obtener_definicion_columnas()
    
    # Train / Test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dimensiones de entrenamiento: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Dimensiones de prueba:        X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Distribución Churn en train:  {y_train.mean():.4f}")
    print(f"Distribución Churn en test:   {y_test.mean():.4f}\n")

    # 2. Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Definición de modelos e hiperparámetros
    modelos_config = {
        "Regresion_Logistica": {
            "descripcion": "Baseline lineal regularizado con optimizador liblinear",
            "pipeline": Pipeline([
                ('preprocessor', crear_preprocesador(cols_num, cols_cat)),
                ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
            ]),
            "param_grid": {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2']
            }
        },
        "Random_Forest": {
            "descripcion": "Ensamble de bagging basado en árboles de decisión aleatorios",
            "pipeline": Pipeline([
                ('preprocessor', crear_preprocesador(cols_num, cols_cat)),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            "param_grid": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        "XGBoost": {
            "descripcion": "Gradient Boosting escalable y regularizado de última generación",
            "pipeline": Pipeline([
                ('preprocessor', crear_preprocesador(cols_num, cols_cat)),
                ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
            ]),
            "param_grid": {
                'classifier__n_estimators': [100, 150],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.05, 0.1]
            }
        }
    }

    resultados_comparativos = []
    modelos_entrenados = {}
    probs_dict = {}

    for nombre_modelo, config in modelos_config.items():
        print(f"\n>>> Optimizando y Evaluando: {nombre_modelo}...")
        
        # Grid Search con 5-Fold Stratified CV optimizando ROC-AUC
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        modelos_entrenados[nombre_modelo] = best_model

        # Métricas en test
        metricas_test, y_pred, y_prob = evaluar_modelo(best_model, X_test, y_test)
        probs_dict[nombre_modelo] = (y_test, y_prob)

        # Métricas en validación cruzada
        idx_best = grid_search.best_index_
        cv_mean_roc_auc = grid_search.cv_results_['mean_test_score'][idx_best]
        cv_std_roc_auc = grid_search.cv_results_['std_test_score'][idx_best]
        cv_train_roc_auc = grid_search.cv_results_['mean_train_score'][idx_best]

        print(f"    Mejores Parámetros: {grid_search.best_params_}")
        print(f"    CV ROC-AUC: {cv_mean_roc_auc:.4f} (+/- {cv_std_roc_auc:.4f})")
        print(f"    Test ROC-AUC: {metricas_test['roc_auc']:.4f} | F1: {metricas_test['f1_score']:.4f} | Acc: {metricas_test['accuracy']:.4f}")

        # Generar artefactos visuales
        artefactos = generar_artefactos_visuales(
            nombre_modelo, y_test, y_pred, y_prob, best_model, X_train, out_dir=out_dir
        )

        # Registro en MLflow
        with mlflow.start_run(run_name=nombre_modelo):
            # 1. Tags
            mlflow.set_tag("algorithm", nombre_modelo)
            mlflow.set_tag("descripcion", config["descripcion"])
            mlflow.set_tag("problem_type", "binary_classification")
            mlflow.set_tag("target", "Churn")
            mlflow.set_tag("dataset_version", "1.0_clean")
            mlflow.set_tag("cv_method", "StratifiedKFold_5")

            # 2. Registrar todos los hiperparámetros
            for param, val in grid_search.best_params_.items():
                mlflow.log_param(param, val)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring_metric", "roc_auc")
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)

            # 3. Registrar métricas
            mlflow.log_metric("cv_mean_roc_auc", cv_mean_roc_auc)
            mlflow.log_metric("cv_std_roc_auc", cv_std_roc_auc)
            mlflow.log_metric("cv_train_roc_auc", cv_train_roc_auc)
            for k, v in metricas_test.items():
                mlflow.log_metric(f"test_{k}", v)

            # 4. Registrar artefactos visuales y reportes
            for art_name, art_file in artefactos.items():
                mlflow.log_artifact(art_file, artifact_path="evaluacion")

            # 5. Registrar el modelo entrenado con inferencia de firma
            sample_input = X_test.head(5)
            signature = infer_signature(sample_input, best_model.predict(sample_input))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="modelo_empaquetado",
                signature=signature,
                input_example=sample_input,
                serialization_format="cloudpickle"
            )

        # Consolidar para reporte comparativo
        resultados_comparativos.append({
            "Modelo": nombre_modelo,
            "CV ROC-AUC (Media)": round(cv_mean_roc_auc, 4),
            "CV ROC-AUC (Desv)": round(cv_std_roc_auc, 4),
            "Test ROC-AUC": round(metricas_test["roc_auc"], 4),
            "Test Accuracy": round(metricas_test["accuracy"], 4),
            "Test Precision": round(metricas_test["precision"], 4),
            "Test Recall": round(metricas_test["recall"], 4),
            "Test F1-Score": round(metricas_test["f1_score"], 4),
            "Test PR-AUC": round(metricas_test["pr_auc"], 4),
            "Test LogLoss": round(metricas_test["log_loss"], 4),
            "Mejores Hiperparámetros": str(grid_search.best_params_)
        })

    # 4. Generar tabla comparativa y gráficos consolidados
    df_res = pd.DataFrame(resultados_comparativos)
    tabla_csv = os.path.join(out_dir, "comparacion_modelos_metricas.csv")
    df_res.to_csv(tabla_csv, index=False)
    print(f"\n[OK] Tabla comparativa guardada en: {tabla_csv}")
    print("\nRESUMEN COMPARATIVO FINAL:")
    print(df_res[["Modelo", "Test ROC-AUC", "Test F1-Score", "Test Accuracy", "Test Recall", "Test Precision"]].to_string(index=False))

    # Gráfico de barras comparativo de métricas
    fig, ax = plt.subplots(figsize=(10, 6))
    metricas_plot = ['Test ROC-AUC', 'Test F1-Score', 'Test Accuracy', 'Test Precision', 'Test Recall']
    df_melt = df_res.melt(id_vars=['Modelo'], value_vars=metricas_plot, var_name='Métrica', value_name='Puntaje')
    sns.barplot(data=df_melt, x='Métrica', y='Puntaje', hue='Modelo', palette='tab10', ax=ax)
    ax.set_ylim(0.4, 0.9)
    ax.set_title('Comparativa de Rendimiento en Conjunto de Prueba', fontsize=14, fontweight='bold')
    ax.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    comp_bar_path = os.path.join(out_dir, "comparativa_metricas_globales.png")
    fig.savefig(comp_bar_path, dpi=300)
    plt.close(fig)

    # Gráfico comparativo de curvas ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    colores = {'Regresion_Logistica': '#1f77b4', 'Random_Forest': '#2ca02c', 'XGBoost': '#d62728'}
    for nombre_m, (y_t, y_p) in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_val = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, label=f"{nombre_m} (AUC = {auc_val:.4f})", lw=2, color=colores.get(nombre_m, None))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)')
    ax.set_title('Comparativa de Curvas ROC Multimodelo', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    comp_roc_path = os.path.join(out_dir, "comparativa_curvas_roc.png")
    fig.savefig(comp_roc_path, dpi=300)
    plt.close(fig)

    # Registro de la corrida global de comparación en MLflow
    with mlflow.start_run(run_name="Comparativa_Global_Modelos"):
        mlflow.set_tag("stage", "model_benchmark")
        mlflow.log_artifact(tabla_csv, artifact_path="benchmark")
        mlflow.log_artifact(comp_bar_path, artifact_path="benchmark")
        mlflow.log_artifact(comp_roc_path, artifact_path="benchmark")
        for idx, row in df_res.iterrows():
            mlflow.log_metric(f"{row['Modelo']}_roc_auc", row["Test ROC-AUC"])
            mlflow.log_metric(f"{row['Modelo']}_f1", row["Test F1-Score"])

    print("\n[MLFLOW] Todos los modelos, hiperparámetros, métricas y artefactos han sido registrados exitosamente.")
    print("=" * 70)
    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenamiento y Registro de Modelos en MLflow")
    parser.add_argument("--datos", default="datos/datos_limp/Telco-Customer-Churn-Clean.csv", help="Ruta de datos limpios")
    parser.add_argument("--db_name", default="mlflow.db", help="Ruta o nombre del archivo de base de datos SQLite para MLflow")
    parser.add_argument("--out_dir", default="evidencias", help="Directorio de artefactos y evidencias")
    args = parser.parse_args()

    configurar_mlflow(db_name=args.db_name)
    ejecutar_experimentos(ruta_datos_limpios=args.datos, out_dir=args.out_dir)
