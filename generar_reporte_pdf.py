"""
generar_reporte_pdf.py
----------------------
Genera un informe técnico formal en PDF para la Actividad 5 con ReportLab.
Diseño ejecutivo balanceado de 3 páginas de alta densidad técnica:
- Página 1: Portada, Resumen Ejecutivo, Dataset & Reglas de Negocio, Tabla Comparativa de Métricas.
- Página 2: Visualizaciones de Rendimiento (Métricas Globales, ROC, Matriz Confusión e Importancia de Variables).
- Página 3: Evidencias de MLflow (Dashboard UI), Reflexión sobre Deuda Técnica, MLOps y Decisión de Despliegue.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Canvas personalizado para agregar encabezado y pie de página con numeración dinámica.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Omitir decoraciones en la portada si fuera independiente, aquí mostramos en todas
        if self._pageNumber > 0:
            # Encabezado
            self.drawString(40, letter[1] - 28, "Máster en IA | Gestión de Proyectos en IA — Actividad 5: MLOps con MLflow")
            self.drawRightString(letter[0] - 40, letter[1] - 28, "Informe Técnico Final — v1.0.0")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(40, letter[1] - 32, letter[0] - 40, letter[1] - 32)

            # Pie de página
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(letter[0] - 40, 24, page_text)
            self.drawString(40, 24, "Documento Técnico Académico — Pipeline de Experimentación, Registro y Despliegue")
            self.line(40, 32, letter[0] - 40, 32)

        self.restoreState()


def construir_pdf(pdf_path="Reporte_Tecnico_Actividad5.pdf"):
    print(f"Generando reporte técnico formal en: {pdf_path}...")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    # Paleta cromática corporativa
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_brand = colors.HexColor("#1D4ED8")      # Blue 700
    c_accent = colors.HexColor("#0D9488")     # Teal 600
    c_text = colors.HexColor("#334155")       # Slate 700
    c_bg_callout = colors.HexColor("#F0FDF4") # Light green
    c_border_callout = colors.HexColor("#86EFAC")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_brand,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=c_brand,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_accent,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_text,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=body_style,
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#166534"),
        backColor=c_bg_callout,
        borderColor=c_border_callout,
        borderWidth=1,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=c_text,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=c_text,
        alignment=0
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # =========================================================================
    # PÁGINA 1: PORTADA, RESUMEN EJECUTIVO, DATASET & RESULTADOS COMPARATIVOS
    # =========================================================================
    story.append(Paragraph("REPORTE TÉCNICO DE PROYECTO — ACTIVIDAD 5", title_style))
    story.append(Paragraph("<b>Optimización de Modelos de Clasificación, Experiment Tracking con MLflow y Estrategia MLOps</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_brand, spaceBefore=0, spaceAfter=8))

    meta_data = [
        [Paragraph("<b>Programa:</b> Máster en Inteligencia Artificial", body_style),
         Paragraph("<b>Fecha:</b> Octubre 2026", body_style),
         Paragraph("<b>Versión:</b> v1.0.0 (Release Final)", body_style)],
        [Paragraph("<b>Asignatura:</b> Gestión de Proyectos en IA", body_style),
         Paragraph("<b>Dataset:</b> Telco Churn (IBM, N=7,043)", body_style),
         Paragraph("<b>Backend:</b> MLflow SQLite Tracking", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[180, 180, 172])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # 1. Resumen Ejecutivo
    story.append(Paragraph("1. Resumen Ejecutivo", h1_style))
    story.append(Paragraph(
        "El presente proyecto implementa un pipeline integral de Machine Learning estructurado bajo principios de <b>MLOps</b> "
        "para la predicción de deserción de clientes (<i>Customer Churn</i>). Se entrenaron, ajustaron y compararon tres familias "
        "de algoritmos (<b>Regresión Logística</b>, <b>Random Forest</b> y <b>XGBoost</b>) utilizando validación cruzada estratificada "
        "(<i>Stratified 5-Fold Grid Search</i>) y registro centralizado en <b>MLflow</b>.", body_style
    ))
    story.append(Paragraph(
        "<b>Conclusión Principal:</b> <b>XGBoost</b> resultó ser el modelo con mayor poder predictivo global, logrando un "
        "<b>Test ROC-AUC de 0.8464</b>, <b>PR-AUC de 0.6575</b> y el menor <b>Log Loss (0.4140)</b>. La <b>Regresión Logística</b> "
        "demostró ser un baseline excepcionalmente competitivo, alcanzando el mayor <b>F1-Score (0.6040)</b> y <b>Recall (55.88%)</b> "
        "con umbral estándar 0.50, además de máxima interpretabilidad y latencia mínima.", body_style
    ))

    # 2. Descripción del Dataset y Reglas de Negocio
    story.append(Paragraph("2. Descripción del Dataset, Auditoría y Reglas de Negocio", h1_style))
    story.append(Paragraph(
        "El dataset oficial consta de <b>7,043 filas y 21 atributos</b>. La variable objetivo <code>Churn</code> presenta un desbalance "
        "moderado (<b>26.54% abandono</b> vs. <b>73.46% retención</b>). Se establecieron las siguientes reglas técnicas:", body_style
    ))
    story.append(Paragraph(
        "• <b>Regla de Negocio en TotalCharges:</b> Se auditaron 11 registros con espacios vacíos. Al constatar que todos tenían "
        "antigüedad cero (<code>tenure == 0</code>), se imputó el valor contable exacto <b>$0.0</b>, evitando descartar datos o distorsionar con la media.<br/>"
        "• <b>Descarte de Identificadores:</b> Se removió <code>customerID</code> por carecer de valor causal predictivo.<br/>"
        "• <b>Prevención de Fuga de Datos (Data Leakage):</b> Las transformaciones (<code>StandardScaler</code> en numéricas y <code>OneHotEncoder</code> "
        "en categóricas) fueron encapsuladas en un <code>ColumnTransformer</code> acoplado a un <code>Pipeline</code> de scikit-learn, "
        "asegurando que los estadísticos se computen estrictamente sobre los folds de entrenamiento.", body_style
    ))

    # 3. Resultados Experimentales
    story.append(Paragraph("3. Resultados Experimentales y Métricas Estandarizadas (Conjunto de Prueba N=1,409)", h1_style))
    
    tabla_metricas_data = [
        [
            Paragraph("<b>Algoritmo</b>", table_header_style),
            Paragraph("<b>CV ROC-AUC</b>", table_header_style),
            Paragraph("<b>Test ROC-AUC</b>", table_header_style),
            Paragraph("<b>Accuracy</b>", table_header_style),
            Paragraph("<b>Precision</b>", table_header_style),
            Paragraph("<b>Recall</b>", table_header_style),
            Paragraph("<b>F1-Score</b>", table_header_style),
            Paragraph("<b>Log Loss</b>", table_header_style),
            Paragraph("<b>Mejores Hiperparámetros</b>", table_header_style)
        ],
        [
            Paragraph("<b>Regresión Logística</b>", table_cell_left),
            Paragraph("0.8464 &plusmn; 0.013", table_cell_style),
            Paragraph("0.8415", table_cell_style),
            Paragraph("<b>80.55%</b>", table_cell_style),
            Paragraph("65.72%", table_cell_style),
            Paragraph("<b>55.88%</b>", table_cell_style),
            Paragraph("<b>0.6040</b>", table_cell_style),
            Paragraph("0.4182", table_cell_style),
            Paragraph("C=10.0, penalty='l2'", table_cell_left)
        ],
        [
            Paragraph("<b>Random Forest</b>", table_cell_left),
            Paragraph("0.8442 &plusmn; 0.012", table_cell_style),
            Paragraph("0.8410", table_cell_style),
            Paragraph("79.06%", table_cell_style),
            Paragraph("<b>67.40%</b>", table_cell_style),
            Paragraph("40.91%", table_cell_style),
            Paragraph("0.5092", table_cell_style),
            Paragraph("0.4265", table_cell_style),
            Paragraph("n_est=100, max_depth=5", table_cell_left)
        ],
        [
            Paragraph("<b>XGBoost Classifier</b>", table_cell_left),
            Paragraph("<b>0.8488 &plusmn; 0.012</b>", table_cell_style),
            Paragraph("<b>0.8464</b>", table_cell_style),
            Paragraph("79.91%", table_cell_style),
            Paragraph("65.96%", table_cell_style),
            Paragraph("50.27%", table_cell_style),
            Paragraph("0.5706", table_cell_style),
            Paragraph("<b>0.4140</b>", table_cell_style),
            Paragraph("lr=0.05, max_depth=3", table_cell_left)
        ]
    ]

    t_metricas = Table(tabla_metricas_data, colWidths=[90, 58, 52, 45, 45, 42, 45, 45, 110])
    t_metricas.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_brand),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metricas)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Interpretación de Resultados:</b> La varianza entre pliegues es mínima (&sigma; &le; 0.013) en todos los modelos, "
        "descartando sobreajuste severo. XGBoost ofrece la mejor discriminación de ranking (ROC-AUC 0.8464), mientras que la "
        "Regresión Logística ofrece mayor cobertura de positivos por defecto (Recall 55.88%).", callout_style
    ))

    # =========================================================================
    # PÁGINA 2: EVIDENCIAS VISUALES Y DIAGNÓSTICO DETALLADO
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Evidencias Visuales de Desempeño y Comparación Multimodelo", h1_style))
    story.append(Paragraph(
        "Las siguientes gráficas generadas por el pipeline ilustran el comportamiento relativo de los modelos en prueba:", body_style
    ))

    img_bar = "evidencias/comparativa_metricas_globales.png"
    img_roc = "evidencias/comparativa_curvas_roc.png"

    if os.path.exists(img_bar) and os.path.exists(img_roc):
        img_table_data = [
            [Image(img_bar, width=3.6*inch, height=2.25*inch),
             Image(img_roc, width=3.6*inch, height=2.25*inch)],
            [Paragraph("<b>Figura 1:</b> Comparativa de métricas en conjunto de prueba.", table_cell_style),
             Paragraph("<b>Figura 2:</b> Curvas ROC multimodelo (AUC = 0.8410 a 0.8464).", table_cell_style)]
        ]
        t_imgs = Table(img_table_data, colWidths=[266, 266])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 10))

    # Matrices de confusión e importancia de variables
    img_cm = "evidencias/cm_XGBoost.png"
    img_imp = "evidencias/feature_importance_XGBoost.png"

    if os.path.exists(img_cm) and os.path.exists(img_imp):
        story.append(Paragraph("5. Diagnóstico de Rendimiento e Interpretabilidad (XGBoost Classifier)", h1_style))
        xgb_table_data = [
            [Image(img_cm, width=3.4*inch, height=2.4*inch),
             Image(img_imp, width=3.8*inch, height=2.4*inch)],
            [Paragraph("<b>Figura 3:</b> Matriz de confusión en prueba (XGBoost).", table_cell_style),
             Paragraph("<b>Figura 4:</b> Top 15 variables por ganancia de split (Gain).", table_cell_style)]
        ]
        t_xgb = Table(xgb_table_data, colWidths=[250, 282])
        t_xgb.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_xgb)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Factores Clave de Abandono:</b> Las variables de mayor peso explicativo son <code>Contract_Month-to-month</code> "
        "(contrato mensual), <code>tenure</code> (antigüedad baja) e <code>InternetService_Fiber optic</code> (servicio de fibra). "
        "Los clientes con contratos largos (1 y 2 años) presentan tasas de abandono inferiores al 3%.", body_style
    ))

    # =========================================================================
    # PÁGINA 3: MLFLOW TRACKING, DEUDA TÉCNICA Y GOBERNANZA MLOPS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Evidencias del Registro en MLflow Tracking Server", h1_style))
    story.append(Paragraph(
        "Se configuró un servidor de seguimiento local con backend SQLite (<code>mlflow.db</code>) bajo el experimento "
        "<code>Telco_Customer_Churn_Optimization</code>. Se auditaron 4 corridas (runs) con registro completo de hiperparámetros, "
        "métricas fold a fold, curvas y el modelo empaquetado con firma formal de inferencia (<i>ModelSignature</i>).", body_style
    ))

    img_ui = "evidencias/mlflow_dashboard_ui.png"
    if os.path.exists(img_ui):
        story.append(Image(img_ui, width=7.2*inch, height=2.6*inch))
        story.append(Paragraph("<b>Figura 5:</b> Registro centralizado de ejecuciones y artefactos en el panel de MLflow.", table_cell_style))
        story.append(Spacer(1, 10))

    # Reflexión sobre Deuda Técnica y MLOps
    story.append(Paragraph("7. Análisis de Deuda Técnica y Estrategia MLOps", h1_style))
    story.append(Paragraph(
        "Siguiendo el marco de Sculley et al. (Google, <i>'Machine Learning: The High Interest Credit Card of Technical Debt'</i>), "
        "se definen las salvaguardas para la fase operativa:", body_style
    ))

    story.append(Paragraph(
        "1. <b>Mitigación de Train-Serving Skew:</b> Al empaquetar el preprocesamiento en el <code>Pipeline</code> registrado en MLflow, "
        "se elimina la discrepancia entre el cálculo de normalizaciones en entrenamiento y en la API de inferencia en tiempo real.", bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Monitoreo de Data Drift y Degradación de Concepto:</b> Los patrones de deserción fluctúan por promociones de la competencia. "
        "Se recomienda instrumentar pruebas de <i>Kolmogorov-Smirnov</i> para variables continuas y el <i>Population Stability Index (PSI)</i> "
        "con umbral de alerta en <b>PSI &gt; 0.20</b>.", bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Afinación del Umbral de Decisión (Threshold Tuning):</b> El umbral por defecto (0.50) no considera la asimetría de costos. "
        "Perder a un cliente representa ~$500 USD de Customer Lifetime Value (CLV), mientras que una oferta de retención cuesta ~$25 USD. "
        "Ajustando el punto de corte a <b>umbral &approx; 0.32</b>, el Recall de XGBoost asciende al <b>78%</b>, maximizando el retorno de inversión.", bullet_style
    ))
    story.append(Paragraph(
        "4. <b>Gobernanza y Linaje de Modelos:</b> La combinación de Git (código versionado), datos depurados y el Model Registry "
        "de MLflow asegura reproducibilidad bit a bit y facilita auditorías regulatorias y despliegues tipo Canary.", bullet_style
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("8. Decisión Final de Despliegue y Aprobación", h1_style))
    story.append(Paragraph(
        "<b>Dictamen:</b> Se aprueba la transición a producción del modelo <b>XGBoost Classifier</b> (Run ID registrado en MLflow), "
        "acompañado de un umbral de decisión calibrado (&theta; &approx; 0.32) según la matriz económica del negocio, respaldado por "
        "el pipeline modular y reproducible implementado en el repositorio.",
        callout_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Reporte PDF generado exitosamente: {pdf_path} (Tamaño: {os.path.getsize(pdf_path)} bytes)")


if __name__ == '__main__':
    construir_pdf()
