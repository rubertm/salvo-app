"""
================================================================================
SALVO Process - Step 4: Evaluate and Optimize Timing of Discrete Options
================================================================================

Sistema integrado TOPSIS + LCC para evaluación multicriterio de alternativas
de infraestructura con optimización de timing de inversión.

Proyecto: redes de acueducto
Cliente: Empresas de Acueducto
Colaboración: En consulta
Año: 2025

Versión: 2.2 (Mejorada con exportación HTML y N alternativas)

NUEVAS CARACTERÍSTICAS v2.2:
- 📥 Exportación HTML: Cada visualización puede descargarse como archivo HTML independiente
- 🔢 Tabla dinámica: Soporta N alternativas (no solo 4 fijas) con detección automática
- ✅ Valores con 1 decimal en tablas para mejor legibilidad

Características v2.1:
- 🌸 Diagrama de trébol (Sunburst) con categorías y criterios
- 📊 Tabla de valoración de subcriterios por alternativas
- 📚 Guía explicativa de componentes Life Cycle Cost

Mejoras anteriores v2.0:
- Docstrings completos en todas las funciones
- Validación de inputs con mensajes descriptivos
- Configuración centralizada
- Manejo robusto de errores
- Visualizaciones mejoradas
- Mensajes de interpretación
- Exportación a Excel multi-hoja
"""

import dash
import pandas as pd
import numpy as np
import warnings
import io
import base64
import os
#from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
from scipy.stats import spearmanr

# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================

CONFIG = {
    'project': {
        'name': 'SALVO Process',
        'client': 'Empresas de Acueducto',
        'collaboration': 'En consulta',
        'year': 2025
    },

    'topsis_monte_carlo': {
        'n_simulations': 1000,
        'noise_sigma': 0.03,
        'seed': 42
    },

    'lcc_monte_carlo': {
        'n_simulations': 500,
        'seed': 42
    },

    'defaults': {
        'discount_rate': 0.11,
        'planning_horizon': 10,
        'budget_capex': 1e9
    },

    'visualization': {
        'template': 'plotly_white',
        'default_height': 680,
        'colors': {
            'primary': '#1A5276',
            'secondary': '#5D6D7E',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'info': '#3498DB'
        }
    },

    'sheets': {
        'alternatives': 'Alternativas',
        'weights': 'Pesos',
        'assumptions': 'Supuestos',
        'evaluation': 'Evaluacion.Criterios',  # NUEVA HOJA
        'lcc_guide': 'LCC'  # NUEVA HOJA
    },

    # COLORES PARA CATEGORÍAS (SUNBURST) - Actualizados
    'category_colors': {
        'Riesgo': '#E74C3C',
        'Capital y sostenibilidad': '#F39C12',
        'Valor funcional / operación': '#3498DB',
        'Cumplimiento / normatividad': '#8E44AD',
        'Intangibles / Imagen': '#16A085',
        # Variantes alternativas por si vienen diferentes del Excel
        'Capital': '#F39C12',
        'Valor funcional': '#3498DB',
        'Cumplimiento': '#8E44AD',
        'Intangibles': '#16A085'
    }
}

# Mensajes de interpretación
INTERPRETATION_GUIDE = {
    'Diagrama de trébol': """
    🌸 INTERPRETACIÓN - Diagrama de Trébol (Sunburst)

    El diagrama de trébol muestra la estructura jerárquica del análisis SALVO:

    📊 NIVELES:
    • Centro: SALVO (100% del peso total)
    • Nivel 1: Categorías principales (5 categorías)
    • Nivel 2: Criterios intermedios
    • Nivel 3: Subcriterios individuales

    🎨 COLORES:
    • Por categoría en niveles externos
    • Verde claro = Criterios de beneficio (mayor es mejor)
    • Rojo claro = Criterios de costo (menor es mejor)

    💡 USO:
    • Tamaño de sección = Peso relativo del criterio
    • Click en sección para explorar ese nivel
    • Identifica visualmente qué criterios tienen más peso
    • Valida distribución de pesos por categoría
    """,

    'Valoración de subcriterios': """
    📊 INTERPRETACIÓN - Tabla de Evaluación de Alternativas

    Esta tabla muestra la evaluación completa de las 4 alternativas en los 21 criterios.

    📋 COLUMNAS:
    • Criterio/Subcriterio: Qué se está evaluando
    • Peso: Importancia relativa (suma = 100%)
    • Tipo: 0 = Costo (menor mejor), 1 = Beneficio (mayor mejor)
    • Profesional: Responsable de la evaluación
    • Alternativas 1, 2, 3, 4: Valores evaluados
    • Comentarios: Escala o notas explicativas

    💡 ANÁLISIS:
    • Comparar valores entre alternativas para cada criterio
    • Criterios con mayor peso (>10%) son más determinantes
    • Valores en criterios de beneficio: mayor es mejor
    • Valores en criterios de costo: menor es mejor
    • Verificar consistencia con escala en comentarios
    """,

    'Guía Life Cycle Cost': """
    📚 INTERPRETACIÓN - Guía de Componentes LCC

    Esta tabla explica los 10 componentes del análisis Life Cycle Cost.

    📦 COMPONENTES:

    💰 INVERSIÓN:
    • CapEx: Costo inicial de construcción
    • CapEx_std_pct: Incertidumbre en la inversión (±%)

    🔧 OPERACIÓN:
    • OpEx_savings_abs: Ahorros operativos anuales
    • OpEx_std_pct: Incertidumbre en ahorros (±%)

    ⚠️ RIESGO:
    • Risk_cost_avoided: Costos de riesgo evitados por año
    • Risk_std_pct: Incertidumbre en riesgos (±%)

    📈 FINANCIEROS:
    • Lifetime_years: Vida útil del proyecto
    • DiscountRate: Tasa de descuento (costo del dinero)
    • PlanningHorizon: Horizonte de análisis
    • BudgetCapEx: Presupuesto disponible

    💡 USO:
    Consultar esta tabla cuando necesites entender:
    - Qué representa cada variable LCC
    - Por qué es importante cada componente
    - Cómo interpretar los resultados NPV
    """,

    'TOPSIS baseline': """
    📊 INTERPRETACIÓN - TOPSIS Baseline

    Score TOPSIS [0-1]: Proximidad a la solución ideal
    • Score > 0.7: Alternativa fuerte (recomendada)
    • Score 0.5-0.7: Alternativa competitiva (analizar trade-offs)
    • Score < 0.5: Alternativa débil (considerar descartar)

    ⚠️ IMPORTANTE: TOPSIS es sensible a pesos de criterios.
    Revisar análisis de sensibilidad extrema.

    Probabilidad de ser mejor: Robustez bajo incertidumbre (Monte Carlo).
    """,

    'LCC: NPV by start year (timing)': """
    💰 INTERPRETACIÓN - Optimización de Timing

    NPV por año de inicio: Identifica el momento óptimo de inversión.

    Patrones comunes:
    • Pico en año X → Mejor momento para iniciar es año X
    • NPV creciente → Conviene ESPERAR (condiciones mejoran)
    • NPV decreciente → Implementar PRONTO (valor se deprecia)
    • NPV < 0 todo el horizonte → Proyecto no viable financieramente

    ⚠️ Considerar:
    - Factores no financieros (riesgos sociales, urgencia comunitaria)
    - Restricciones presupuestarias anuales
    - Oportunidades perdidas por esperar
    """,

    'Trade-off: TOPSIS vs NPV': """
    ⚖️ INTERPRETACIÓN - Trade-off Estratégico vs Financiero

    Cuadrantes:
    • Superior derecho: IDEAL (alto TOPSIS + alto NPV) → Primera opción
    • Superior izquierdo: Estratégicamente buena pero NPV negativo → Revisar supuestos LCC
    • Inferior derecho: Rentable pero estratégicamente débil → ¿Captura TOPSIS todo el valor?
    • Inferior izquierdo: EVITAR (baja en ambos criterios)

    Resolución de conflictos:
    1. ¿Criterios TOPSIS capturan valor económico de largo plazo?
    2. ¿Supuestos LCC son realistas y completos?
    3. ¿Hay beneficios/costos no monetizados?

    💡 Si hay desalineación sistemática → Revisar ponderación TOPSIS
    """,

    'Trade-Off: CapEx vs reliability': """
    🔧 INTERPRETACIÓN - Inversión vs Confiabilidad

    Relación esperada: Mayor CapEx → Mayor confiabilidad

    Análisis:
    • Pendiente positiva: Trade-off clásico (se cumple expectativa)
    • Outliers superiores: Alta confiabilidad con bajo CapEx → Alternativas eficientes
    • Outliers inferiores: Alto CapEx con baja confiabilidad → Revisar diseño

    Decisión:
    - Evaluar si el incremento en confiabilidad justifica el CapEx adicional
    - Consultar NPV: captura este trade-off en valor presente
    - Considerar restricciones presupuestarias
    """
}

warnings.filterwarnings("ignore")

# ============================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================

def validate_topsis_inputs(alts_df, pesos_df):
    """Valida estructura y consistencia de datos para análisis TOPSIS."""
    errors = []

    if "Peso" not in pesos_df.columns:
        errors.append("❌ Falta columna 'Peso' en hoja de Pesos")
    else:
        peso_total = pesos_df["Peso"].sum()
        if not (0.99 <= peso_total <= 1.01):
            errors.append(f"❌ Los pesos suman {peso_total:.4f}, deberían sumar 1.0")

    if "Criterio" not in pesos_df.columns:
        errors.append("❌ Falta columna 'Criterio' en hoja de Pesos")
    else:
        expected_criteria = set(pesos_df["Criterio"])
        actual_criteria = set(alts_df.columns)

        lcc_columns = {'Lifetime_years', 'Risk_cost_avoided_annual',
                      'OpEx_savings_abs', 'CapEx_std_pct', 'OpEx_std_pct',
                      'Risk_std_pct'}
        actual_topsis = actual_criteria - lcc_columns

        missing = expected_criteria - actual_topsis

        if missing:
            errors.append(f"❌ Criterios faltantes en Alternativas: {missing}")

    if "Tipo" not in pesos_df.columns:
        errors.append("❌ Falta columna 'Tipo' en hoja de Pesos")
    elif not pesos_df["Tipo"].isin([0, 1]).all():
        invalid = pesos_df[~pesos_df["Tipo"].isin([0, 1])]
        errors.append(f"❌ Columna 'Tipo' debe ser 0 (costo) o 1 (beneficio). " +
                     f"Valores inválidos en: {invalid['Criterio'].tolist()}")

    if alts_df.isnull().any().any():
        cols_with_nan = alts_df.columns[alts_df.isnull().any()].tolist()
        errors.append(f"⚠️ Valores faltantes (NaN) en columnas: {cols_with_nan}")

    if errors:
        raise ValueError("\n".join(["ERRORES DE VALIDACIÓN TOPSIS:"] + errors))

    return True


def validate_lcc_inputs(alts_df, asum):
    """Valida que datos requeridos para LCC existan y sean válidos."""
    errors = []

    required_cols = [
        'CapEx', 'CapEx_std_pct',
        'OpEx_savings_abs', 'OpEx_std_pct',
        'Risk_cost_avoided_annual', 'Risk_std_pct',
        'Lifetime_years'
    ]

    missing = [col for col in required_cols if col not in alts_df.columns]
    if missing:
        errors.append(f"❌ Columnas faltantes para LCC: {missing}")

    if "DiscountRate" not in asum:
        errors.append("❌ Falta 'DiscountRate' en Supuestos")
    else:
        dr = float(asum["DiscountRate"])
        if not (0 < dr < 1):
            errors.append(f"❌ DiscountRate debe estar en (0,1), valor actual: {dr}")

    if "PlanningHorizon" not in asum:
        errors.append("❌ Falta 'PlanningHorizon' en Supuestos")
    else:
        horizon = int(asum["PlanningHorizon"])
        if horizon <= 0:
            errors.append(f"❌ PlanningHorizon debe ser > 0, valor actual: {horizon}")

    if 'CapEx' in alts_df.columns:
        if (alts_df['CapEx'] < 0).any():
            neg_alts = alts_df[alts_df['CapEx'] < 0].index.tolist()
            errors.append(f"❌ CapEx negativo en alternativas: {neg_alts}")

    std_cols = ['CapEx_std_pct', 'OpEx_std_pct', 'Risk_std_pct']
    for col in std_cols:
        if col in alts_df.columns:
            if ((alts_df[col] < 0) | (alts_df[col] > 1)).any():
                errors.append(f"❌ {col} debe estar en [0,1]")

    if errors:
        raise ValueError("\n".join(["ERRORES DE VALIDACIÓN LCC:"] + errors))

    return True


# ============================================================
# FUNCIONES DE LECTURA DE DATOS
# ============================================================

def read_inputs_from_bytes(b):
    """Lee datos de Excel desde bytes con manejo robusto de errores."""
    try:
        xl = pd.ExcelFile(io.BytesIO(b))
    except Exception as e:
        raise ValueError(f"❌ Error al leer archivo Excel: {str(e)}")

    # Leer Alternativas
    sheet_alts = CONFIG['sheets']['alternatives']
    if sheet_alts not in xl.sheet_names:
        raise ValueError(f"❌ Falta hoja '{sheet_alts}' en Excel")
    alts = pd.read_excel(xl, sheet_name=sheet_alts, index_col=0)

    # Leer Pesos
    sheet_pesos = CONFIG['sheets']['weights']
    if sheet_pesos not in xl.sheet_names:
        raise ValueError(f"❌ Falta hoja '{sheet_pesos}' en Excel")
    pesos = pd.read_excel(xl, sheet_name=sheet_pesos)

    # Leer Supuestos
    sheet_sup = CONFIG['sheets']['assumptions']
    if sheet_sup not in xl.sheet_names:
        print(f"⚠️  Hoja '{sheet_sup}' no encontrada. Usando valores por defecto:")
        print(f"   - DiscountRate: {CONFIG['defaults']['discount_rate']}")
        print(f"   - PlanningHorizon: {CONFIG['defaults']['planning_horizon']}")
        print(f"   - BudgetCapEx: {CONFIG['defaults']['budget_capex']}")

        sup = pd.Series({
            'DiscountRate': CONFIG['defaults']['discount_rate'],
            'PlanningHorizon': CONFIG['defaults']['planning_horizon'],
            'BudgetCapEx': CONFIG['defaults']['budget_capex']
        })
    else:
        sup_df = pd.read_excel(xl, sheet_name=sheet_sup, index_col=0, header=None)
        sup = sup_df[1]

    # NUEVO: Leer hoja de Evaluación de Criterios (si existe)
    sheet_eval = CONFIG['sheets']['evaluation']
    eval_df = None
    if sheet_eval in xl.sheet_names:
        try:
            eval_df = pd.read_excel(xl, sheet_name=sheet_eval)
            print(f"✅ Hoja '{sheet_eval}' cargada ({len(eval_df)} filas)")
        except Exception as e:
            print(f"⚠️  Error al leer '{sheet_eval}': {str(e)}")

    # NUEVO: Leer hoja de Guía LCC (si existe)
    sheet_lcc = CONFIG['sheets']['lcc_guide']
    lcc_guide_df = None
    if sheet_lcc in xl.sheet_names:
        try:
            lcc_guide_df = pd.read_excel(xl, sheet_name=sheet_lcc)
            print(f"✅ Hoja '{sheet_lcc}' cargada ({len(lcc_guide_df)} filas)")
        except Exception as e:
            print(f"⚠️  Error al leer '{sheet_lcc}': {str(e)}")

    return alts, pesos, sup, eval_df, lcc_guide_df


def read_inputs_local(path="salvo_inputs.xlsx"):
    """Lee datos de Excel desde archivo local."""
    try:
        with open(path, 'rb') as f:
            return read_inputs_from_bytes(f.read())
    except FileNotFoundError:
        raise ValueError(f"❌ Archivo no encontrado: {path}")


# ============================================================
# FUNCIONES TOPSIS
# ============================================================

def compute_topsis(decision_df, weights, benefit_flags):
    """Calcula scores TOPSIS y matriz normalizada."""
    X = decision_df.values.astype(float)

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(X)

    for i, is_benefit in enumerate(benefit_flags):
        if not bool(is_benefit):
            norm[:, i] = 1 - norm[:, i]

    weighted = norm * weights

    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)

    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

    scores = d_worst / (d_best + d_worst + 1e-12)

    scores_series = pd.Series(scores, index=decision_df.index, name='TOPSIS_Score')
    norm_df = pd.DataFrame(norm, index=decision_df.index, columns=decision_df.columns)

    return scores_series, norm_df


def montecarlo_topsis(decision_df, weights, benefit_flags,
                     n_sim=None, noise_sigma=None, seed=None):
    """Simula scores TOPSIS bajo incertidumbre en datos."""
    n_sim = n_sim or CONFIG['topsis_monte_carlo']['n_simulations']
    noise_sigma = noise_sigma or CONFIG['topsis_monte_carlo']['noise_sigma']
    seed = seed or CONFIG['topsis_monte_carlo']['seed']

    rng = np.random.default_rng(seed)

    scaler = MinMaxScaler()
    X = decision_df.values.astype(float)
    norm_base = scaler.fit_transform(X)

    for i, is_benefit in enumerate(benefit_flags):
        if not bool(is_benefit):
            norm_base[:, i] = 1 - norm_base[:, i]

    n_alts = norm_base.shape[0]
    sims = np.zeros((n_sim, n_alts))

    for s in range(n_sim):
        noise = rng.normal(0, noise_sigma, size=norm_base.shape)
        noisy = np.clip(norm_base + noise, 0, 1)

        weighted = noisy * weights
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)
        d_best = np.linalg.norm(weighted - ideal_best, axis=1)
        d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
        sims[s, :] = d_worst / (d_best + d_worst + 1e-12)

    sims_df = pd.DataFrame(sims, columns=decision_df.index)
    mean_transform = pd.DataFrame(norm_base, index=decision_df.index,
                                  columns=decision_df.columns)

    return sims_df, mean_transform


# ============================================================
# FUNCIONES LCC (Life Cycle Cost)
# ============================================================

def lcc_timing_montecarlo(alts_df, asum, n_sims=None, seed=None):
    """Simula NPV para cada alternativa considerando incertidumbre y timing."""
    validate_lcc_inputs(alts_df, asum)

    n_sims = n_sims or CONFIG['lcc_monte_carlo']['n_simulations']
    seed = seed or CONFIG['lcc_monte_carlo']['seed']

    rng = np.random.default_rng(seed)

    dr = float(asum.get("DiscountRate", CONFIG['defaults']['discount_rate']))
    horizon = int(asum.get("PlanningHorizon", CONFIG['defaults']['planning_horizon']))

    results = {}

    for alt in alts_df.index:
        row = alts_df.loc[alt]

        capex = float(row.get("CapEx", 0.0))
        capex_std = float(row.get("CapEx_std_pct", 0.15)) * capex

        op_savings = float(row.get("OpEx_savings_abs", 0.0))
        op_std = float(row.get("OpEx_std_pct", 0.10)) * max(1.0, abs(op_savings))

        risk_avoided = float(row.get("Risk_cost_avoided_annual", 0.0))
        risk_std = float(row.get("Risk_std_pct", 0.25)) * max(1.0, abs(risk_avoided))

        lifetime = int(row.get("Lifetime_years", horizon))

        sims_npvs = np.zeros((n_sims, horizon))

        for s in range(n_sims):
            sim_capex = max(0.0, rng.normal(capex, capex_std))
            sim_op_savings = max(0.0, rng.normal(op_savings, op_std))
            sim_risk_avoided = max(0.0, rng.normal(risk_avoided, risk_std))

            for t in range(horizon):
                pv_capex = sim_capex / ((1 + dr) ** t)

                pv_benefits = 0.0
                for y in range(lifetime):
                    year = t + y

                    if year >= horizon:
                        break

                    benefit_annual = sim_op_savings + sim_risk_avoided
                    pv_benefits += benefit_annual / ((1 + dr) ** year)

                sims_npvs[s, t] = pv_benefits - pv_capex

        results[alt] = sims_npvs

    return results


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def adjust_weights(base_weights, idx, new_value):
    """Ajusta un peso específico y reescala los demás proporcionalmente."""
    w = base_weights.copy().astype(float)
    others = [i for i in range(len(w)) if i != idx]

    remaining = 1.0 - new_value

    current_sum = w[others].sum()
    if current_sum > 0:
        scale_factor = remaining / current_sum
        w[others] *= scale_factor

    w[idx] = new_value

    return w


# ============================================================
# NUEVAS FUNCIONES PARA VISUALIZACIONES
# ============================================================

def create_sunburst_chart(eval_df):
    """
    Crea gráfico Sunburst (diagrama de trébol) desde datos de evaluación.

    ESTRUCTURA CORRECTA (sin iconos en categorías):
    1. SALVO (centro)
    2. Categorías (Columna C) → SIN ICONOS
    3. Criterios (Columna D) → SIN ICONOS
    4. Subcriterios (Columna E) → CON COLORES POR TIPO

    Args:
        eval_df: DataFrame con columnas: Categoría 2, Criterio 2, Subcriterio, Peso, Tipo

    Returns:
        plotly.graph_objects.Figure: Gráfico Sunburst
    """
    # Mapeo de colores por categoría
    COLORES_CATEGORIAS = {
        "Riesgo": "#E74C3C",
        "Capital y sostenibilidad": "#F39C12",
        "Valor funcional": "#3498DB",
        "Valor funcional / operación": "#3498DB",  # Variante
        "Cumplimiento normativo": "#8E44AD",
        "Cumplimiento / normatividad": "#8E44AD",  # Variante
        "Intangibles": "#16A085",
        "Intangibles / Imagen": "#16A085"  # Variante
    }

    COLOR_COSTO = "#FADBD8"      # Rojo claro para tipo 0
    COLOR_BENEFICIO = "#D5F4E6"  # Verde claro para tipo 1

    # Renombrar columnas
    df = eval_df.rename(columns={
        'Categoría 2': 'categoria',
        'Criterio 2': 'criterio',
        'Subcriterio': 'subcriterio',
        'Peso': 'peso',
        'Tipo': 'tipo',
        'Profesional': 'profesional'
    })

    # Limpiar nulos
    df = df.dropna(subset=['categoria', 'criterio', 'subcriterio', 'peso'])

    print("="*70)
    print("📊 CREANDO SUNBURST CHART - SALVO")
    print("="*70)
    print(f"   Total criterios: {len(df)}")
    print(f"   Suma de pesos: {df['peso'].sum():.3f}")

    # Construir estructura jerárquica
    labels = ["SALVO"]
    parents = [""]
    values = [1.0]
    colors = ["#2C3E50"]
    hover_texts = ["<b>SALVO</b><br>Análisis Multicriterio TOPSIS<br>Peso Total: 100%"]

    # ========== NIVEL 1: CATEGORÍAS (SIN ICONOS) ==========
    print("\n📂 Nivel 1: Categorías")
    categorias = df.groupby('categoria')['peso'].sum().to_dict()

    for cat, peso_cat in categorias.items():
        labels.append(cat)
        parents.append("SALVO")
        values.append(peso_cat)
        colors.append(COLORES_CATEGORIAS.get(cat, "#95A5A6"))
        hover_texts.append(f"<b>{cat}</b><br>Peso: {peso_cat:.1%}")
        print(f"   ✓ {cat}: {peso_cat:.1%}")

    # ========== NIVEL 2: CRITERIOS ==========
    print("\n📋 Nivel 2: Criterios")
    criterios_dict = {}
    for _, row in df.iterrows():
        key = f"{row['categoria']}|{row['criterio']}"
        if key not in criterios_dict:
            criterios_dict[key] = {
                'categoria': row['categoria'],
                'criterio': row['criterio'],
                'peso': 0
            }
        criterios_dict[key]['peso'] += row['peso']

    for key, data in criterios_dict.items():
        cat = data['categoria']
        crit = data['criterio']
        peso = data['peso']

        labels.append(crit)
        parents.append(cat)  # ✅ Parent es la categoría SIN icono
        values.append(peso)
        colors.append(COLORES_CATEGORIAS.get(cat, "#95A5A6") + "CC")  # Con transparencia
        hover_texts.append(f"<b>{crit}</b><br>Categoría: {cat}<br>Peso: {peso:.1%}")
        print(f"   ✓ {crit}: {peso:.1%}")

    # ========== NIVEL 3: SUBCRITERIOS ==========
    print("\n📌 Nivel 3: Subcriterios")
    for _, row in df.iterrows():
        sub = row['subcriterio']
        crit = row['criterio']
        peso = row['peso']
        tipo = row['tipo']
        prof = row.get('profesional', 'N/A')

        labels.append(sub)
        parents.append(crit)  # ✅ Parent es el criterio SIN icono
        values.append(peso)

        # Color según tipo
        if tipo == 0:
            colors.append(COLOR_COSTO)
            tipo_str = "⬇️ Costo (menor mejor)"
        else:
            colors.append(COLOR_BENEFICIO)
            tipo_str = "⬆️ Beneficio (mayor mejor)"

        hover_text = (
            f"<b>{sub}</b><br>"
            f"Peso: {peso:.1%}<br>"
            f"Tipo: {tipo_str}<br>"
            f"Responsable: {prof}"
        )
        hover_texts.append(hover_text)
        print(f"   ✓ {sub[:50]}... ({peso:.1%})")

    print(f"\n📈 Total labels generados: {len(labels)}")
    print(f"   Nivel 1 (SALVO): 1")
    print(f"   Nivel 2 (Categorías): {len(categorias)}")
    print(f"   Nivel 3 (Criterios): {len(criterios_dict)}")
    print(f"   Nivel 4 (Subcriterios): {len(df)}")
    print("="*70)

    # Crear figura Sunburst
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        text=[f"{v:.1%}" for v in values],
        hovertext=hover_texts,
        hoverinfo='text',
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textfont=dict(size=11, color='#2C3E50'),
        branchvalues='total',
        insidetextorientation='radial'
    ))

    fig.update_layout(
        title="🌸 Diagrama de Trébol - Estructura Jerárquica SALVO<br>" +
              "<sub>Haga clic en las secciones para explorar niveles | Tamaño = Peso relativo</sub>",
        height=800,
        margin=dict(t=100, l=10, r=10, b=10),
        template=CONFIG['visualization']['template'],
        font=dict(family="Arial", size=12)
    )

    return fig


def create_evaluation_table(eval_df):
    """
    Crea tabla interactiva de evaluación de alternativas.

    Detecta automáticamente todas las columnas "Alternativa X" y las muestra.
    Soporta N alternativas (no solo 4 fijas).
    Formatea valores numéricos con 2 cifras decimales.

    Args:
        eval_df: DataFrame con columnas: Criterio 2, Subcriterio, Peso, Tipo,
                 Profesional, Alternativa 1, Alternativa 2, ..., Alternativa N, Comentarios

    Returns:
        plotly.graph_objects.Figure: Tabla interactiva
    """
    # Renombrar columnas básicas
    df = eval_df.rename(columns={
        'Criterio 2': 'Criterio',
        'Subcriterio': 'Subcriterio',
        'Peso': 'Peso',
        'Tipo': 'Tipo',
        'Profesional': 'Profesional',
        'Comentarios': 'Comentarios'
    })

    # Detectar automáticamente columnas de alternativas
    # Buscar columnas que empiecen con "Alternativa"
    alt_columns = [col for col in df.columns if col.startswith('Alternativa')]
    alt_columns_sorted = sorted(alt_columns, key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))

    n_alternativas = len(alt_columns_sorted)

    print(f"\n📊 Tabla de Evaluación:")
    print(f"   Alternativas detectadas: {n_alternativas}")
    print(f"   Columnas: {alt_columns_sorted}")

    # Renombrar columnas de alternativas a formato corto
    rename_dict = {}
    for col in alt_columns_sorted:
        # Extraer número de la alternativa (ej: "Alternativa 1A" -> "1A")
        alt_name = col.replace('Alternativa ', '').strip()
        rename_dict[col] = f'Alt_{alt_name}'

    df = df.rename(columns=rename_dict)

    # Limpiar nulos
    df = df.dropna(subset=['Subcriterio'])

    # Formatear peso como porcentaje
    df['Peso_Fmt'] = df['Peso'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")

    # Formatear tipo con icono
    df['Tipo_Icon'] = df['Tipo'].apply(lambda x: "⬇️" if x == 0 else "⬆️" if x == 1 else "")
    df['Tipo_Fmt'] = df['Tipo'].apply(lambda x: "Costo" if x == 0 else "Beneficio" if x == 1 else "")
    df['Tipo_Display'] = df['Tipo_Icon'] + " " + df['Tipo_Fmt']

    # FUNCIÓN PARA FORMATEAR VALORES DE ALTERNATIVAS CON 2 DECIMALES
    def formatear_valor(x):
        """
        Formatea valores: números con 2 decimales, texto sin cambios.
        """
        if pd.isna(x):
            return ""

        # Intentar convertir a float
        try:
            valor_num = float(x)
            # Si es entero pequeño (0-10), mantener sin decimales
            if valor_num == int(valor_num) and 0 <= valor_num <= 10:
                return str(int(valor_num))
            # Si tiene decimales o es número grande, formatear con 2 decimales
            else:
                return f"{valor_num:.2f}"
        except (ValueError, TypeError):
            # Si no se puede convertir a número, devolver como texto
            return str(x)

    # Formatear valores de alternativas con 2 decimales
    alt_cols_renamed = [f'Alt_{col.replace("Alternativa ", "").strip()}' for col in alt_columns_sorted]

    for col in alt_cols_renamed:
        if col in df.columns:
            df[f'{col}_Fmt'] = df[col].apply(formatear_valor)

    # Formatear profesional
    df['Profesional'] = df['Profesional'].fillna("")

    # Formatear comentarios
    df['Comentarios'] = df['Comentarios'].fillna("")

    # Crear colores alternados por fila
    n_rows = len(df)
    fill_colors_rows = ['#f8f9fa' if i % 2 == 0 else '#ffffff' for i in range(n_rows)]

    # Construir headers dinámicamente
    header_values = [
        '<b>Criterio</b>',
        '<b>Subcriterio</b>',
        '<b>Peso</b>',
        '<b>Tipo</b>',
        '<b>Profesional</b>'
    ]

    # Agregar headers de alternativas
    for col in alt_columns_sorted:
        alt_name = col.replace('Alternativa ', '').strip()
        header_values.append(f'<b>Alt {alt_name}</b>')

    header_values.append('<b>Comentarios</b>')

    # Construir cell values dinámicamente
    cell_values = [
        df['Criterio'],
        df['Subcriterio'],
        df['Peso_Fmt'],
        df['Tipo_Display'],
        df['Profesional']
    ]

    # Agregar values de alternativas formateadas
    for col in alt_cols_renamed:
        if f'{col}_Fmt' in df.columns:
            cell_values.append(df[f'{col}_Fmt'])
        else:
            cell_values.append(df.get(col, [''] * n_rows))

    cell_values.append(df['Comentarios'])

    # Calcular ancho de columnas dinámicamente
    # Base: Criterio(15) + Subcriterio(25) + Peso(8) + Tipo(10) + Prof(12) + Comentarios(30) = 100
    # Espacio para alternativas: ~8 por cada una
    n_total_cols = 5 + n_alternativas + 1
    columnwidth = [15, 25, 8, 10, 12] + [8] * n_alternativas + [30]

    # Alineación dinámica
    align = ['left', 'left', 'center', 'center', 'left'] + ['center'] * n_alternativas + ['left']

    # Crear tabla con configuración dinámica
    fig = go.Figure(data=[go.Table(
        columnwidth=columnwidth,
        header=dict(
            values=header_values,
            fill_color='#667eea',
            align=align,
            font=dict(color='white', size=12, family='Arial'),
            height=40
        ),
        cells=dict(
            values=cell_values,
            fill_color=[fill_colors_rows] * n_total_cols,
            align=align,
            font=dict(size=11, family='Arial', color='#2C3E50'),
            height=35,
            line=dict(color='#e2e8f0', width=1)
        )
    )])

    # Calcular altura dinámica basada en número de filas
    altura_tabla = min(800, max(400, n_rows * 35 + 100))

    fig.update_layout(
        title="📊 Evaluación de Alternativas - Valoración por Subcriterios<br>" +
              f"<sub>{len(df)} criterios evaluados | {n_alternativas} alternativas comparadas | Valores con 2 decimales</sub>",
        height=altura_tabla,
        margin=dict(t=80, l=10, r=10, b=10),
        template=CONFIG['visualization']['template']
    )

    return fig

    # Calcular altura dinámica basada en número de filas
    altura_tabla = min(800, max(400, n_rows * 35 + 100))

    fig.update_layout(
        title="📊 Evaluación de Alternativas - Valoración por Subcriterios<br>" +
              f"<sub>{len(df)} criterios evaluados | 4 alternativas comparadas | Valores con 2 decimales</sub>",
        height=altura_tabla,
        margin=dict(t=80, l=10, r=10, b=10),
        template=CONFIG['visualization']['template']
    )

    return fig


def create_lcc_guide_table(lcc_df):
    """
    Crea tabla explicativa de componentes LCC.

    Args:
        lcc_df: DataFrame con columnas: Componente, Significado Simple,
                ¿Qué es?, Por Qué Importa

    Returns:
        plotly.graph_objects.Figure: Tabla explicativa
    """
    # Crear tabla
    fig = go.Figure(data=[go.Table(
        columnwidth=[15, 20, 35, 35],
        header=dict(
            values=['<b>Componente</b>', '<b>Significado Simple</b>',
                   '<b>¿Qué es?</b>', '<b>Por Qué Importa</b>'],
            fill_color='#667eea',
            align='left',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[
                lcc_df['Componente'],
                lcc_df['Significado Simple'],
                lcc_df['¿Qué es?'],
                lcc_df['Por Qué Importa']
            ],
            fill_color='#ffffff',
            align='left',
            font=dict(size=11),
            height=60,
            line=dict(color='#e2e8f0', width=1)
        )
    )])

    fig.update_layout(
        title="📚 Guía de Componentes Life Cycle Cost (LCC)<br>" +
              f"<sub>{len(lcc_df)} componentes explicados | Referencia para análisis económico</sub>",
        height=CONFIG['visualization']['default_height'],
        template=CONFIG['visualization']['template']
    )

    return fig


# ============================================================
# APLICACIÓN DASH
# ============================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = dash.Dash(__name__, assets_folder=os.path.join(_BASE_DIR, 'assets'))
server = app.server

# Layout de la aplicación
app.layout = html.Div([
    # Encabezado principal
    html.Div([
        html.H2(
            "SALVO - Step 4: Evaluate and Optimize Timing of Discrete Options",
            style={'color': CONFIG['visualization']['colors']['primary'],
                  'textAlign': 'center', 'marginBottom': '5px', 'fontSize': 24,
                  'fontWeight': 'bold'}
        ),
        html.H4(
            "Paso 4: Evaluar y Optimizar el Momento de Implementación de Las Opciones Discretas",
            style={'color': CONFIG['visualization']['colors']['secondary'],
                  'textAlign': 'center', 'fontWeight': 'normal',
                  'marginTop': '5px', 'marginBottom': '5px', 'fontSize': 16,
                  'fontStyle': 'italic'}
        ),
        html.H5(
            f"{CONFIG['project']['client']} + {CONFIG['project']['collaboration']} | Strategic Sectoral Cooperation Project",
            style={'color': CONFIG['visualization']['colors']['secondary'],
                  'textAlign': 'center', 'fontWeight': 'normal',
                  'marginTop': '5px', 'marginBottom': '0px', 'fontSize': 14}
        ),
    ], style={'backgroundColor': '#EBF5FB', 'padding': '15px 20px', 'borderRadius': '10px',
             'marginBottom': '15px'}),

    # FILA 1: Carga de Datos + Logos
    html.Div([
        # Columna 1: Carga de datos
        html.Div([
            html.H4("📁 Carga de Datos",
                   style={'color': CONFIG['visualization']['colors']['primary'],
                         'marginTop': '0px', 'marginBottom': '15px'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Button("📤 Cargar Archivo Excel",
                               style={'fontSize': 15, 'padding': '12px 20px',
                                     'backgroundColor': CONFIG['visualization']['colors']['info'],
                                     'color': 'white', 'border': 'none',
                                     'borderRadius': '5px', 'cursor': 'pointer',
                                     'width': '100%', 'fontWeight': 'bold'})
                ]),
                multiple=False,
                style={'marginBottom': '10px'}
            ),
            html.Button("📂 Usar Archivo Local (salvo_inputs.xlsx)",
                       id="use-local", n_clicks=0,
                       style={'fontSize': 14, 'padding': '10px 20px',
                             'backgroundColor': CONFIG['visualization']['colors']['secondary'],
                             'color': 'white', 'border': 'none',
                             'borderRadius': '5px', 'cursor': 'pointer',
                             'width': '100%', 'marginBottom': '15px'}),
            html.Div(id='upload-status',
                    style={'fontSize': 13, 'padding': '12px',
                          'borderRadius': '5px', 'backgroundColor': '#F8F9F9',
                          'border': '1px solid #E5E7E9', 'minHeight': '60px'})
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'backgroundColor': '#FFFFFF', 'padding': '20px', 'borderRadius': '10px',
                 'border': '1px solid #D5D8DC', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

        # Columna 2: Logos
        html.Div([
            html.H4("🤝 Colaboradores",
                   style={'color': CONFIG['visualization']['colors']['primary'],
                         'marginTop': '0px', 'marginBottom': '15px', 'textAlign': 'center'}),
            html.Div([
                html.Img(src='/assets/colaborador1.png',
                        style={'height':'65px', 'margin':'5px 10px'},
                        alt='EPM'),
                html.Img(src='/assets/escudoCiudad2.png',
                        style={'height':'65px', 'margin':'5px 10px'},
                        alt='Copenhagen'),
            ], style={'textAlign': 'center', 'marginBottom': '8px'}),
            html.Div([
                html.Img(src='/assets/escudoCiudad1.png',
                        style={'height':'75px', 'margin':'5px 10px'},
                        alt='Medellín'),
                html.Img(src='/assets/colaborador2.png',
                        style={'height':'65px', 'margin':'5px 10px'},
                        alt='HOFOR')
            ], style={'textAlign': 'center', 'marginBottom': '8px'}),
            html.Div("Strategic Sectoral Cooperation",
                    style={'fontSize': 11, 'color': '#95A5A6', 'textAlign': 'center',
                          'fontStyle': 'italic'})
        ], style={'width': '45%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top',
                 'marginLeft': '4%', 'backgroundColor': '#FFFFFF',
                 'padding': '20px', 'borderRadius': '10px',
                 'border': '1px solid #D5D8DC', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'marginBottom': '15px'}),

    # FILA 2: Selector de análisis + Gráfico
    html.Div([
        # Columna 1: Selector
        html.Div([
            html.H4("📊 Tipo de Análisis",
                   style={'color': CONFIG['visualization']['colors']['primary'],
                         'marginTop': '0px', 'marginBottom': '15px'}),
            dcc.RadioItems(
                id='tipo_grafico',
                options=[
                    # NUEVAS OPCIONES AL INICIO
                    {'label': ' 🌸 Diagrama de trébol', 'value': 'Diagrama de trébol'},
                    {'label': ' 📊 Valoración de subcriterios', 'value': 'Valoración de subcriterios'},

                    # OPCIONES EXISTENTES
                    {'label': ' 🎯 TOPSIS baseline', 'value': 'TOPSIS baseline'},
                    {'label': ' 🎲 Monte Carlo - Boxplot', 'value': 'Monte Carlo - Boxplot'},
                    {'label': ' 🕸️ Radar comparativo', 'value': 'Comparative radar (normed)'},
                    {'label': ' ⚡ Sensibilidad extrema', 'value': 'Extreme sensitivity (línea+marc)'},
                    {'label': ' 🔥 Heatmap fortalezas', 'value': 'Heatmap: Strengths and weaknesses 1 - 0 (alts x crits)'},
                    {'label': ' 💰 LCC: NPV timing', 'value': 'LCC: NPV by start year (timing)'},
                    {'label': ' 📦 Boxplot LCC (NPV)', 'value': 'Boxplot LCC (NPV by alt)'},
                    {'label': ' ⚖️ Trade-off TOPSIS/NPV', 'value': 'Trade-off: TOPSIS vs NPV'},
                    {'label': ' 🔧 Trade-off CapEx', 'value': 'Trade-Off: CapEx vs reliability'},

                    # NUEVA OPCIÓN AL FINAL
                    {'label': ' 📚 Guía Life Cycle Cost', 'value': 'Guía Life Cycle Cost'}
                ],
                value='Diagrama de trébol',  # NUEVA OPCIÓN POR DEFECTO
                labelStyle={'display': 'block', 'padding': '8px 5px', 'fontSize': 14,
                           'cursor': 'pointer', 'borderRadius': '4px',
                           'marginBottom': '2px'},
                inputStyle={'marginRight': '8px'},
                style={'overflowY': 'auto', 'maxHeight': '600px'}
            )
        ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'backgroundColor': '#FFFFFF', 'padding': '20px', 'borderRadius': '10px',
                 'border': '1px solid #D5D8DC', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'height': '600px'}),

        # Columna 2: Gráfico
        html.Div([
            # Botón de exportación HTML
            html.Div([
                html.Button("📥 Exportar como HTML",
                           id='export-html-button',
                           style={'fontSize': 13, 'padding': '8px 15px',
                                 'backgroundColor': CONFIG['visualization']['colors']['success'],
                                 'color': 'white', 'border': 'none',
                                 'borderRadius': '5px', 'cursor': 'pointer',
                                 'marginBottom': '10px', 'fontWeight': 'bold',
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'textAlign': 'right', 'padding': '10px'}),

            dcc.Graph(
                id='grafica',
                config={'displayModeBar': True, 'displaylogo': False,
                       'modeBarButtonsToRemove': ['lasso2d', 'select2d']},
                style={'height': '600px', 'width': '100%'}
            ),

            # Componentes para exportación
            dcc.Download(id='download-html'),
            dcc.Store(id='current-figure-store')  # Almacena la figura actual
        ], style={'width': '75%', 'float': 'right', 'display': 'inline-block', 'marginLeft': '2%',
                 'verticalAlign': 'top', 'backgroundColor': '#FFFFFF',
                 'borderRadius': '10px', 'border': '1px solid #D5D8DC',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '0px'}),
    ], style={'marginBottom': '15px'}),

    # Interpretación
    html.Div(id='texto_salida',
            style={'whiteSpace': 'pre-wrap', 'padding': '20px',
                  'backgroundColor': '#F8F9F9', 'borderRadius': '10px',
                  'fontSize': 13, 'fontFamily': 'Consolas, Monaco, monospace',
                  'border': '1px solid #D5D8DC', 'lineHeight': '1.6',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'})
], style={'maxWidth': '1650px', 'margin': '0 auto', 'padding': '20px',
         'backgroundColor': '#F4F6F7', 'minHeight': '100vh'})

# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    Input('use-local', 'n_clicks'),
    State('upload-data', 'filename')
)
def update_upload_status(contents, nclicks, filename):
    """Muestra estado de carga de archivo."""
    if contents is None and nclicks == 0:
        return html.Div("⏳ Esperando carga de archivo...",
                       style={'color': CONFIG['visualization']['colors']['secondary']})

    try:
        if contents:
            content_string = contents.split(",")[1]
            decoded = base64.b64decode(content_string)
            alts_df, pesos_df, sup, eval_df, lcc_df = read_inputs_from_bytes(decoded)
        else:
            alts_df, pesos_df, sup, eval_df, lcc_df = read_inputs_local()

        validate_topsis_inputs(alts_df, pesos_df)
        validate_lcc_inputs(alts_df, sup.to_dict())

        n_alts = len(alts_df)
        n_crit = len(pesos_df)

        status_parts = [
            html.Strong("✅ Archivo cargado exitosamente",
                       style={'color': CONFIG['visualization']['colors']['success']}),
            html.Div(f"📊 {n_alts} alternativas, {n_crit} criterios"),
            html.Div(f"💰 Horizonte: {int(sup.get('PlanningHorizon', 10))} años | " +
                    f"Tasa: {float(sup.get('DiscountRate', 0.11))*100:.0f}%")
        ]

        # Indicar hojas adicionales cargadas
        if eval_df is not None:
            status_parts.append(
                html.Div(f"🌸 Evaluación: {len(eval_df)} criterios",
                        style={'color': CONFIG['visualization']['colors']['success']})
            )

        if lcc_df is not None:
            status_parts.append(
                html.Div(f"📚 Guía LCC: {len(lcc_df)} componentes",
                        style={'color': CONFIG['visualization']['colors']['success']})
            )

        return html.Div(status_parts)

    except Exception as e:
        return html.Div([
            html.Strong("❌ Error: ",
                       style={'color': CONFIG['visualization']['colors']['danger']}),
            html.Pre(str(e), style={'color': CONFIG['visualization']['colors']['danger'],
                                   'whiteSpace': 'pre-wrap'})
        ])


@app.callback(
    [Output('grafica', 'figure'),
     Output('texto_salida', 'children')],
    [Input('tipo_grafico', 'value'),
     Input('upload-data', 'contents'),
     Input('use-local', 'n_clicks')],
    State('upload-data', 'filename')
)
def update_graph(tipo_grafico, contents, nclicks, filename):
    """Callback principal: genera visualizaciones según análisis seleccionado."""
    try:
        # Cargar datos
        if contents:
            content_string = contents.split(",")[1]
            decoded = base64.b64decode(content_string)
            alts_df, pesos_df, sup, eval_df, lcc_df = read_inputs_from_bytes(decoded)
        else:
            alts_df, pesos_df, sup, eval_df, lcc_df = read_inputs_local()

        # ========== NUEVAS VISUALIZACIONES ==========

        # DIAGRAMA DE TRÉBOL
        if tipo_grafico == 'Diagrama de trébol':
            if eval_df is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"❌ Hoja '{CONFIG['sheets']['evaluation']}' no encontrada en Excel",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=16, color='red')
                )
                return fig, "La hoja de evaluación de criterios no está disponible."

            fig = create_sunburst_chart(eval_df)
            interpretation = INTERPRETATION_GUIDE.get('Diagrama de trébol', '')

            texto = f"🌸 Diagrama de Trébol generado\n"
            texto += f"📊 {len(eval_df)} criterios organizados jerárquicamente\n"
            texto += f"🎨 Colores por categoría y tipo de criterio\n\n"
            texto += interpretation

            return fig, texto

        # VALORACIÓN DE SUBCRITERIOS
        elif tipo_grafico == 'Valoración de subcriterios':
            if eval_df is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"❌ Hoja '{CONFIG['sheets']['evaluation']}' no encontrada en Excel",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=16, color='red')
                )
                return fig, "La hoja de evaluación de criterios no está disponible."

            fig = create_evaluation_table(eval_df)
            interpretation = INTERPRETATION_GUIDE.get('Valoración de subcriterios', '')

            texto = f"📊 Tabla de Evaluación generada\n"
            texto += f"✅ {len(eval_df)} criterios evaluados\n"
            texto += f"🎯 4 alternativas comparadas\n\n"
            texto += interpretation

            return fig, texto

        # GUÍA LIFE CYCLE COST
        elif tipo_grafico == 'Guía Life Cycle Cost':
            if lcc_df is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"❌ Hoja '{CONFIG['sheets']['lcc_guide']}' no encontrada en Excel",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=16, color='red')
                )
                return fig, "La hoja de guía LCC no está disponible."

            fig = create_lcc_guide_table(lcc_df)
            interpretation = INTERPRETATION_GUIDE.get('Guía Life Cycle Cost', '')

            texto = f"📚 Guía de Componentes LCC\n"
            texto += f"📦 {len(lcc_df)} componentes explicados\n"
            texto += f"💡 Referencia para análisis económico\n\n"
            texto += interpretation

            return fig, texto

        # ========== ANÁLISIS TOPSIS Y LCC EXISTENTES ==========

        # Preparar datos para análisis existentes
        crits = pesos_df["Criterio"].tolist()
        weights = pesos_df["Peso"].values
        benefit_flags = pesos_df["Tipo"].values
        asum = sup.to_dict()

        # Calcular TOPSIS base
        scores, norm_df = compute_topsis(alts_df[crits], weights, benefit_flags)

        # Calcular Monte Carlo TOPSIS
        sims_df, mean_transform = montecarlo_topsis(alts_df[crits], weights, benefit_flags)

        # Calcular LCC
        lcc_results = lcc_timing_montecarlo(alts_df, asum)

        # Probabilidad de ser mejor
        best_counts = (sims_df.eq(sims_df.max(axis=1), axis=0)).sum()
        prob_best = (best_counts / sims_df.shape[0] * 100).round(2)

        prob_text = "\n\n📊 Probabilidad (%) de ser mejor alternativa (Monte Carlo):\n" + \
                   prob_best.to_string()

        interpretation = INTERPRETATION_GUIDE.get(tipo_grafico, "")

        # TOPSIS BASELINE
        if tipo_grafico == "TOPSIS baseline":
            fig = px.bar(
                x=scores.index, y=scores.values,
                title=f"TOPSIS Baseline - {CONFIG['project']['name']}<br>" +
                      f"<sub>{len(crits)} criterios | Scores [0-1] | Mayor es mejor</sub>",
                labels={'x': 'Alternativa', 'y': 'Score TOPSIS'},
                template=CONFIG['visualization']['template']
            )
            fig.update_traces(marker_color=CONFIG['visualization']['colors']['info'])
            fig.update_layout(height=CONFIG['visualization']['default_height'])

            texto = "🏆 Ranking TOPSIS:\n" + scores.sort_values(ascending=False).round(3).to_string()
            texto += prob_text + "\n\n" + interpretation

            return fig, texto

        # MONTE CARLO BOXPLOT
        elif tipo_grafico == "Monte Carlo - Boxplot":
            m = sims_df.melt(var_name="Alternativa", value_name="Score TOPSIS")
            fig = px.box(
                m, x="Alternativa", y="Score TOPSIS", color="Alternativa",
                title=f"Análisis Monte Carlo - Distribución de Scores<br>" +
                      f"<sub>{CONFIG['topsis_monte_carlo']['n_simulations']} simulaciones | " +
                      f"Ruido: ±{CONFIG['topsis_monte_carlo']['noise_sigma']*100:.0f}%</sub>",
                template=CONFIG['visualization']['template']
            )
            fig.update_layout(showlegend=False,
                            height=CONFIG['visualization']['default_height'])

            stats = pd.DataFrame({
                'Mean': sims_df.mean(),
                'Std': sims_df.std(),
                'P25': sims_df.quantile(0.25),
                'Median': sims_df.median(),
                'P75': sims_df.quantile(0.75)
            }).round(3)

            texto = "📊 Estadísticas Monte Carlo:\n" + stats.to_string()
            texto += prob_text + "\n\n" + interpretation

            return fig, texto

        # RADAR COMPARATIVO
        elif tipo_grafico == "Comparative radar (normed)":
            fig = go.Figure()
            for alt in mean_transform.index:
                fig.add_trace(go.Scatterpolar(
                    r=mean_transform.loc[alt].values,
                    theta=mean_transform.columns,
                    fill='toself',
                    name=alt
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Radar Comparativo - Perfil Normalizado por Criterios",
                showlegend=True,
                height=CONFIG['visualization']['default_height'],
                template=CONFIG['visualization']['template']
            )

            texto = "🎯 Perfil de fortalezas (valores normalizados [0-1])"
            texto += prob_text + "\n\n" + interpretation

            return fig, texto

        # SENSIBILIDAD EXTREMA
        elif tipo_grafico == "Extreme sensitivity (línea+marc)":
            scenarios, scenario_order = {}, []

            for i, crit in enumerate(crits):
                for extreme in [0.05, 0.50]:
                    scen_name = f"{crit[:20]}={int(extreme*100)}%"
                    scenario_order.append(scen_name)

                    new_w = adjust_weights(weights, i, extreme)
                    s, _ = compute_topsis(alts_df[crits], new_w, benefit_flags)
                    scenarios[scen_name] = s

            df_scen = pd.DataFrame(scenarios)
            df_long = df_scen.reset_index().melt(
                id_vars="index", var_name="Escenario", value_name="Score TOPSIS"
            ).rename(columns={"index": "Alternativa"})

            fig = px.line(
                df_long, x="Escenario", y="Score TOPSIS",
                color="Alternativa", markers=True,
                category_orders={"Escenario": scenario_order},
                title="Análisis de Sensibilidad Extrema - Variación de Pesos",
                template=CONFIG['visualization']['template']
            )
            fig.update_layout(height=CONFIG['visualization']['default_height'],
                            xaxis_tickangle=-45)
            fig.update_xaxes(tickfont=dict(size=9))

            texto = "⚡ Análisis de sensibilidad completado\n"
            texto += "Escenarios: Peso de cada criterio al 5% y 50%\n"
            texto += "Cruces de líneas indican alta sensibilidad"

            return fig, texto

        # HEATMAP
        elif tipo_grafico == "Heatmap: Strengths and weaknesses 1 - 0 (alts x crits)":
            fig = go.Figure(data=go.Heatmap(
                z=norm_df.values,
                x=[c[:25] for c in norm_df.columns],
                y=norm_df.index,
                colorscale="Viridis",
                text=norm_df.round(2).astype(str),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Valor<br>Normalizado")
            ))

            fig.update_layout(
                title="Heatmap de Fortalezas y Debilidades<br>" +
                      "<sub>1 = Mejor valor normalizado | 0 = Peor valor normalizado</sub>",
                height=max(500, len(norm_df) * 100),
                template=CONFIG['visualization']['template']
            )
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))

            texto = "🔥 Heatmap de desempeño normalizado\n"
            texto += "Verde = Fortaleza | Morado = Debilidad"

            return fig, texto

        # LCC TIMING
        elif tipo_grafico == "LCC: NPV by start year (timing)":
            horizon = int(asum.get("PlanningHorizon", 10))

            fig = go.Figure()
            for alt, arr in lcc_results.items():
                mean_by_t = arr.mean(axis=0)
                std_by_t = arr.std(axis=0)

                fig.add_trace(go.Scatter(
                    x=list(range(horizon)),
                    y=mean_by_t,
                    mode='lines+markers',
                    name=alt,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title=f"Optimización de Timing LCC - NPV Esperado por Año de Inicio<br>" +
                      f"<sub>Horizonte: {horizon} años | Tasa: {float(asum.get('DiscountRate', 0.11))*100:.0f}%</sub>",
                xaxis_title="Año de Inicio de Inversión",
                yaxis_title="NPV Esperado (millones COP)",
                hovermode='x unified',
                height=CONFIG['visualization']['default_height'],
                template=CONFIG['visualization']['template']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red",
                         annotation_text="NPV = 0")

            optimal_years = {}
            for alt, arr in lcc_results.items():
                mean_npv = arr.mean(axis=0)
                optimal_year = np.argmax(mean_npv)
                optimal_years[alt] = (optimal_year, mean_npv[optimal_year])

            texto = "💰 Años óptimos de inversión:\n"
            for alt, (year, npv) in optimal_years.items():
                texto += f"  • {alt}: Año {year} (NPV={npv:.0f}M)\n"

            texto += "\n" + interpretation

            return fig, texto

        # BOXPLOT LCC
        elif tipo_grafico == "Boxplot LCC (NPV by alt)":
            data = []
            for alt, arr in lcc_results.items():
                npv0 = arr[:, 0]
                data.append(pd.Series(npv0, name=alt))

            dfm = pd.concat(data, axis=1).melt(var_name="Alternativa", value_name="NPV")

            fig = px.box(
                dfm, x="Alternativa", y="NPV", color="Alternativa",
                title=f"Distribución NPV por Alternativa (inversión en t=0)<br>" +
                      f"<sub>{CONFIG['lcc_monte_carlo']['n_simulations']} simulaciones</sub>",
                labels={'NPV': 'NPV (millones COP)'},
                template=CONFIG['visualization']['template']
            )
            fig.update_layout(showlegend=False,
                            height=CONFIG['visualization']['default_height'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")

            stats_npv = dfm.groupby('Alternativa')['NPV'].agg(['mean', 'std', 'median']).round(0)

            texto = "📊 Estadísticas NPV (t=0):\n" + stats_npv.to_string()

            return fig, texto

        # TRADE-OFF: TOPSIS vs NPV
        elif tipo_grafico == "Trade-off: TOPSIS vs NPV":
            mean_npvs = {alt: arr[:, 0].mean() for alt, arr in lcc_results.items()}

            df_trade = pd.DataFrame({
                "TOPSIS": scores,
                "NPV": pd.Series(mean_npvs)
            })

            fig = px.scatter(
                df_trade, x="NPV", y="TOPSIS",
                text=df_trade.index,
                size=df_trade["NPV"].abs() + 1,
                title="Trade-off: Viabilidad Estratégica (TOPSIS) vs Financiera (NPV)",
                labels={'NPV': 'NPV (millones COP)', 'TOPSIS': 'Score TOPSIS [0-1]'},
                template=CONFIG['visualization']['template']
            )
            fig.update_traces(textposition="top center", textfont_size=12)
            fig.update_layout(height=CONFIG['visualization']['default_height'])

            fig.add_hline(y=scores.median(), line_dash="dash",
                         annotation_text="Mediana TOPSIS",
                         annotation_position="right")
            fig.add_vline(x=0, line_dash="dash", line_color="red",
                         annotation_text="NPV = 0")

            fig.add_annotation(
                x=df_trade["NPV"].max() * 0.7,
                y=df_trade["TOPSIS"].max() * 0.9,
                text="<b>Ideal</b><br>Alto TOPSIS + Alto NPV",
                showarrow=False,
                bgcolor="rgba(0,255,0,0.2)",
                bordercolor="green",
                font=dict(size=11)
            )

            texto = "⚖️ Análisis Trade-off TOPSIS vs NPV\n\n"
            for alt in df_trade.index:
                texto += f"  • {alt}: TOPSIS={df_trade.loc[alt, 'TOPSIS']:.3f}, " +\
                        f"NPV={df_trade.loc[alt, 'NPV']:.0f}M\n"

            texto += "\n" + interpretation

            return fig, texto

        # TRADE-OFF: CAPEX vs RELIABILITY
        elif tipo_grafico == "Trade-Off: CapEx vs reliability":
            if "CapEx" in alts_df.columns and "Risk_cost_avoided_annual" in alts_df.columns:
                fig = px.scatter(
                    alts_df, x="CapEx", y="Risk_cost_avoided_annual",
                    text=alts_df.index,
                    size="CapEx",
                    color="Risk_cost_avoided_annual",
                    title="Trade-off: Inversión Inicial (CapEx) vs Confiabilidad<br>" +
                          "<sub>Confiabilidad medida como costos de riesgo evitados</sub>",
                    labels={
                        'CapEx': 'CapEx (millones COP)',
                        'Risk_cost_avoided_annual': 'Costos Riesgo Evitados Anual (millones COP)'
                    },
                    template=CONFIG['visualization']['template']
                )
                fig.update_traces(textposition="top center", textfont_size=12)
                fig.update_layout(height=CONFIG['visualization']['default_height'])

                texto = "🔧 Trade-off CapEx vs Confiabilidad"
                texto += "\n" + interpretation

                return fig, texto
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="❌ Faltan columnas requeridas:<br>CapEx y Risk_cost_avoided_annual",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=16, color='red')
                )
                return fig, "Columnas requeridas no encontradas en datos."

        # Default
        return go.Figure(), "Opción no implementada"

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ Error en análisis:<br>{str(e)}",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color='red')
        )
        return fig, f"ERROR:\n{str(e)}"


@app.callback(
    Output('download-html', 'data'),
    Input('export-html-button', 'n_clicks'),
    [State('grafica', 'figure'),
     State('tipo_grafico', 'value'),
     State('upload-data', 'filename')],  # NUEVO: nombre del archivo Excel
    prevent_initial_call=True
)
def export_figure_as_html(n_clicks, figure_dict, tipo_grafico, excel_filename):
    """
    Exporta la visualización actual como archivo HTML descargable.

    Args:
        n_clicks: Número de clicks en el botón (trigger del callback)
        figure_dict: Diccionario con la figura actual de Plotly
        tipo_grafico: Tipo de análisis seleccionado (para el nombre del archivo)
        excel_filename: Nombre del archivo Excel cargado (opcional)

    Returns:
        dict: Datos para descargar el archivo HTML
    """
    if n_clicks is None or n_clicks == 0:
        return None

    try:
        # Convertir el diccionario a una figura de Plotly
        fig = go.Figure(figure_dict)

        # Generar nombre de archivo limpio
        # Remover caracteres especiales y espacios
        nombre_limpio = tipo_grafico.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace('/', '_')
        nombre_archivo = f"salvo_{nombre_limpio}.html"

        # Generar HTML completo con encabezado mejorado
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determinar qué mostrar en la línea de proyecto
        if excel_filename:
            # Si hay archivo Excel cargado, mostrar su nombre
            proyecto_info = f"<strong>📁 Archivo Excel:</strong> {excel_filename}<br>"
        else:
            # Si se usó archivo local, mostrar mensaje genérico o no mostrar nada
            proyecto_info = ""

        html_header = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SALVO - {tipo_grafico}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #F8F9F9 0%, #EBF5FB 100%);
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #1A5276 0%, #3498DB 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .subtitle {{
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.9;
        }}
        .info {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #3498DB;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #7F8C8D;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌟 SALVO Process - Análisis Multicriterio</h1>
        <div class="subtitle">EPM Medellín + Copenhagen (HOFOR) | Strategic Sectoral Cooperation 2025</div>
    </div>

    <div class="info">
        <strong>📊 Tipo de Análisis:</strong> {tipo_grafico}<br>
        <strong>📅 Fecha de Exportación:</strong> {fecha_actual}<br>
        {proyecto_info}
    </div>

    <div class="chart-container">
"""

        html_footer = """
    </div>

    <div class="footer">
        <p>Generado por SALVO Process - Sistema integrado TOPSIS + LCC para evaluación multicriterio</p>
        <p>© 2025 EPM Medellín - Copenhagen (HOFOR) Collaboration</p>
    </div>
</body>
</html>
"""

        # Convertir figura a HTML (solo el div del gráfico)
        fig_html = fig.to_html(
            include_plotlyjs='cdn',  # Usar CDN para reducir tamaño
            div_id='grafico-salvo',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        # Extraer solo el div del gráfico (sin <html>, <head>, <body>)
        # El método to_html con include_plotlyjs='cdn' ya genera un HTML completo,
        # pero queremos integrarlo en nuestra plantilla

        # Mejor opción: generar HTML completo personalizado
        html_completo = html_header + fig_html + html_footer

        print(f"\n📥 Exportando visualización:")
        print(f"   Tipo: {tipo_grafico}")
        print(f"   Archivo: {nombre_archivo}")
        if excel_filename:
            print(f"   Excel fuente: {excel_filename}")
        print(f"   Tamaño: ~{len(html_completo)/1024:.1f} KB")

        # Devolver datos para descarga
        return dict(
            content=html_completo,
            filename=nombre_archivo
        )

    except Exception as e:
        print(f"❌ Error al exportar HTML: {str(e)}")
        return None


# ============================================================
# EJECUTAR APLICACIÓN
# ============================================================

if __name__ == '__main__':
    print("="*80)
    print(" ✅ VERSIÓN 2.2 - EXPORTACIÓN HTML + N ALTERNATIVAS")
    print(" 📥 Exportar cada visualización como HTML independiente")
    print(" 🔢 Tabla dinámica con N alternativas (detección automática)")
    print(" 🌸 Diagrama de trébol (Sunburst)")
    print(" 📊 Tabla de evaluación de alternativas")
    print(" 📚 Guía de componentes LCC")
    print("="*80)
    print(f" {CONFIG['project']['name']}")
    print(f" {CONFIG['project']['client']} + {CONFIG['project']['collaboration']}")
    print(f" Strategic Sectoral Cooperation Project - {CONFIG['project']['year']}")
    print("="*80)
    print(f" 🌐 Servidor: 0.0.0.0:8080")
    print(f" 📊 TOPSIS MC: {CONFIG['topsis_monte_carlo']['n_simulations']} sims")
    print(f" 💰 LCC MC: {CONFIG['lcc_monte_carlo']['n_simulations']} sims")
    print("="*80)
    print(" 🔹 Abrir en navegador: http://localhost:8080")
    print("="*80)

    app.run(debug=True, host="0.0.0.0", port=8080)




    #app.run_server(debug=True, host="0.0.0.0", port=8080)

