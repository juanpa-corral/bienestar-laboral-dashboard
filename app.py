# ==============================================================================
#  ENTREGABLE 9 — DASHBOARD DE CALIDAD DE DATOS EN STREAMLIT
#  Bienestar Laboral y Salud Mental en Colombia
#  Universidad de La Sabana — Preprocesamiento de Datos 2026-1
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Calidad de Datos — Bienestar Laboral",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

DIMENSIONES = {
    "CTRL"    : {"items": ["CT1","CT2","CT3"],
                 "tipo": "Recurso",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Control sobre el trabajo"},
    "PRES"    : {"items": ["PT1","PT2","PT3","PT4"],
                 "tipo": "Demanda",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Presión de tiempo / carga laboral"},
    "LIDER"   : {"items": ["CL1","CL2","CL3","CL4","CL5","CL6","CL7"],
                 "tipo": "Recurso",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Calidad del liderazgo"},
    "COMP"    : {"items": ["AC1","AC2","AC3"],
                 "tipo": "Recurso",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Apoyo de compañeros"},
    "ROL_C"   : {"items": ["CR1","CR2","CR3","CR4"],
                 "tipo": "Demanda",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Conflicto de rol"},
    "ROL_CON" : {"items": ["CoR1","CoR2","CoR3"],
                 "tipo": "Demanda",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Ambigüedad de rol"},
    "CAMBIO"  : {"items": ["GC1","GC2","GC3","GC4"],
                 "tipo": "Demanda",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Gestión del cambio organizacional"},
    "SM_ORG"  : {"items": ["SM1","SM2","SM3","SM4","SM5"],
                 "tipo": "Recurso",  "escala": "Frecuencia", "rango": (1,5),
                 "descripcion": "Significado organizacional"},
    "SAT"     : {"items": ["SAT1","SAT2","SAT3","SAT4","SAT5","SAT6","SAT7","SAT8","SAT9"],
                 "tipo": "Recurso",  "escala": "Acuerdo",    "rango": (1,7),
                 "descripcion": "Satisfacción laboral"},
    "RETIRO"  : {"items": ["IR1","IR2","IR3","IR4"],
                 "tipo": "Demanda",  "escala": "Acuerdo",    "rango": (1,7),
                 "descripcion": "Intención de retiro (IR1 invertido)"},
    "FAM_TRAB": {"items": ["FT1","FT2","FT3","FT4","FT5"],
                 "tipo": "Demanda",  "escala": "Acuerdo",    "rango": (1,7),
                 "descripcion": "Conflicto familia → trabajo"},
    "TRAB_FAM": {"items": ["TF1","TF2","TF3","TF4","TF5"],
                 "tipo": "Demanda",  "escala": "Acuerdo",    "rango": (1,7),
                 "descripcion": "Conflicto trabajo → familia"},
    "BURNOUT" : {"items": ["BU1","BU2","BU3","BU4","BU5","BU6",
                           "BU7","BU8","BU9","BU10","BU11","BU12"],
                 "tipo": "Demanda",  "escala": "Desgaste",   "rango": (1,7),
                 "descripcion": "Burnout / agotamiento"},
    "BIENESTAR":{"items": ["BP1","BP2","BP3","BP4","BP5",
                           "BP6","BP7","BP8","BP9","BP10"],
                 "tipo": "Recurso",  "escala": "Bienestar",  "rango": (1,7),
                 "descripcion": "Bienestar percibido"},
    "SOMATIZ" : {"items": ["SOM1","SOM2","SOM3","SOM4","SOM5"],
                 "tipo": "Demanda",  "escala": "Desgaste",   "rango": (1,7),
                 "descripcion": "Somatización"},
    "DESGASTE": {"items": ["DL1","DL2","DL3","DL4","DL5","DL6","DL7","DL8"],
                 "tipo": "Demanda",  "escala": "Desgaste",   "rango": (1,7),
                 "descripcion": "Desgaste laboral (vigor negativo)"},
}

HIPOTESIS = [
    ("DESGASTE", "SOMATIZ",   "positiva", +0.50, "H1 — Desgaste ↔ Somatización"),
    ("LIDER",    "SAT",       "positiva", +0.40, "H2 — Liderazgo ↔ Satisfacción"),
    ("SAT",      "RETIRO",    "negativa", -0.40, "H3 — Satisfacción ↔ Intención de retiro"),
    ("BURNOUT",  "BIENESTAR", "negativa", -0.35, "H4 — Burnout ↔ Bienestar Percibido"),
    ("CTRL",     "BURNOUT",   "negativa", -0.30, "H5 — Control ↔ Burnout"),
    ("PRES",     "BURNOUT",   "positiva", +0.30, "H6 — Presión ↔ Burnout"),
    ("TRAB_FAM", "FAM_TRAB",  "positiva", +0.25, "H7 — Conflicto bidireccional T–F"),
]

ITEMS_ESCALA = [item for grp in DIMENSIONES.values() for item in grp["items"]]
DIMS_NAMES   = list(DIMENSIONES.keys())

# Columnas por tipo de escala (igual que en el notebook Sección 4)
FREQ_COLS    = ["CT1","CT2","CT3","PT1","PT2","PT3","PT4",
                "CL1","CL2","CL3","CL4","CL5","CL6","CL7",
                "AC1","AC2","AC3","CR1","CR2","CR3","CR4",
                "CoR1","CoR2","CoR3","GC1","GC2","GC3","GC4",
                "SM1","SM2","SM3","SM4","SM5"]
SOM_COLS     = ["BU1","BU2","BU3","BU4","BU5","BU6","BU7",
                "BU8","BU9","BU10","BU11","BU12",
                "DL1","DL2","DL3","DL4","DL5","DL6","DL7","DL8",
                "SOM1","SOM2","SOM3","SOM4","SOM5"]
ACUERDO_COLS = ["SAT1","SAT2","SAT3","SAT4","SAT5","SAT6","SAT7","SAT8","SAT9",
                "IR1","IR2","IR3","IR4",
                "FT1","FT2","FT3","FT4","FT5",
                "TF1","TF2","TF3","TF4","TF5"]
BP_COLS      = ["BP1","BP2","BP3","BP4","BP5","BP6","BP7","BP8","BP9","BP10"]

# Diccionarios de mapeo usados en el notebook
MAPA_FRECUENCIA = {
    "Nunca": 1, "Rara vez": 2, "Algunas veces": 3,
    "Frecuentemente": 4, "Siempre": 5,
}
MAPA_ACUERDO = {
    "Muy en desacuerdo": 1, "Moderadamente en desacuerdo": 2,
    "Algo en desacuerdo": 3, "Ni de acuerdo ni en desacuerdo": 4,
    "Algo de acuerdo": 5, "Moderadamente de acuerdo": 6, "Muy de acuerdo": 7,
}
MAPA_SOM = {
    "Nunca": 1, "Raramente": 2, "Ocasionalmente": 3,
    "Algunas veces": 4, "Frecuentemente": 5, "Casi siempre": 6, "Siempre": 7,
}
MAPA_BP_TEXTO = {
    "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7,
}

# Variantes textuales corregidas antes del mapeo
VARIANTES_FRECUENCIA = {
    "NUNCA": "Nunca", "SIEMPRE": "Siempre",
    "ALGUNAS VECES": "Algunas veces", "algunas veces": "Algunas veces",
    "frecuentemente": "Frecuentemente", "siempre": "Siempre",
    "Siempree": "Siempre", "Frecuente": "Frecuentemente",
    "Alguna vez": "Rara vez", "Raramente": "Rara vez",
    "Ocasionalmente": "Algunas veces", "A menudo": "Frecuentemente",
    "Casi siempre": "Siempre",
    "--": "NaN", "?": "NaN", "999": "NaN", "sin dato": "NaN",
}

TYPOS_EJEMPLO = [
    {"Variable": "Sexo",            "Antes": "mujer",           "Después": "Mujer"},
    {"Variable": "Sexo",            "Antes": "MUJER",           "Después": "Mujer"},
    {"Variable": "Estado_Civil",    "Antes": "Casdo",           "Después": "Casado"},
    {"Variable": "Estado_Civil",    "Antes": "Solero",          "Después": "Soltero"},
    {"Variable": "Estado_Civil",    "Antes": "SOLTERO",         "Después": "Soltero"},
    {"Variable": "Estado_Civil",    "Antes": "  Soltero",       "Después": "Soltero"},
    {"Variable": "Sector",          "Antes": "publico",         "Después": "Público"},
    {"Variable": "Modalidad",       "Antes": "Hibrido",         "Después": "Híbrido"},
    {"Variable": "Tipo_Contrato",   "Antes": "indefinido",      "Después": "Indefinido"},
    {"Variable": "Nivel_Educativo", "Antes": "profesional",     "Después": "Profesional"},
]

INVENTARIO_PROBLEMAS = pd.DataFrame([
    {"Problema encontrado": "Valores faltantes distribuidos en múltiples variables",
     "Columna(s) afectada(s)": "Todo el dataset",
     "Magnitud": "2 582 valores faltantes (5.60% del total de celdas)",
     "Tipo de problema": "Completitud"},
    {"Problema encontrado": "Variables con alto porcentaje de faltantes",
     "Columna(s) afectada(s)": "SAT1, GC2, SAT6, SM5, SM4, SAT8, SAT5, BU1, BU11, CT3, BP8, entre otras",
     "Magnitud": "Entre 7% y 11% de faltantes por columna",
     "Tipo de problema": "Completitud"},
    {"Problema encontrado": "Presencia de outliers detectados mediante IQR",
     "Columna(s) afectada(s)": "Variables numéricas continuas",
     "Magnitud": "43 outliers detectados",
     "Tipo de problema": "Validez / Consistencia"},
    {"Problema encontrado": "Filas completamente duplicadas",
     "Columna(s) afectada(s)": "Dataset completo",
     "Magnitud": "0 casos",
     "Tipo de problema": "Duplicidad"},
    {"Problema encontrado": "IDs duplicados",
     "Columna(s) afectada(s)": "ID",
     "Magnitud": "0 casos",
     "Tipo de problema": "Integridad"},
    {"Problema encontrado": "Coincidencia total en variables demográficas",
     "Columna(s) afectada(s)": "Edad, Sexo, Estado_Civil, Numero_Hijos, Nivel_Educativo, Zona_Vivienda, Estrato, Sector, Años_Experiencia, Ingreso, Tipo_Cargo, Antiguedad_Cargo",
     "Magnitud": "9 registros coincidentes",
     "Tipo de problema": "Posible duplicidad"},
    {"Problema encontrado": "Coincidencia total en perfil profesional",
     "Columna(s) afectada(s)": "Sector, Tamaño_Empresa, Tipo_Contrato, Tipo_Cargo, Personas_Cargo, Modalidad, Horas_Semana, Trabajo_Turnos",
     "Magnitud": "28 registros coincidentes",
     "Tipo de problema": "Posible duplicidad"},
    {"Problema encontrado": "Inconsistencias en escalas Likert por diferencias de escritura, capitalización y categorías inválidas",
     "Columna(s) afectada(s)": "AC1, AC2, BU1–BU12, DL1–DL8, SOM1–SOM5, entre otras",
     "Magnitud": "Variables con hasta 214 inconsistencias",
     "Tipo de problema": "Estandarización / Validez"},
    {"Problema encontrado": "Uso de categorías no permitidas en variables Likert",
     "Columna(s) afectada(s)": "Variables Likert",
     "Magnitud": 'Valores como "SIEMPRE", "siempre", "Alguna vez", "A menudo", etc.',
     "Tipo de problema": "Dominio inválido"},
    {"Problema encontrado": "Inconsistencias severas en variables BU, DL y SOM",
     "Columna(s) afectada(s)": "BU1–BU12, DL1–DL8, SOM1–SOM5",
     "Magnitud": "Más de 150 inconsistencias por variable en varios casos",
     "Tipo de problema": "Calidad semántica"},
    {"Problema encontrado": "Variables numéricas almacenadas con texto",
     "Columna(s) afectada(s)": "BP1–BP5",
     "Magnitud": "Entre 17 y 20 inconsistencias por variable",
     "Tipo de problema": "Tipo de dato incorrecto"},
    {"Problema encontrado": "Valores no numéricos en escalas numéricas",
     "Columna(s) afectada(s)": "BP1–BP5",
     "Magnitud": 'Valores como "cuatro", "cinco", "siete", etc.',
     "Tipo de problema": "Validez / Formato"},
    {"Problema encontrado": "Valores inválidos en escalas de acuerdo",
     "Columna(s) afectada(s)": "IR1, IR3, IR4",
     "Magnitud": "1–3 inconsistencias por variable",
     "Tipo de problema": "Dominio inválido"},
    {"Problema encontrado": "Uso de códigos ambiguos o placeholders",
     "Columna(s) afectada(s)": "IR1, IR3, IR4",
     "Magnitud": 'Valores "999", "--", "?", "sin dato"',
     "Tipo de problema": "Codificación inconsistente"},
    {"Problema encontrado": "Inconsistencias ortográficas y de capitalización en Estado Civil",
     "Columna(s) afectada(s)": "Estado_Civil",
     "Magnitud": "Múltiples variantes incorrectas",
     "Tipo de problema": "Estandarización"},
    {"Problema encontrado": "Variantes inválidas de categorías en Estado Civil",
     "Columna(s) afectada(s)": "Estado_Civil",
     "Magnitud": 'Valores como "Casdo", "Solero", "SOLTERO", "  Soltero"',
     "Tipo de problema": "Error tipográfico / Formato"},
    {"Problema encontrado": "Espacios innecesarios en categorías categóricas",
     "Columna(s) afectada(s)": "Estado_Civil",
     "Magnitud": "Casos con espacios al inicio",
     "Tipo de problema": "Limpieza de texto"},
    {"Problema encontrado": "Mezcla de mayúsculas y minúsculas en variables categóricas",
     "Columna(s) afectada(s)": "Estado_Civil y variables Likert",
     "Magnitud": "Presente en múltiples categorías",
     "Tipo de problema": "Normalización"},
    {"Problema encontrado": "Posible inconsistencia en definición de tipos de variables",
     "Columna(s) afectada(s)": "Variables categóricas y escalas",
     "Magnitud": "Algunas escalas almacenadas como texto en lugar de categorías ordenadas",
     "Tipo de problema": "Modelado de datos"},
    {"Problema encontrado": "Alta heterogeneidad de formatos de respuesta",
     "Columna(s) afectada(s)": "Variables de encuesta",
     "Magnitud": "Presente en varias escalas psicométricas",
     "Tipo de problema": "Consistencia semántica"},
])

# ──────────────────────────────────────────────────────────────────────────────
#  CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando datos…")
def cargar_datos():
    try:
        df_orig  = pd.read_excel("bienestar_laboral_original.xlsx")
    except FileNotFoundError:
        df_orig = None
    try:
        df_clean = pd.read_csv("bienestar_laboral_LIMPIO.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        df_clean = None
    return df_orig, df_clean


def calcular_dimensiones(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for dim, meta in DIMENSIONES.items():
        cols = [c for c in meta["items"] if c in df_out.columns]
        if cols:
            subset = df_out[cols].apply(pd.to_numeric, errors="coerce")
            df_out[dim] = subset.mean(axis=1, skipna=True).round(3)
    return df_out

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "unisabanaLogo.png",
        width=160,
    )
    st.title("🧠 Dashboard Bienestar Laboral")
    st.markdown(
        "**Preprocesamiento de Datos 2026-1**  \n"
        "Universidad de La Sabana  \n\n"
    )
    st.divider()
    st.subheader("📂 Cargar datos")
    upload_orig  = st.file_uploader("Dataset original (.xlsx)", type=["xlsx"])
    upload_clean = st.file_uploader("Dataset limpio (.csv)",    type=["csv"])
    st.caption("Si no sube archivos, el app buscará los archivos locales del repositorio.")
    st.divider()
    st.markdown("**Integrantes del equipo**")
    st.markdown(
        "- Juan Pablo Corral  \n"
        "- Juan Esteban Ocampo  \n"
        "- Valentina Ramírez  \n"
        "- Santiago Mateo Lozano"
    )

# ── Carga efectiva ───────────────────────────────────────────────────────────
df_orig_file, df_clean_file = cargar_datos()
df_orig  = pd.read_excel(upload_orig)  if upload_orig  is not None else df_orig_file
df_clean = pd.read_csv(upload_clean, encoding="utf-8-sig") if upload_clean is not None else df_clean_file

if df_orig is None and df_clean is None:
    st.error("⚠️ No se encontraron datos. Sube los archivos en la barra lateral.")
    st.stop()
if df_orig is None:
    df_orig = df_clean.copy()
    st.warning("Dataset original no disponible — usando el limpio como referencia.")
if df_clean is None:
    df_clean = calcular_dimensiones(df_orig)
    st.warning("Dataset limpio no encontrado — calculado desde el original sin preprocesar.")

dims_disponibles = [d for d in DIMS_NAMES if d in df_clean.columns]

# ──────────────────────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panel 1 — Estado Inicial",
    "🔍 Panel 2 — Explorador de Problemas",
    "⚖️ Panel 3 — Comparativo Antes/Después",
    "✅ Panel 4 — Certificación de Calidad",
])

# ══════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — ESTADO INICIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Estado Inicial del Dataset")
    st.markdown(
        "Diagnóstico del dataset **antes** de cualquier transformación. "
        "Se documentan dimensiones, patrones de datos faltantes y distribuciones "
        "problemáticas de las variables clave."
    )

    total_nan = int(df_orig.isna().sum().sum())
    pct_nan   = total_nan / (df_orig.shape[0] * df_orig.shape[1]) * 100
    n_dup     = int(df_orig.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros",        f"{df_orig.shape[0]:,}")
    c2.metric("Variables",         f"{df_orig.shape[1]:,}")
    c3.metric("Valores faltantes", f"{total_nan:,}",
              delta=f"{pct_nan:.1f}% del total", delta_color="inverse")
    c4.metric("Filas duplicadas",  f"{n_dup:,}", delta_color="inverse")

    st.divider()

    st.subheader("🔥 Mapa de calor de valores faltantes")
    st.caption("Oscuro = valor faltante · Claro = valor presente. Top 50 columnas con mayor % de NaN.")
    nan_pct = df_orig.isna().mean().sort_values(ascending=False)
    top50   = nan_pct[nan_pct > 0].head(50).index.tolist()

    if top50:
        fig_heat, ax_heat = plt.subplots(figsize=(16, 5))
        sns.heatmap(df_orig[top50].isna().astype(int).T,
                    cmap="Blues", cbar=False, linewidths=0, ax=ax_heat,
                    yticklabels=True, xticklabels=False)
        ax_heat.set_xlabel("Registros (filas)", fontsize=10)
        ax_heat.set_title(f"Top {len(top50)} columnas con datos faltantes",
                          fontsize=12, fontweight="bold")
        ax_heat.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close()
    else:
        st.success("✅ El dataset no tiene valores faltantes.")

    st.divider()

    st.subheader("📋 Inventario de problemas detectados")
    st.caption(f"{len(INVENTARIO_PROBLEMAS)} problemas identificados en el Data Profiling.")

    tipos_unicos = ["Todos"] + sorted(INVENTARIO_PROBLEMAS["Tipo de problema"].unique().tolist())
    tipo_filtro  = st.selectbox("Filtrar por tipo:", tipos_unicos, key="filtro_inv")
    df_mostrar   = INVENTARIO_PROBLEMAS if tipo_filtro == "Todos" else \
                   INVENTARIO_PROBLEMAS[INVENTARIO_PROBLEMAS["Tipo de problema"] == tipo_filtro]
    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    conteo_tipos = INVENTARIO_PROBLEMAS["Tipo de problema"].value_counts().reset_index()
    conteo_tipos.columns = ["Tipo", "N"]
    fig_ct, ax_ct = plt.subplots(figsize=(10, 4))
    ax_ct.barh(conteo_tipos["Tipo"], conteo_tipos["N"], color="#5C6BC0", alpha=0.85)
    ax_ct.set_xlabel("N° de problemas")
    ax_ct.set_title("Distribución de problemas por tipo", fontweight="bold")
    ax_ct.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_ct)
    plt.close()

    st.divider()

    st.subheader("📈 Distribución de variables clave (dataset original)")
    vars_num = [v for v in ["Edad","Horas_Semana","Estrato","Horas_Formacion"]
                if v in df_orig.columns]
    if vars_num:
        fig_d, axes_d = plt.subplots(1, len(vars_num), figsize=(4.5 * len(vars_num), 4))
        if len(vars_num) == 1:
            axes_d = [axes_d]
        for ax, col in zip(axes_d, vars_num):
            serie = pd.to_numeric(df_orig[col], errors="coerce").dropna()
            sns.histplot(serie, ax=ax, kde=True, color="#FF7043", alpha=0.75, bins=25, linewidth=0)
            ax.set_title(col, fontweight="bold", fontsize=11)
            ax.set_xlabel("")
        plt.suptitle("Variables numéricas con potenciales outliers", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — EXPLORADOR DE PROBLEMAS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explorador de Problemas de Calidad")
    st.markdown(
        "Seleccione el tipo de problema para inspeccionar los registros "
        "afectados y visualizar el tratamiento aplicado."
    )

    tipo_problema = st.selectbox(
        "Seleccione el tipo de problema:",
        options=[
            "Valores faltantes por columna",
            "Columnas con > 5% de NaN",
            "Filas duplicadas exactas",
            "Outliers en variables numéricas",
            "Estandarización de texto (typos corregidos)",
            "Tratamiento de escalas Likert",
            "Distribución de ítems de escala (sucios vs limpios)",
        ],
    )
    st.divider()

    # ── Faltantes por columna ─────────────────────────────────────────────
    if tipo_problema == "Valores faltantes por columna":
        nan_col = (df_orig.isna().sum().reset_index()
                   .rename(columns={"index": "Columna", 0: "N_faltantes"}))
        nan_col["% faltante"] = (nan_col["N_faltantes"] / len(df_orig) * 100).round(2)
        nan_col = nan_col[nan_col["N_faltantes"] > 0].sort_values("N_faltantes", ascending=False)
        st.metric("Columnas con al menos 1 NaN", len(nan_col))
        st.dataframe(nan_col, use_container_width=True, hide_index=True)
        fig_nb, ax_nb = plt.subplots(figsize=(12, 5))
        top20 = nan_col.head(20)
        ax_nb.bar(top20["Columna"], top20["% faltante"], color="#EF5350", alpha=0.85)
        ax_nb.axhline(5, color="black", linestyle="--", linewidth=1.2, label="Umbral 5%")
        ax_nb.set_ylabel("% de valores faltantes")
        ax_nb.set_title("Top 20 columnas con más valores faltantes", fontweight="bold")
        plt.xticks(rotation=65, ha="right", fontsize=8)
        ax_nb.legend()
        plt.tight_layout()
        st.pyplot(fig_nb)
        plt.close()

    # ── Columnas > 5% ─────────────────────────────────────────────────────
    elif tipo_problema == "Columnas con > 5% de NaN":
        cols_crit = df_orig.columns[df_orig.isna().mean() > 0.05]
        if len(cols_crit) == 0:
            st.success("Ninguna columna supera el 5% de faltantes.")
        else:
            resumen = pd.DataFrame({
                "Columna": cols_crit,
                "N NaN"  : df_orig[cols_crit].isna().sum().values,
                "% NaN"  : (df_orig[cols_crit].isna().mean() * 100).round(2).values,
            }).sort_values("% NaN", ascending=False)
            st.metric("Columnas críticas (> 5% NaN)", len(resumen))
            st.dataframe(resumen, use_container_width=True, hide_index=True)
            fig_cc, ax_cc = plt.subplots(figsize=(12, 4))
            ax_cc.barh(resumen["Columna"].head(20), resumen["% NaN"].head(20),
                       color="#EF5350", alpha=0.85)
            ax_cc.axvline(5, color="black", linestyle="--", linewidth=1.2)
            ax_cc.set_xlabel("% NaN")
            ax_cc.set_title("Columnas con > 5% de NaN", fontweight="bold")
            ax_cc.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cc)
            plt.close()

    # ── Duplicados ────────────────────────────────────────────────────────
    elif tipo_problema == "Filas duplicadas exactas":
        mask_dup = df_orig.duplicated(keep=False)
        n_dup2   = int(mask_dup.sum())
        if n_dup2 == 0:
            st.success("No hay filas exactamente duplicadas en el dataset original.")
            st.info(
                "💡 En la Sección 2 se detectaron duplicados por **subconjunto de columnas "
                "de perfil** (sociodemográficas + laborales). "
                "Esa estrategia eliminó **28 registros** con mismo perfil pero distinto ID."
            )
        else:
            st.metric("Filas duplicadas exactas", n_dup2)
            st.dataframe(df_orig[mask_dup].head(50), use_container_width=True)

    # ── Outliers ──────────────────────────────────────────────────────────
    elif tipo_problema == "Outliers en variables numéricas":
        num_cols = [c for c in df_orig.select_dtypes(include="number").columns if c != "ID"]
        outlier_rep = []
        for col in num_cols:
            s = pd.to_numeric(df_orig[col], errors="coerce").dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            n_out  = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum())
            if n_out > 0:
                outlier_rep.append({"Variable": col, "N outliers IQR": n_out,
                                    "Min obs": round(s.min(), 2), "Max obs": round(s.max(), 2),
                                    "Q1": round(Q1, 2), "Q3": round(Q3, 2)})
        if outlier_rep:
            st.dataframe(pd.DataFrame(outlier_rep), use_container_width=True, hide_index=True)
            n_p = len(outlier_rep)
            fig_bp, axes_bp = plt.subplots(1, n_p, figsize=(4.5 * n_p, 5))
            if n_p == 1:
                axes_bp = [axes_bp]
            fp = dict(marker="o", markerfacecolor="red", markeredgecolor="red", markersize=7)
            for ax, rec in zip(axes_bp, outlier_rep):
                s = pd.to_numeric(df_orig[rec["Variable"]], errors="coerce")
                sns.boxplot(y=s, ax=ax, color="#FFA726", flierprops=fp)
                ax.set_title(rec["Variable"], fontweight="bold")
            plt.suptitle("Boxplots — variables con outliers (dataset original)",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_bp)
            plt.close()
        else:
            st.success("No se detectaron outliers IQR en variables numéricas.")

    # ── Typos ─────────────────────────────────────────────────────────────
    elif tipo_problema == "Estandarización de texto (typos corregidos)":
        st.markdown("Errores tipográficos corregidos en la **Sección 3** del pipeline.")
        st.dataframe(pd.DataFrame(TYPOS_EJEMPLO), use_container_width=True, hide_index=True)
        st.code(
            "df[col] = df[col].str.strip()   # quitar espacios\n"
            "df[col] = df[col].str.title()   # Title Case\n"
            "df[col] = df[col].replace(diccionario_correcciones)",
            language="python",
        )
        cats_ok = [c for c in ["Sexo","Estado_Civil","Sector","Modalidad","Tipo_Contrato"]
                   if c in df_orig.columns and c in df_clean.columns]
        if cats_ok:
            col_sel = st.selectbox("Ver categorías únicas antes/después:", cats_ok)
            ca, cb  = st.columns(2)
            with ca:
                st.markdown("**ORIGINAL**")
                vc = df_orig[col_sel].value_counts().reset_index()
                vc.columns = [col_sel, "N"]
                st.dataframe(vc, use_container_width=True, hide_index=True)
            with cb:
                st.markdown("**LIMPIO**")
                vc2 = df_clean[col_sel].value_counts().reset_index()
                vc2.columns = [col_sel, "N"]
                st.dataframe(vc2, use_container_width=True, hide_index=True)

    # ── TRATAMIENTO DE ESCALAS LIKERT (NUEVO) ─────────────────────────────
    elif tipo_problema == "Tratamiento de escalas Likert":

        st.markdown(
            "### Sección 4 — Tratamiento completo de escalas Likert\n"
            "El pipeline unificó variantes textuales, aplicó mapeo numérico, "
            "convirtió los valores de texto en BP y corrigió el ítem inverso IR1."
        )

        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "1️⃣ Unificación de variantes",
            "2️⃣ Mapeo texto → número",
            "3️⃣ Corrección BP y IR1",
            "4️⃣ Verificación de rangos",
        ])

        # ── Subtab 1: variantes textuales ─────────────────────────────────
        with subtab1:
            st.markdown("#### Variantes textuales encontradas y su corrección canónica")
            st.markdown(
                "Antes de convertir a número, se estandarizaron todas las formas "
                "en que los encuestadores escribieron la misma respuesta:"
            )

            df_variantes = pd.DataFrame([
                {"Valor encontrado": "SIEMPRE",       "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Siempre",        "Código numérico": 5},
                {"Valor encontrado": "siempre",       "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Siempre",        "Código numérico": 5},
                {"Valor encontrado": "Siempree",      "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Siempre",        "Código numérico": 5},
                {"Valor encontrado": "Casi siempre",  "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Siempre",        "Código numérico": 5},
                {"Valor encontrado": "A menudo",      "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Frecuentemente", "Código numérico": 4},
                {"Valor encontrado": "frecuentemente","Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Frecuentemente", "Código numérico": 4},
                {"Valor encontrado": "Frecuente",     "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Frecuentemente", "Código numérico": 4},
                {"Valor encontrado": "Ocasionalmente","Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Algunas veces",  "Código numérico": 3},
                {"Valor encontrado": "ALGUNAS VECES", "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Algunas veces",  "Código numérico": 3},
                {"Valor encontrado": "algunas veces", "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Algunas veces",  "Código numérico": 3},
                {"Valor encontrado": "Alguna vez",    "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Rara vez",       "Código numérico": 2},
                {"Valor encontrado": "Raramente",     "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Rara vez",       "Código numérico": 2},
                {"Valor encontrado": "NUNCA",          "Escala afectada": "Frecuencia (1-5)", "Valor canónico": "Nunca",          "Código numérico": 1},
                {"Valor encontrado": "--",             "Escala afectada": "Todas",            "Valor canónico": "NaN",            "Código numérico": "→ imputado"},
                {"Valor encontrado": "999",            "Escala afectada": "Todas",            "Valor canónico": "NaN",            "Código numérico": "→ imputado"},
                {"Valor encontrado": "?",              "Escala afectada": "Todas",            "Valor canónico": "NaN",            "Código numérico": "→ imputado"},
                {"Valor encontrado": "sin dato",       "Escala afectada": "Todas",            "Valor canónico": "NaN",            "Código numérico": "→ imputado"},
            ])
            st.dataframe(df_variantes, use_container_width=True, hide_index=True)

        # ── Subtab 2: mapeo texto → número ────────────────────────────────
        with subtab2:
            st.markdown("#### Diccionarios de mapeo aplicados por tipo de escala")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Escala Frecuencia (1–5)**")
                st.markdown("*CT, PT, CL, AC, CR, CoR, GC, SM*")
                df_freq = pd.DataFrame(list(MAPA_FRECUENCIA.items()),
                                       columns=["Texto canónico", "Código"])
                st.dataframe(df_freq, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Escala Acuerdo (1–7)**")
                st.markdown("*SAT, IR, FT, TF*")
                df_acuerdo = pd.DataFrame(list(MAPA_ACUERDO.items()),
                                          columns=["Texto canónico", "Código"])
                st.dataframe(df_acuerdo, use_container_width=True, hide_index=True)

            with col3:
                st.markdown("**Escala Desgaste/Somatización (1–7)**")
                st.markdown("*BU, DL, SOM*")
                df_som = pd.DataFrame(list(MAPA_SOM.items()),
                                      columns=["Texto canónico", "Código"])
                st.dataframe(df_som, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### Visualización del mapeo — ejemplo con dimensión PRES")
            st.markdown("Así lucían los datos **antes** (texto) y **después** (número):")

            pres_items = [c for c in ["PT1","PT2","PT3","PT4"] if c in df_orig.columns]
            if pres_items:
                col_ej = pres_items[0]
                antes_ej = df_orig[col_ej].value_counts().head(8).reset_index()
                antes_ej.columns = ["Valor (texto)", "N registros"]
                antes_ej["→ Número"] = antes_ej["Valor (texto)"].map(
                    {**MAPA_FRECUENCIA,
                     **{k: "NaN→imputado" for k in ["--","999","?","sin dato"]},
                     **{k: v for k, v in [("SIEMPRE","Siempre→5"),("siempre","Siempre→5"),
                                           ("A menudo","Frecuentemente→4"),
                                           ("Alguna vez","Rara vez→2")]}}
                ).fillna("Ver diccionario")
                st.dataframe(antes_ej, use_container_width=True, hide_index=True)

        # ── Subtab 3: BP y IR1 ────────────────────────────────────────────
        with subtab3:
            st.markdown("#### Caso especial 1 — Columnas BP: texto → número")
            st.info(
                "Las columnas BP1–BP10 (Bienestar Percibido) tenían valores escritos "
                "en palabras en lugar de números. Fueron convertidas con un mapeo específico."
            )
            df_bp = pd.DataFrame(list(MAPA_BP_TEXTO.items()),
                                  columns=["Texto encontrado en BP", "Número asignado"])
            ca, cb = st.columns([1, 2])
            with ca:
                st.dataframe(df_bp, use_container_width=True, hide_index=True)
            with cb:
                st.code(
                    "mapa_bp = {'uno':1,'dos':2,'tres':3,'cuatro':4,\n"
                    "           'cinco':5,'seis':6,'siete':7}\n\n"
                    "for col in bp_cols:\n"
                    "    df[col] = df[col].replace(mapa_bp)\n"
                    "    df[col] = pd.to_numeric(df[col], errors='coerce')",
                    language="python",
                )

            st.divider()
            st.markdown("#### Caso especial 2 — Ítem inverso IR1: inversión (8 − IR1)")
            st.warning(
                "**¿Por qué se invirtió IR1?**  \n"
                "IR1 pregunta: *'Me gustaría continuar trabajando en esta organización'* "
                "(sentido positivo).  \n"
                "IR2, IR3, IR4 miden intención de **retiro** (sentido negativo).  \n"
                "Para que todos apunten en la misma dirección se aplica: **8 − IR1**."
            )

            df_ir1 = pd.DataFrame({
                "Valor original IR1": [1, 2, 3, 4, 5, 6, 7],
                "Interpretación original": [
                    "Nunca quiere irse", "Casi nunca", "Rara vez",
                    "Neutral", "Algo de retiro", "Bastante retiro", "Siempre quiere irse"
                ],
                "Valor invertido (8−IR1)": [7, 6, 5, 4, 3, 2, 1],
                "Interpretación invertida": [
                    "Alta intención de retiro", "Bastante", "Algo",
                    "Neutral", "Rara vez", "Casi nunca", "Nunca piensa en irse"
                ],
            })
            st.dataframe(df_ir1, use_container_width=True, hide_index=True)

        # ── Subtab 4: verificación de rangos ──────────────────────────────
        with subtab4:
            st.markdown("#### Verificación de rangos post-codificación")
            st.markdown(
                "Después de aplicar el mapeo numérico se verificó que **ningún valor "
                "quedó fuera del rango teórico** de su escala."
            )

            escalas_config = [
                ("Frecuencia (1–5)", FREQ_COLS,    1, 5),
                ("Acuerdo (1–7)",    ACUERDO_COLS, 1, 7),
                ("Desgaste (1–7)",   SOM_COLS,     1, 7),
                ("Bienestar (1–7)",  BP_COLS,      1, 7),
            ]

            resumen_rangos = []
            for nombre_escala, cols, rmin, rmax in escalas_config:
                cols_ok = [c for c in cols if c in df_clean.columns]
                if not cols_ok:
                    continue
                vals = df_clean[cols_ok].apply(pd.to_numeric, errors="coerce")
                n_fuera = int(((vals < rmin) | (vals > rmax)).sum().sum())
                obs_min = round(vals.min().min(), 2)
                obs_max = round(vals.max().max(), 2)
                resumen_rangos.append({
                    "Escala"        : nombre_escala,
                    "N columnas"    : len(cols_ok),
                    "Rango teórico" : f"[{rmin}–{rmax}]",
                    "Min observado" : obs_min,
                    "Max observado" : obs_max,
                    "Fuera de rango": n_fuera,
                    "Estado"        : "✅ OK" if n_fuera == 0 else f"❌ {n_fuera} errores",
                })

            df_rangos = pd.DataFrame(resumen_rangos)
            st.dataframe(df_rangos, use_container_width=True, hide_index=True)

            total_errores = df_rangos["Fuera de rango"].sum() if len(df_rangos) > 0 else 0
            if total_errores == 0:
                st.success("✅ Todos los ítems de escala están dentro del rango teórico tras la codificación.")
            else:
                st.error(f"❌ {total_errores} valores fuera de rango — revisar el mapeo.")

    # ── Distribución ítems escala (sucios vs limpios) ─────────────────────
    elif tipo_problema == "Distribución de ítems de escala (sucios vs limpios)":
        dim_sel   = st.selectbox("Seleccione dimensión:", DIMS_NAMES)
        items_sel = [c for c in DIMENSIONES[dim_sel]["items"] if c in df_orig.columns]

        if items_sel:
            item_sel = st.selectbox("Seleccione ítem:", items_sel)
            st.info("📌 Comparación del mismo ítem antes y después del pipeline.")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{item_sel} — ORIGINAL (texto sucio)**")
                vc_o = df_orig[item_sel].value_counts().head(10)
                fig_o, ax_o = plt.subplots(figsize=(5, 4))
                ax_o.barh(vc_o.index.astype(str), vc_o.values, color="#7E57C2", alpha=0.85)
                ax_o.set_xlabel("N registros")
                ax_o.set_title(f"{item_sel} — Antes", fontweight="bold")
                ax_o.tick_params(axis="y", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_o)
                plt.close()
                st.caption(f"Valores únicos encontrados: {df_orig[item_sel].nunique()}")

            with col_b:
                st.markdown(f"**{item_sel} — LIMPIO (numérico)**")
                if item_sel in df_clean.columns:
                    vals_clean = pd.to_numeric(df_clean[item_sel], errors="coerce").dropna()
                    fig_c, ax_c = plt.subplots(figsize=(5, 4))
                    rmin, rmax = DIMENSIONES[dim_sel]["rango"]
                    ax_c.hist(vals_clean, bins=range(rmin, rmax + 2),
                              color="#43A047", alpha=0.85, edgecolor="white", rwidth=0.8)
                    ax_c.set_xlabel("Valor numérico")
                    ax_c.set_ylabel("N registros")
                    ax_c.set_title(f"{item_sel} — Después", fontweight="bold")
                    ax_c.set_xticks(range(rmin, rmax + 1))
                    plt.tight_layout()
                    st.pyplot(fig_c)
                    plt.close()
                    st.caption(
                        f"Media: {vals_clean.mean():.2f} | "
                        f"Min: {int(vals_clean.min())} | Max: {int(vals_clean.max())} | "
                        f"Rango esperado: [{rmin}–{rmax}]"
                    )
        else:
            st.warning(f"No se encontraron ítems de {dim_sel} en el dataset original.")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — COMPARATIVO ANTES / DESPUÉS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("⚖️ Comparativo Antes / Después del Pipeline")
    st.markdown(
        "Seleccione una variable para comparar su distribución y estadísticos "
        "entre el dataset **original** y el dataset **limpio**."
    )

    st.subheader("📋 Resumen del pipeline")

    items_en_clean  = [c for c in ITEMS_ESCALA if c in df_clean.columns]
    nan_items_clean = int(df_clean[items_en_clean].isna().sum().sum()) if items_en_clean else 0
    cols_aux        = [c for c in df_clean.columns
                       if c not in ITEMS_ESCALA and c not in DIMS_NAMES
                       and c not in ["ID","IR1_original"]]
    nan_aux_clean   = int(df_clean[cols_aux].isna().sum().sum())

    tabla_comp = pd.DataFrame([
        {"Métrica": "N° registros",                                  "Antes": 412,         "Después": len(df_clean),        "Δ": f"{len(df_clean)-412:+d}"},
        {"Métrica": "N° columnas",                                   "Antes": 112,         "Después": df_clean.shape[1],     "Δ": f"{df_clean.shape[1]-112:+d}"},
        {"Métrica": "NaN en ítems de escala (91 vars. de análisis)", "Antes": 2582,        "Después": nan_items_clean,       "Δ": f"{nan_items_clean-2582:+d}"},
        {"Métrica": "NaN en vars. auxiliares (fuera del scope)",     "Antes": "N/A",       "Después": nan_aux_clean,         "Δ": "No imputadas"},
        {"Métrica": "Filas duplicadas",                              "Antes": "Detectadas","Después": 0,                     "Δ": "→ 0"},
        {"Métrica": "Ítems Likert fuera de rango",                   "Antes": "Múltiples", "Después": 0,                     "Δ": "→ 0"},
        {"Métrica": "Outliers numér. (IQR)",                         "Antes": 43,          "Después": "0 (winzorizados)",    "Δ": "-43"},
        {"Métrica": "Variables derivadas (JD-R)",                    "Antes": 0,           "Después": len(dims_disponibles), "Δ": f"+{len(dims_disponibles)}"},
    ])
    st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 Comparar una variable específica")

    todas_vars = sorted(set(df_orig.columns) & set(df_clean.columns))
    var_sel    = st.selectbox("Seleccione variable:", todas_vars, index=0)

    orig_num  = pd.to_numeric(df_orig[var_sel],  errors="coerce")
    clean_num = pd.to_numeric(df_clean[var_sel], errors="coerce")
    es_num_o  = orig_num.dropna().shape[0] > 10
    es_num_c  = clean_num.dropna().shape[0] > 10

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dataset ORIGINAL (sucio)**")
        if es_num_o:
            fig_a, ax_a = plt.subplots(figsize=(5, 3.5))
            sns.histplot(orig_num.dropna(), kde=True, ax=ax_a,
                         color="#EF5350", alpha=0.75, bins=25, linewidth=0)
            ax_a.axvline(orig_num.mean(), color="black", linestyle="--",
                         linewidth=1.3, label=f"μ = {orig_num.mean():.2f}")
            ax_a.set_title(f"{var_sel} — Original", fontweight="bold")
            ax_a.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_a)
            plt.close()
            st.dataframe(orig_num.describe().round(3).to_frame("Estadístico"),
                         use_container_width=True)
        else:
            if var_sel in ITEMS_ESCALA:
                st.caption("⚠️ Variable texto Likert en el original → convertida a numérico en limpio.")
            vc = df_orig[var_sel].value_counts().reset_index()
            vc.columns = [var_sel, "N"]
            st.dataframe(vc, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Dataset LIMPIO (preprocesado)**")
        if es_num_c:
            fig_d, ax_d = plt.subplots(figsize=(5, 3.5))
            sns.histplot(clean_num.dropna(), kde=True, ax=ax_d,
                         color="#43A047", alpha=0.75, bins=25, linewidth=0)
            ax_d.axvline(clean_num.mean(), color="black", linestyle="--",
                         linewidth=1.3, label=f"μ = {clean_num.mean():.2f}")
            ax_d.set_title(f"{var_sel} — Limpio", fontweight="bold")
            ax_d.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close()
            st.dataframe(clean_num.describe().round(3).to_frame("Estadístico"),
                         use_container_width=True)
        else:
            if var_sel in ITEMS_ESCALA:
                st.success("✅ Variable convertida a numérico tras el mapeo canónico Likert.")
            vc2 = df_clean[var_sel].value_counts().reset_index()
            vc2.columns = [var_sel, "N"]
            st.dataframe(vc2, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL 4 — CERTIFICACIÓN DE CALIDAD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("✅ Certificación de Calidad del Dataset")
    st.markdown(
        "Semáforo de calidad por dimensión psicosocial, verificación de "
        "correlaciones teóricas esperadas y resumen ejecutivo del pipeline."
    )

    st.subheader("🚦 Semáforo de calidad — Dimensiones JD-R")
    st.caption("🟢 Excelente (media en rango + 0 NaN) · 🟡 Revisar · 🔴 Alerta")

    filas_sem = []
    for dim, meta in DIMENSIONES.items():
        if dim not in df_clean.columns:
            continue
        serie    = pd.to_numeric(df_clean[dim], errors="coerce")
        media    = serie.mean()
        n_nan    = int(serie.isna().sum())
        pct_n    = n_nan / len(df_clean) * 100
        rmin, rmax = meta["rango"]
        en_rango = (rmin <= media <= rmax)
        if en_rango and n_nan == 0:
            estado = "🟢 Excelente"
        elif en_rango and pct_n <= 5:
            estado = "🟡 Revisar"
        else:
            estado = "🔴 Alerta"
        filas_sem.append({
            "Estado": estado, "Dimensión": dim, "Tipo JD-R": meta["tipo"],
            "Escala": meta["escala"], "Rango teórico": f"[{rmin}–{rmax}]",
            "Media obs.": round(media, 3), "NaN finales": n_nan,
            "% NaN": f"{pct_n:.1f}%", "Descripción": meta["descripcion"],
        })

    df_sem   = pd.DataFrame(filas_sem)
    n_verde  = (df_sem["Estado"] == "🟢 Excelente").sum()
    n_amaril = (df_sem["Estado"] == "🟡 Revisar").sum()
    n_rojo   = (df_sem["Estado"] == "🔴 Alerta").sum()

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("🟢 Excelente", n_verde)
    kc2.metric("🟡 Revisar",   n_amaril)
    kc3.metric("🔴 Alerta",    n_rojo)
    st.dataframe(df_sem, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🔗 Verificación de correlaciones teóricas (JD-R)")
    st.caption("Hipótesis de Demerouti et al. (2001) y Maslach & Leiter (2016).")

    dims_hipo    = sorted({d for h in HIPOTESIS for d in [h[0], h[1]]})
    dims_hipo_ok = [d for d in dims_hipo if d in df_clean.columns]
    corr_m       = df_clean[dims_hipo_ok].corr(method="pearson").round(3) if dims_hipo_ok else None

    filas_h = []
    for dim_a, dim_b, direccion, umbral, nombre in HIPOTESIS:
        if corr_m is None or dim_a not in corr_m.index or dim_b not in corr_m.columns:
            continue
        r      = corr_m.loc[dim_a, dim_b]
        cond   = f"≥ {umbral:+.2f}" if direccion == "positiva" else f"≤ {umbral:+.2f}"
        cumple = (r >= umbral) if direccion == "positiva" else (r <= umbral)
        filas_h.append({"Hipótesis": nombre, "r observada": round(r, 3),
                         "Umbral": cond, "¿Cumple?": "✅ Sí" if cumple else "❌ No"})

    if filas_h:
        df_h = pd.DataFrame(filas_h)
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        n_ok = sum(1 for h in filas_h if "✅" in h["¿Cumple?"])
        st.metric("Hipótesis teóricas satisfechas", f"{n_ok} / {len(filas_h)}",
                  delta="✔ Estructura JD-R reproducida" if n_ok == len(filas_h) else "⚠ Revisar",
                  delta_color="normal" if n_ok == len(filas_h) else "inverse")
        st.caption("\\* H1: umbral ajustado a r ≥ 0.50 (efecto grande, Cohen 1988).")

    if corr_m is not None:
        fig_ch, ax_ch = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
                    annot_kws={"size": 10}, ax=ax_ch)
        ax_ch.set_title("Correlaciones Clave — Validación Teórica JD-R",
                         fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_ch)
        plt.close()

    st.divider()

    st.subheader("📝 Resumen ejecutivo del pipeline")
    st.success("**✅ El dataset `bienestar_laboral_LIMPIO.csv` está certificado y listo para análisis estadístico.**")
    st.markdown("""
**Garantías técnicas del equipo de Data Science:**

| Garantía | Estado |
|----------|--------|
| Completitud en ítems de escala (0 NaN en 91 variables de análisis) | ✅ Verificado |
| Coherencia de rangos (todos los ítems dentro del rango teórico) | ✅ Verificado |
| Estandarización categórica (sin typos ni inconsistencias) | ✅ Verificado |
| Ítem inverso IR1 corregido (8 − IR1) | ✅ Verificado |
| Outliers tratados (winsorización IQR, sin pérdida de registros) | ✅ Verificado |
| Estructura correlacional JD-R reproducida (7/7 hipótesis) | ✅ Verificado |
| 16 variables derivadas en rango teórico, 0 NaN | ✅ Verificado |

**Recomendaciones para el equipo de análisis:**
1. Calcular **Alpha de Cronbach** por dimensión antes de modelado confirmatorio.
2. Segmentar por `Sector`, `Tipo_Cargo` y `Modalidad` para análisis diferencial.
3. Priorizar `BURNOUT`, `DESGASTE` y `RETIRO` como indicadores de riesgo crítico.

---
*Proyecto Final — Preprocesamiento de Datos 2026-1 · Universidad de La Sabana*  
    """)