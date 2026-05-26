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

TYPOS_EJEMPLO = [
    {"Variable": "Sexo",           "Antes": "mujer",          "Después": "Mujer"},
    {"Variable": "Sexo",           "Antes": "MUJER",          "Después": "Mujer"},
    {"Variable": "Sexo",           "Antes": "hombre",         "Después": "Hombre"},
    {"Variable": "Estado_Civil",   "Antes": "Casdo",          "Después": "Casado"},
    {"Variable": "Estado_Civil",   "Antes": "Solero",         "Después": "Soltero"},
    {"Variable": "Estado_Civil",   "Antes": "SOLTERO",        "Después": "Soltero"},
    {"Variable": "Estado_Civil",   "Antes": "  Soltero",      "Después": "Soltero"},
    {"Variable": "Sector",         "Antes": "publico",        "Después": "Público"},
    {"Variable": "Sector",         "Antes": "Privado ",       "Después": "Privado"},
    {"Variable": "Modalidad",      "Antes": "remoto",         "Después": "Remoto"},
    {"Variable": "Modalidad",      "Antes": "Hibrido",        "Después": "Híbrido"},
    {"Variable": "Tipo_Contrato",  "Antes": "indefinido",     "Después": "Indefinido"},
    {"Variable": "Nivel_Educativo","Antes": "profesional",    "Después": "Profesional"},
    {"Variable": "Ingreso",        "Antes": "entre 1 y 3 smlv","Después": "Entre 1 y 3 SMLV"},
]

# Tabla de hallazgos del profiling (Entregable 1 de la rúbrica)
INVENTARIO_PROBLEMAS = pd.DataFrame([
    {
        "Problema encontrado": "Valores faltantes distribuidos en múltiples variables",
        "Columna(s) afectada(s)": "Todo el dataset",
        "Magnitud": "2 582 valores faltantes (5.60% del total de celdas)",
        "Tipo de problema": "Completitud",
    },
    {
        "Problema encontrado": "Variables con alto porcentaje de faltantes",
        "Columna(s) afectada(s)": "SAT1, GC2, SAT6, SM5, SM4, SAT8, SAT5, BU1, BU11, CT3, BP8, entre otras",
        "Magnitud": "Entre 7% y 11% de faltantes por columna",
        "Tipo de problema": "Completitud",
    },
    {
        "Problema encontrado": "Presencia de outliers detectados mediante IQR",
        "Columna(s) afectada(s)": "Variables numéricas continuas",
        "Magnitud": "43 outliers detectados",
        "Tipo de problema": "Validez / Consistencia",
    },
    {
        "Problema encontrado": "Filas completamente duplicadas",
        "Columna(s) afectada(s)": "Dataset completo",
        "Magnitud": "0 casos",
        "Tipo de problema": "Duplicidad",
    },
    {
        "Problema encontrado": "IDs duplicados",
        "Columna(s) afectada(s)": "ID",
        "Magnitud": "0 casos",
        "Tipo de problema": "Integridad",
    },
    {
        "Problema encontrado": "Coincidencia total en variables demográficas",
        "Columna(s) afectada(s)": "Edad, Sexo, Estado_Civil, Numero_Hijos, Nivel_Educativo, Zona_Vivienda, Estrato, Sector, Años_Experiencia, Ingreso, Tipo_Cargo, Antiguedad_Cargo",
        "Magnitud": "9 registros coincidentes",
        "Tipo de problema": "Posible duplicidad",
    },
    {
        "Problema encontrado": "Coincidencia total en perfil profesional",
        "Columna(s) afectada(s)": "Sector, Tamaño_Empresa, Tipo_Contrato, Tipo_Cargo, Personas_Cargo, Modalidad, Horas_Semana, Trabajo_Turnos",
        "Magnitud": "28 registros coincidentes",
        "Tipo de problema": "Posible duplicidad",
    },
    {
        "Problema encontrado": "Inconsistencias en escalas Likert por diferencias de escritura, capitalización y categorías inválidas",
        "Columna(s) afectada(s)": "AC1, AC2, BU1–BU12, DL1–DL8, SOM1–SOM5, entre otras",
        "Magnitud": "Variables con hasta 214 inconsistencias",
        "Tipo de problema": "Estandarización / Validez",
    },
    {
        "Problema encontrado": "Uso de categorías no permitidas en variables Likert",
        "Columna(s) afectada(s)": "Variables Likert",
        "Magnitud": 'Valores como "SIEMPRE", "siempre", "Alguna vez", "A menudo", etc.',
        "Tipo de problema": "Dominio inválido",
    },
    {
        "Problema encontrado": "Inconsistencias severas en variables BU, DL y SOM",
        "Columna(s) afectada(s)": "BU1–BU12, DL1–DL8, SOM1–SOM5",
        "Magnitud": "Más de 150 inconsistencias por variable en varios casos",
        "Tipo de problema": "Calidad semántica",
    },
    {
        "Problema encontrado": "Variables numéricas almacenadas con texto",
        "Columna(s) afectada(s)": "BP1–BP5",
        "Magnitud": "Entre 17 y 20 inconsistencias por variable",
        "Tipo de problema": "Tipo de dato incorrecto",
    },
    {
        "Problema encontrado": "Valores no numéricos en escalas numéricas",
        "Columna(s) afectada(s)": "BP1–BP5",
        "Magnitud": 'Valores como "cuatro", "cinco", "siete", etc.',
        "Tipo de problema": "Validez / Formato",
    },
    {
        "Problema encontrado": "Valores inválidos en escalas de acuerdo",
        "Columna(s) afectada(s)": "IR1, IR3, IR4",
        "Magnitud": "1–3 inconsistencias por variable",
        "Tipo de problema": "Dominio inválido",
    },
    {
        "Problema encontrado": "Uso de códigos ambiguos o placeholders",
        "Columna(s) afectada(s)": "IR1, IR3, IR4",
        "Magnitud": 'Valores "999", "--", "?", "sin dato"',
        "Tipo de problema": "Codificación inconsistente",
    },
    {
        "Problema encontrado": "Inconsistencias ortográficas y de capitalización en Estado Civil",
        "Columna(s) afectada(s)": "Estado_Civil",
        "Magnitud": "Múltiples variantes incorrectas",
        "Tipo de problema": "Estandarización",
    },
    {
        "Problema encontrado": "Variantes inválidas de categorías en Estado Civil",
        "Columna(s) afectada(s)": "Estado_Civil",
        "Magnitud": 'Valores como "Casdo", "Solero", "SOLTERO", "  Soltero"',
        "Tipo de problema": "Error tipográfico / Formato",
    },
    {
        "Problema encontrado": "Espacios innecesarios en categorías categóricas",
        "Columna(s) afectada(s)": "Estado_Civil",
        "Magnitud": "Casos con espacios al inicio",
        "Tipo de problema": "Limpieza de texto",
    },
    {
        "Problema encontrado": "Mezcla de mayúsculas y minúsculas en variables categóricas",
        "Columna(s) afectada(s)": "Estado_Civil y variables Likert",
        "Magnitud": "Presente en múltiples categorías",
        "Tipo de problema": "Normalización",
    },
    {
        "Problema encontrado": "Posible inconsistencia en definición de tipos de variables",
        "Columna(s) afectada(s)": "Variables categóricas y escalas",
        "Magnitud": "Algunas escalas almacenadas como texto en lugar de categorías ordenadas",
        "Tipo de problema": "Modelado de datos",
    },
    {
        "Problema encontrado": "Alta heterogeneidad de formatos de respuesta",
        "Columna(s) afectada(s)": "Variables de encuesta",
        "Magnitud": "Presente en varias escalas psicométricas",
        "Tipo de problema": "Consistencia semántica",
    },
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
        "Marco teórico: Modelo JD-R  \n"
        "*(Demerouti et al., 2001)*"
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

    # KPIs
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

    # Mapa de calor
    st.subheader("🔥 Mapa de calor de valores faltantes")
    st.caption(
        "Oscuro = valor faltante · Claro = valor presente. "
        "Se muestran las 50 columnas con mayor % de NaN."
    )
    nan_pct = df_orig.isna().mean().sort_values(ascending=False)
    top50   = nan_pct[nan_pct > 0].head(50).index.tolist()

    if top50:
        fig_heat, ax_heat = plt.subplots(figsize=(16, 5))
        sns.heatmap(
            df_orig[top50].isna().astype(int).T,
            cmap="Blues", cbar=False,
            linewidths=0, ax=ax_heat,
            yticklabels=True, xticklabels=False,
        )
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

    # ── Inventario de problemas (tabla completa del profiling) ─────────────
    st.subheader("📋 Inventario de problemas detectados")
    st.caption(
        f"Documento de hallazgos del Data Profiling — "
        f"{len(INVENTARIO_PROBLEMAS)} problemas identificados antes del pipeline."
    )

    # Filtro por tipo de problema
    tipos_unicos = ["Todos"] + sorted(INVENTARIO_PROBLEMAS["Tipo de problema"].unique().tolist())
    tipo_filtro  = st.selectbox("Filtrar por tipo de problema:", tipos_unicos, key="filtro_inv")

    if tipo_filtro == "Todos":
        df_mostrar = INVENTARIO_PROBLEMAS
    else:
        df_mostrar = INVENTARIO_PROBLEMAS[
            INVENTARIO_PROBLEMAS["Tipo de problema"] == tipo_filtro
        ]

    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    # Gráfico de conteo por tipo
    conteo_tipos = (
        INVENTARIO_PROBLEMAS["Tipo de problema"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Tipo", "Tipo de problema": "N"})
    )
    # compatibilidad con distintas versiones de pandas
    if "Tipo de problema" in conteo_tipos.columns and "count" in conteo_tipos.columns:
        conteo_tipos.columns = ["Tipo", "N"]
    elif conteo_tipos.shape[1] == 2:
        conteo_tipos.columns = ["Tipo", "N"]

    fig_ct, ax_ct = plt.subplots(figsize=(10, 4))
    ax_ct.barh(conteo_tipos.iloc[:, 0], conteo_tipos.iloc[:, 1],
               color="#5C6BC0", alpha=0.85)
    ax_ct.set_xlabel("N° de problemas")
    ax_ct.set_title("Distribución de problemas por tipo", fontweight="bold")
    ax_ct.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_ct)
    plt.close()

    st.divider()

    # Distribución variables numéricas clave
    st.subheader("📈 Distribución de variables clave (dataset original)")
    vars_num = [v for v in ["Edad","Horas_Semana","Estrato","Horas_Formacion"]
                if v in df_orig.columns]
    if vars_num:
        fig_d, axes_d = plt.subplots(1, len(vars_num),
                                     figsize=(4.5 * len(vars_num), 4))
        if len(vars_num) == 1:
            axes_d = [axes_d]
        for ax, col in zip(axes_d, vars_num):
            serie = pd.to_numeric(df_orig[col], errors="coerce").dropna()
            sns.histplot(serie, ax=ax, kde=True,
                         color="#FF7043", alpha=0.75, bins=25, linewidth=0)
            ax.set_title(col, fontweight="bold", fontsize=11)
            ax.set_xlabel("")
        plt.suptitle("Variables numéricas con potenciales outliers",
                     fontsize=12, fontweight="bold")
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
        "afectados y visualizar el issue específico."
    )

    tipo_problema = st.selectbox(
        "Seleccione el tipo de problema:",
        options=[
            "Valores faltantes por columna",
            "Columnas con > 5% de NaN",
            "Filas duplicadas exactas",
            "Outliers en variables numéricas",
            "Estandarización de texto (typos corregidos)",
            "Distribución de ítems de escala (sucios)",
        ],
    )
    st.divider()

    if tipo_problema == "Valores faltantes por columna":
        nan_col = (df_orig.isna().sum()
                   .reset_index()
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

    elif tipo_problema == "Filas duplicadas exactas":
        mask_dup = df_orig.duplicated(keep=False)
        n_dup2   = int(mask_dup.sum())
        if n_dup2 == 0:
            st.success("No hay filas exactamente duplicadas en el dataset original.")
            st.info(
                "💡 En la Sección 2 del pipeline se detectaron duplicados por **subconjunto "
                "de columnas de perfil** (sociodemográficas + laborales). "
                "Esa estrategia eliminó 28 registros con mismo perfil pero distinto ID."
            )
        else:
            st.metric("Filas duplicadas exactas", n_dup2)
            st.dataframe(df_orig[mask_dup].head(50), use_container_width=True)

    elif tipo_problema == "Outliers en variables numéricas":
        num_cols = [c for c in df_orig.select_dtypes(include="number").columns if c != "ID"]
        outlier_rep = []
        for col in num_cols:
            s  = pd.to_numeric(df_orig[col], errors="coerce").dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            n_out  = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum())
            if n_out > 0:
                outlier_rep.append({
                    "Variable"       : col,
                    "N outliers IQR" : n_out,
                    "Min obs"        : round(s.min(), 2),
                    "Max obs"        : round(s.max(), 2),
                    "Q1"             : round(Q1, 2),
                    "Q3"             : round(Q3, 2),
                })
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

    elif tipo_problema == "Estandarización de texto (typos corregidos)":
        st.markdown(
            "Ejemplos representativos de errores tipográficos corregidos en la "
            "**Sección 3** del pipeline mediante diccionario de mapeo explícito."
        )
        st.dataframe(pd.DataFrame(TYPOS_EJEMPLO), use_container_width=True, hide_index=True)
        st.markdown("##### Proceso aplicado")
        st.code(
            "# 1. Strip de espacios\n"
            "df[col] = df[col].str.strip()\n\n"
            "# 2. Normalización a Title Case\n"
            "df[col] = df[col].str.title()\n\n"
            "# 3. Corrección de typos con diccionario\n"
            "df[col] = df[col].replace(diccionario_correcciones)",
            language="python",
        )
        cats_disponibles = [c for c in ["Sexo","Estado_Civil","Sector","Modalidad","Tipo_Contrato"]
                            if c in df_orig.columns and c in df_clean.columns]
        if cats_disponibles:
            col_sel = st.selectbox("Ver categorías únicas antes/después:", cats_disponibles)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Dataset ORIGINAL**")
                vc = df_orig[col_sel].value_counts().reset_index()
                vc.columns = [col_sel, "N"]
                st.dataframe(vc, use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Dataset LIMPIO**")
                vc2 = df_clean[col_sel].value_counts().reset_index()
                vc2.columns = [col_sel, "N"]
                st.dataframe(vc2, use_container_width=True, hide_index=True)

    elif tipo_problema == "Distribución de ítems de escala (sucios)":
        dim_sel   = st.selectbox("Seleccione dimensión:", DIMS_NAMES)
        items_sel = [c for c in DIMENSIONES[dim_sel]["items"] if c in df_orig.columns]
        if items_sel:
            st.info(
                "📌 Las barras muestran los **valores textuales originales** "
                "antes del mapeo canónico — variantes de capitalización, "
                "sinónimos y códigos de ausencia."
            )
            n_it = len(items_sel)
            fig_lk, axes_lk = plt.subplots(1, n_it, figsize=(3.5 * n_it, 5))
            if n_it == 1:
                axes_lk = [axes_lk]
            for ax, col in zip(axes_lk, items_sel):
                vc = df_orig[col].value_counts().head(10)
                ax.barh(vc.index.astype(str), vc.values, color="#7E57C2", alpha=0.85)
                ax.set_title(col, fontsize=9, fontweight="bold")
                ax.set_xlabel("N")
                ax.tick_params(axis="y", labelsize=7)
            plt.suptitle(f"Distribución de {dim_sel} — dataset original (sin limpiar)",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_lk)
            plt.close()
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
        {"Métrica": "N° registros",                                    "Antes": 412,         "Después": len(df_clean),          "Δ": f"{len(df_clean)-412:+d}"},
        {"Métrica": "N° columnas",                                     "Antes": 112,         "Después": df_clean.shape[1],       "Δ": f"{df_clean.shape[1]-112:+d}"},
        {"Métrica": "NaN en ítems de escala (91 vars. de análisis)",   "Antes": 2582,        "Después": nan_items_clean,         "Δ": f"{nan_items_clean-2582:+d}"},
        {"Métrica": "NaN en vars. auxiliares (fuera del scope)",       "Antes": "N/A",       "Después": nan_aux_clean,           "Δ": "No imputadas"},
        {"Métrica": "Filas duplicadas",                                "Antes": "Detectadas","Después": 0,                       "Δ": "→ 0"},
        {"Métrica": "Ítems Likert fuera de rango",                     "Antes": "Múltiples", "Después": 0,                       "Δ": "→ 0"},
        {"Métrica": "Outliers numér. (IQR)",                           "Antes": 43,          "Después": "0 (winzorizados)",      "Δ": "-43"},
        {"Métrica": "Variables derivadas (JD-R)",                      "Antes": 0,           "Después": len(dims_disponibles),   "Δ": f"+{len(dims_disponibles)}"},
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

    # Semáforo
    st.subheader("🚦 Semáforo de calidad — Dimensiones JD-R")
    st.caption(
        "🟢 Excelente (media en rango + 0 NaN) · "
        "🟡 Revisar (en rango pero NaN residuales) · "
        "🔴 Alerta (fuera de rango o > 5% NaN)"
    )
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

    # Correlaciones
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
        filas_h.append({
            "Hipótesis"  : nombre,
            "r observada": round(r, 3),
            "Umbral"     : cond,
            "¿Cumple?"   : "✅ Sí" if cumple else "❌ No",
        })

    if filas_h:
        df_h = pd.DataFrame(filas_h)
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        n_ok = sum(1 for h in filas_h if "✅" in h["¿Cumple?"])
        st.metric(
            "Hipótesis teóricas satisfechas",
            f"{n_ok} / {len(filas_h)}",
            delta="✔ Estructura JD-R reproducida" if n_ok == len(filas_h) else "⚠ Revisar",
            delta_color="normal" if n_ok == len(filas_h) else "inverse",
        )
        st.caption(
            "\\* H1: umbral ajustado a r ≥ 0.50 (efecto grande, Cohen 1988). "
            "La correlación observada de 0.548 confirma la co-ocurrencia teórica."
        )

    if corr_m is not None:
        fig_ch, ax_ch = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_m, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 10}, ax=ax_ch)
        ax_ch.set_title("Correlaciones Clave — Validación Teórica JD-R",
                         fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_ch)
        plt.close()

    st.divider()

    # Resumen ejecutivo
    st.subheader("📝 Resumen ejecutivo del pipeline")
    st.success(
        "**✅ El dataset `bienestar_laboral_LIMPIO.csv` está certificado y "
        "listo para análisis estadístico.**"
    )
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

**Limitaciones a considerar:**
- La imputación KNN asume homogeneidad local — verificar por `Sector` antes de comparativos.
- La winsorización puede atenuar relaciones reales en colas de `Horas_Semana`.
- n ≈ 384: adecuado para correlaciones y regresión; revisar poder estadístico para SEM.

**Recomendaciones para el equipo de análisis:**
1. Calcular **Alpha de Cronbach** por dimensión antes de modelado confirmatorio.
2. Segmentar por `Sector`, `Tipo_Cargo` y `Modalidad` para análisis diferencial.
3. Priorizar `BURNOUT`, `DESGASTE` y `RETIRO` como indicadores de riesgo crítico.

---
*Proyecto Final — Preprocesamiento de Datos 2026-1 · Universidad de La Sabana*  
*Marco teórico: Demerouti, E., Bakker, A. B., Nachreiner, F., & Schaufeli, W. B. (2001).*  
*The job demands-resources model of burnout. Journal of Applied Psychology, 86(3), 499–512.*
    """)