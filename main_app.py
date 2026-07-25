"""Dashboard de energía renovable para Streamlit.

Ejecución local:
    pip install -r requirements.txt
    streamlit run app.py

El archivo ``energia_renovable.csv`` debe estar en la misma carpeta. Si no
está disponible, la aplicación permite cargarlo desde la barra lateral.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA | Energías renovables",
    page_icon="⚡",
    layout="wide",
)

RUTA_DATOS = Path(__file__).resolve().parent / "energia_renovable.csv"

# Paleta accesible y consistente para todas las visualizaciones.
COLORES_TECNOLOGIA = {
    "Solar Fotovoltaica": "#E69F00",
    "Eólica Onshore": "#0072B2",
    "Pequeña Hidroeléctrica (PCH)": "#009E73",
    "Biomasa": "#CC79A7",
}

COLUMNAS_REQUERIDAS = {
    "ID_Proyecto",
    "Tecnologia",
    "Operador",
    "Capacidad_Instalada_MW",
    "Generacion_Diaria_MWh",
    "Eficiencia_Planta_Pct",
    "Conectado_SIN",
    "Estado_Actual",
    "Inversion_Inicial_MUSD",
    "Fecha_Entrada_Operacion",
}


# ---------------------------------------------------------------------------
# 2. Funciones de carga, validación y transformación
# ---------------------------------------------------------------------------
def preparar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Valida las columnas y crea las variables necesarias para el EDA."""
    faltantes = COLUMNAS_REQUERIDAS.difference(datos.columns)
    if faltantes:
        columnas = ", ".join(sorted(faltantes))
        raise ValueError(f"El CSV no contiene estas columnas requeridas: {columnas}")

    df = datos.copy()

    # Convertimos explícitamente las variables para evitar errores silenciosos.
    columnas_numericas = [
        "Capacidad_Instalada_MW",
        "Generacion_Diaria_MWh",
        "Eficiencia_Planta_Pct",
        "Inversion_Inicial_MUSD",
    ]
    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df["Fecha_Entrada_Operacion"] = pd.to_datetime(
        df["Fecha_Entrada_Operacion"], errors="coerce"
    )

    # KPI principal: cuánta generación diaria se obtiene por cada MUSD invertido.
    df["Generacion_por_MUSD"] = np.where(
        df["Inversion_Inicial_MUSD"] > 0,
        df["Generacion_Diaria_MWh"] / df["Inversion_Inicial_MUSD"],
        np.nan,
    )

    # Métrica inversa: inversión asociada a una unidad de generación diaria.
    df["MUSD_por_MWh_Dia"] = np.where(
        df["Generacion_Diaria_MWh"] > 0,
        df["Inversion_Inicial_MUSD"] / df["Generacion_Diaria_MWh"],
        np.nan,
    )

    # Control de plausibilidad física. Un resultado superior a 1 requiere revisión.
    df["Factor_Capacidad_Aparente"] = np.where(
        df["Capacidad_Instalada_MW"] > 0,
        df["Generacion_Diaria_MWh"] / (df["Capacidad_Instalada_MW"] * 24),
        np.nan,
    )
    df["Anio_Entrada"] = df["Fecha_Entrada_Operacion"].dt.year.astype("Int64")
    return df


@st.cache_data(show_spinner=False)
def cargar_datos_locales(ruta: str) -> pd.DataFrame:
    """Carga y prepara el CSV local una sola vez por sesión."""
    return preparar_datos(pd.read_csv(ruta, encoding="utf-8"))


def resumen_por_tecnologia(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el reporte que responde la pregunta de negocio."""
    resumen = (
        df.groupby("Tecnologia", as_index=False)
        .agg(
            Proyectos=("ID_Proyecto", "count"),
            Capacidad_Total_MW=("Capacidad_Instalada_MW", "sum"),
            Generacion_Total_MWh_Dia=("Generacion_Diaria_MWh", "sum"),
            Inversion_Total_MUSD=("Inversion_Inicial_MUSD", "sum"),
            Eficiencia_Media_Pct=("Eficiencia_Planta_Pct", "mean"),
            Mediana_Proyecto_MWh_Dia_por_MUSD=("Generacion_por_MUSD", "median"),
        )
    )
    resumen["MWh_Dia_por_MUSD"] = np.where(
        resumen["Inversion_Total_MUSD"] > 0,
        resumen["Generacion_Total_MWh_Dia"] / resumen["Inversion_Total_MUSD"],
        np.nan,
    )
    resumen["MUSD_por_MWh_Dia"] = np.where(
        resumen["Generacion_Total_MWh_Dia"] > 0,
        resumen["Inversion_Total_MUSD"] / resumen["Generacion_Total_MWh_Dia"],
        np.nan,
    )
    return resumen.sort_values("MWh_Dia_por_MUSD", ascending=False)


def resumen_por_operador(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa capacidad, generación e inversión por operador."""
    return (
        df.groupby("Operador", as_index=False)
        .agg(
            Proyectos=("ID_Proyecto", "count"),
            Capacidad_Total_MW=("Capacidad_Instalada_MW", "sum"),
            Generacion_Total_MWh_Dia=("Generacion_Diaria_MWh", "sum"),
            Inversion_Total_MUSD=("Inversion_Inicial_MUSD", "sum"),
        )
        .sort_values("Capacidad_Total_MW", ascending=False)
    )


def resumen_por_estado(df: pd.DataFrame) -> pd.DataFrame:
    """Resume el portafolio según su estado actual."""
    return (
        df.groupby("Estado_Actual", as_index=False)
        .agg(
            Proyectos=("ID_Proyecto", "count"),
            Capacidad_Total_MW=("Capacidad_Instalada_MW", "sum"),
            Generacion_Total_MWh_Dia=("Generacion_Diaria_MWh", "sum"),
            Inversion_Total_MUSD=("Inversion_Inicial_MUSD", "sum"),
        )
        .sort_values("Generacion_Total_MWh_Dia", ascending=False)
    )


def formato_numero(valor: float, decimales: int = 0) -> str:
    """Presenta números con separadores habituales en español."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def convertir_csv(df: pd.DataFrame) -> bytes:
    """Genera un CSV compatible con Excel y listo para descargar."""
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# 3. Encabezado, carga de datos y filtros globales
# ---------------------------------------------------------------------------
st.title("⚡ Energías renovables: inversión y generación")
st.caption(
    "Pregunta de negocio: ¿qué tecnología presenta la mejor relación entre "
    "inversión inicial y generación diaria?"
)

archivo_subido = st.sidebar.file_uploader(
    "Cargar energia_renovable.csv",
    type="csv",
    help="Es opcional si el CSV ya está en la misma carpeta de app.py.",
)

try:
    if archivo_subido is not None:
        df = preparar_datos(pd.read_csv(archivo_subido, encoding="utf-8"))
        origen_datos = "archivo cargado por el usuario"
    elif RUTA_DATOS.exists():
        df = cargar_datos_locales(str(RUTA_DATOS))
        origen_datos = RUTA_DATOS.name
    else:
        st.error(
            "No se encontró energia_renovable.csv. Cárgalo desde la barra lateral "
            "o ubícalo junto a app.py."
        )
        st.stop()
except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"No fue posible preparar el archivo: {error}")
    st.stop()

st.sidebar.header("Filtros")
tecnologias = sorted(df["Tecnologia"].dropna().unique())
operadores = sorted(df["Operador"].dropna().unique())
estados = sorted(df["Estado_Actual"].dropna().unique())

tecnologias_seleccionadas = st.sidebar.multiselect(
    "Tecnología", tecnologias, default=tecnologias
)
operadores_seleccionados = st.sidebar.multiselect(
    "Operador", operadores, default=operadores
)
estados_seleccionados = st.sidebar.multiselect(
    "Estado actual", estados, default=estados
)

fecha_minima = df["Fecha_Entrada_Operacion"].min().date()
fecha_maxima = df["Fecha_Entrada_Operacion"].max().date()
intervalo_fechas = st.sidebar.date_input(
    "Fecha de entrada en operación",
    value=(fecha_minima, fecha_maxima),
    min_value=fecha_minima,
    max_value=fecha_maxima,
)

if isinstance(intervalo_fechas, (tuple, list)) and len(intervalo_fechas) == 2:
    fecha_inicio, fecha_fin = intervalo_fechas
else:
    fecha_inicio = fecha_fin = intervalo_fechas

df_filtrado = df[
    df["Tecnologia"].isin(tecnologias_seleccionadas)
    & df["Operador"].isin(operadores_seleccionados)
    & df["Estado_Actual"].isin(estados_seleccionados)
    & df["Fecha_Entrada_Operacion"].between(
        pd.Timestamp(fecha_inicio), pd.Timestamp(fecha_fin)
    )
].copy()

st.sidebar.caption(f"Fuente activa: {origen_datos}")
st.sidebar.caption(f"{len(df_filtrado):,} de {len(df):,} proyectos visibles")

if df_filtrado.empty:
    st.warning("Los filtros seleccionados no contienen proyectos.")
    st.stop()

reporte_tecnologia = resumen_por_tecnologia(df_filtrado)
reporte_operador = resumen_por_operador(df_filtrado)
reporte_estado = resumen_por_estado(df_filtrado)

lider = reporte_tecnologia.iloc[0]
segundo_valor = (
    reporte_tecnologia.iloc[1]["MWh_Dia_por_MUSD"]
    if len(reporte_tecnologia) > 1
    else np.nan
)
ventaja_lider = (
    (lider["MWh_Dia_por_MUSD"] / segundo_valor - 1) * 100
    if pd.notna(segundo_valor) and segundo_valor > 0
    else np.nan
)


# ---------------------------------------------------------------------------
# 4. Contenido principal
# ---------------------------------------------------------------------------
tab_resumen, tab_eda, tab_calidad, tab_reportes = st.tabs(
    ["Resumen ejecutivo", "Análisis exploratorio", "Calidad y datos", "Reportes"]
)

with tab_resumen:
    st.subheader("Panorama del portafolio filtrado")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Proyectos", formato_numero(len(df_filtrado)))
    kpi2.metric(
        "Capacidad instalada",
        f"{formato_numero(df_filtrado['Capacidad_Instalada_MW'].sum())} MW",
    )
    kpi3.metric(
        "Generación diaria",
        f"{formato_numero(df_filtrado['Generacion_Diaria_MWh'].sum())} MWh",
    )
    kpi4.metric(
        "Inversión inicial",
        f"USD {formato_numero(df_filtrado['Inversion_Inicial_MUSD'].sum(), 1)} M",
    )
    kpi5.metric(
        "Mejor productividad",
        f"{formato_numero(lider['MWh_Dia_por_MUSD'], 1)} MWh/MUSD",
        delta=(
            f"{formato_numero(ventaja_lider, 1)} % vs. segundo"
            if pd.notna(ventaja_lider)
            else None
        ),
    )

    st.success(
        f"Hallazgo principal: **{lider['Tecnologia']}** lidera el filtro actual "
        f"con **{lider['MWh_Dia_por_MUSD']:.1f} MWh/día por MUSD invertido**. "
        "La comparación usa la razón entre generación total e inversión total "
        "para reducir la influencia de proyectos individuales extremos."
    )

    columna_a, columna_b = st.columns(2)

    with columna_a:
        st.markdown("#### Productividad de la inversión por tecnología")
        orden_ratio = reporte_tecnologia.sort_values("MWh_Dia_por_MUSD")
        colores = [
            "#0B7A53" if tecnologia == lider["Tecnologia"] else "#B8C2CC"
            for tecnologia in orden_ratio["Tecnologia"]
        ]
        fig_ratio = px.bar(
            orden_ratio,
            x="MWh_Dia_por_MUSD",
            y="Tecnologia",
            orientation="h",
            text="MWh_Dia_por_MUSD",
            labels={
                "MWh_Dia_por_MUSD": "MWh/día por MUSD",
                "Tecnologia": "Tecnología",
            },
        )
        fig_ratio.update_traces(
            marker_color=colores,
            texttemplate="%{text:.1f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Productividad: %{x:.2f}<extra></extra>",
        )
        fig_ratio.update_layout(showlegend=False, margin=dict(l=10, r=25, t=10, b=10))
        st.plotly_chart(fig_ratio, width="stretch")
        st.caption(
            "Lectura: una barra más larga indica más generación diaria asociada "
            "a cada millón de dólares invertido."
        )

    with columna_b:
        st.markdown("#### Capacidad instalada por operador")
        fig_operador, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=reporte_operador,
            x="Capacidad_Total_MW",
            y="Operador",
            color="#2A6FBB",
            ax=ax,
        )
        ax.set(
            title="Capacidad instalada acumulada",
            xlabel="Capacidad instalada (MW)",
            ylabel="",
        )
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=3, fontsize=8)
        sns.despine(ax=ax)
        fig_operador.tight_layout()
        st.pyplot(fig_operador)
        plt.close(fig_operador)
        operador_lider = reporte_operador.iloc[0]
        st.caption(
            f"Lectura: {operador_lider['Operador']} reúne la mayor capacidad del "
            f"filtro ({operador_lider['Capacidad_Total_MW']:,.0f} MW). "
            "La capacidad no equivale automáticamente a mayor generación."
        )

    st.info(
        "Decisión sugerida: usar la productividad agregada como señal de "
        "priorización, pero validar primero que la generación corresponda a datos "
        "reales comparables y no a proyecciones de proyectos en planeación."
    )

with tab_eda:
    st.subheader("Relaciones, distribuciones y evolución del portafolio")

    st.markdown("#### 1. Inversión vs. generación diaria — Plotly")
    correlacion = df_filtrado["Inversion_Inicial_MUSD"].corr(
        df_filtrado["Generacion_Diaria_MWh"]
    )
    fig_dispersion = px.scatter(
        df_filtrado,
        x="Inversion_Inicial_MUSD",
        y="Generacion_Diaria_MWh",
        color="Tecnologia",
        size="Capacidad_Instalada_MW",
        hover_name="ID_Proyecto",
        hover_data={
            "Operador": True,
            "Estado_Actual": True,
            "Capacidad_Instalada_MW": ":.2f",
            "Generacion_por_MUSD": ":.2f",
        },
        color_discrete_map=COLORES_TECNOLOGIA,
        size_max=38,
        labels={
            "Inversion_Inicial_MUSD": "Inversión inicial (MUSD)",
            "Generacion_Diaria_MWh": "Generación diaria (MWh)",
            "Tecnologia": "Tecnología",
        },
        title=f"Relación lineal global: r = {correlacion:.2f}",
    )
    fig_dispersion.update_layout(legend_title_text="Tecnología")
    st.plotly_chart(fig_dispersion, width="stretch")
    st.caption(
        "Cada punto es un proyecto y el tamaño representa su capacidad. Una "
        f"correlación de {correlacion:.2f} describe asociación lineal, no causalidad."
    )

    columna_c, columna_d = st.columns(2)
    with columna_c:
        st.markdown("#### 2. Productividad por proyecto — Seaborn")
        fig_box, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_filtrado,
            x="Tecnologia",
            y="Generacion_por_MUSD",
            hue="Tecnologia",
            palette=COLORES_TECNOLOGIA,
            legend=False,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set(
            xlabel="",
            ylabel="MWh/día por MUSD (escala logarítmica)",
            title="Dispersión de la productividad individual",
        )
        ax.tick_params(axis="x", rotation=25)
        sns.despine(ax=ax)
        fig_box.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)
        st.caption(
            "La escala logarítmica permite observar la mediana y los valores "
            "extremos sin ocultar la mayoría de los proyectos."
        )

    with columna_d:
        st.markdown("#### 3. Distribución de generación — Pyplot")
        fig_hist, ax = plt.subplots(figsize=(8, 5))
        for tecnologia, grupo in df_filtrado.groupby("Tecnologia"):
            ax.hist(
                grupo["Generacion_Diaria_MWh"],
                bins=18,
                alpha=0.45,
                label=tecnologia,
                color=COLORES_TECNOLOGIA.get(tecnologia),
            )
        ax.axvline(
            df_filtrado["Generacion_Diaria_MWh"].median(),
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="Mediana global",
        )
        ax.set(
            xlabel="Generación diaria (MWh)",
            ylabel="Número de proyectos",
            title="Distribución de generación por tecnología",
        )
        ax.legend(fontsize=8)
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)
        st.caption(
            "El histograma muestra si una tecnología concentra proyectos en "
            "niveles bajos, medios o altos de generación."
        )

    st.markdown("#### 4. Capacidad incorporada por año — Plotly")
    temporal = (
        df_filtrado.dropna(subset=["Anio_Entrada"])
        .groupby(["Anio_Entrada", "Tecnologia"], as_index=False)
        .agg(Capacidad_Incorporada_MW=("Capacidad_Instalada_MW", "sum"))
    )
    fig_temporal = px.line(
        temporal,
        x="Anio_Entrada",
        y="Capacidad_Incorporada_MW",
        color="Tecnologia",
        markers=True,
        color_discrete_map=COLORES_TECNOLOGIA,
        labels={
            "Anio_Entrada": "Año de entrada",
            "Capacidad_Incorporada_MW": "Capacidad incorporada (MW)",
            "Tecnologia": "Tecnología",
        },
    )
    fig_temporal.update_xaxes(dtick=1)
    st.plotly_chart(fig_temporal, width="stretch")
    st.caption(
        "Esta vista describe la capacidad asociada a la fecha de entrada registrada; "
        "no representa una serie histórica de generación."
    )

with tab_calidad:
    st.subheader("Diagnóstico de calidad y plausibilidad")
    nulos = int(df_filtrado.isna().sum().sum())
    duplicados = int(df_filtrado.duplicated().sum())
    factores_imposibles = int((df_filtrado["Factor_Capacidad_Aparente"] > 1).sum())
    estados_tempranos = ["En Planeación", "En Construcción"]
    generacion_en_etapa_temprana = int(
        (
            df_filtrado["Estado_Actual"].isin(estados_tempranos)
            & (df_filtrado["Generacion_Diaria_MWh"] > 0)
        ).sum()
    )

    calidad1, calidad2, calidad3, calidad4 = st.columns(4)
    calidad1.metric("Valores nulos", formato_numero(nulos))
    calidad2.metric("Filas duplicadas", formato_numero(duplicados))
    calidad3.metric("Factor aparente > 100 %", formato_numero(factores_imposibles))
    calidad4.metric(
        "Generación en planeación/construcción",
        formato_numero(generacion_en_etapa_temprana),
    )

    if factores_imposibles > 0:
        porcentaje = factores_imposibles / len(df_filtrado) * 100
        st.warning(
            f"{factores_imposibles} proyectos ({porcentaje:.1f} %) superan un "
            "factor de capacidad aparente de 100 %. Antes de usar el KPI en una "
            "decisión financiera, confirma las unidades y si la generación es "
            "real, estimada o acumulada en otro periodo."
        )

    st.markdown("#### Valores nulos por variable")
    tabla_nulos = (
        df_filtrado.isna()
        .sum()
        .rename("Valores_Nulos")
        .to_frame()
        .assign(Porcentaje=lambda x: x["Valores_Nulos"] / len(df_filtrado) * 100)
        .reset_index(names="Variable")
    )
    st.dataframe(
        tabla_nulos,
        width="stretch",
        hide_index=True,
        column_config={"Porcentaje": st.column_config.NumberColumn(format="%.2f %%")},
    )

    st.markdown("#### Datos filtrados")
    st.dataframe(df_filtrado, width="stretch", hide_index=True)

with tab_reportes:
    st.subheader("Reportes listos para revisión o descarga")

    st.markdown("#### Reporte por tecnología")
    st.dataframe(
        reporte_tecnologia,
        width="stretch",
        hide_index=True,
        column_config={
            "MWh_Dia_por_MUSD": st.column_config.NumberColumn(format="%.2f"),
            "MUSD_por_MWh_Dia": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.markdown("#### Reporte por operador")
    st.dataframe(reporte_operador, width="stretch", hide_index=True)

    st.markdown("#### Reporte por estado")
    st.dataframe(reporte_estado, width="stretch", hide_index=True)

    st.markdown("#### Conclusión ejecutiva")
    st.write(
        f"Con los filtros actuales, {lider['Tecnologia']} tiene la mayor "
        f"productividad agregada ({lider['MWh_Dia_por_MUSD']:.2f} MWh/día por "
        "MUSD). Este resultado es descriptivo y debe contrastarse con el estado "
        "operativo, la vida útil, los costos de operación y la confiabilidad de "
        "las unidades antes de recomendar una inversión."
    )

    descarga1, descarga2, descarga3 = st.columns(3)
    descarga1.download_button(
        "Descargar reporte por tecnología",
        data=convertir_csv(reporte_tecnologia),
        file_name="reporte_tecnologia.csv",
        mime="text/csv",
    )
    descarga2.download_button(
        "Descargar reporte por operador",
        data=convertir_csv(reporte_operador),
        file_name="reporte_operador.csv",
        mime="text/csv",
    )
    descarga3.download_button(
        "Descargar datos filtrados",
        data=convertir_csv(df_filtrado),
        file_name="energia_renovable_filtrada.csv",
        mime="text/csv",
    )

