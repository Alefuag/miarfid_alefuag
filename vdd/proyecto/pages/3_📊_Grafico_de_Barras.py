
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months, download_file, get_fig_size
from plotter import plot_fwi, plot_lineplot, plot_bar_chart, get_datos_provincia

# Configuración de la página
st.set_page_config(
    page_title="FWI Bar Plot",
    page_icon=":bar_chart:",
    layout="wide"
)

# Carga de datos
with st.spinner("Cargando datos..."):
    data = load_data('eimri_estadistica_basica_all.csv')


# Título y descripción
st.title("Forest Weather Index Visualization Tool")

st.header("Gráficas de Barras")


fig_size = get_fig_size(fig_size=(1, 0.5))

# Sidebar para seleccionar el tipo de gráfico
st.sidebar.header("Opciones de visualización")

province = st.sidebar.selectbox("Selecciona las provincias", data['Provincia'].unique())

# medida radio button
medida = st.sidebar.radio("Selecciona la medida", pd.Series(data['Estadisticos'].unique()).str.replace("_", " "), index=3).replace(" ", "_")

absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)

# plot map
fig, df = plot_bar_chart(data, province, medida, figsize=fig_size, absolute=absolute)

st.pyplot(fig, use_container_width=False)

download_file(df, file_name=f"barplot_data_{medida}_{province}.csv")
