import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months, download_file, get_fig_size
from plotter import plot_fwi, plot_lineplot, get_datos_provincia


# Configuración de la página
st.set_page_config(
    page_title="FWI Spain Map",
    page_icon=":bar_chart:",
    layout="wide"
)

# Carga de datos
with st.spinner("Cargando datos..."):
    data = load_data('eimri_estadistica_basica_all.csv')


# Título y descripción
st.title("Forest Weather Index Visualization Tool")

st.header("Mapas")

fig_size = get_fig_size(fig_size=(1, 0.6))

# Sidebar para seleccionar el tipo de gráfico
st.sidebar.header("Opciones de visualización")
# mes selectbox
mes = st.sidebar.selectbox("Selecciona el mes", months[:-1])
# year slider
year = st.sidebar.slider("Selecciona el año", min_value=data['year'].min(), max_value=data['year'].max(), value=data['year'].max(), step=1)
# medida radio button
medida = st.sidebar.radio("Selecciona la medida", pd.Series(data['Estadisticos'].unique()).str.replace("_", " "), index=3).replace(" ", "_")



# absolute values
st.sidebar.markdown("### Opciones adicionales")
absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)

# plot map
fig, df = plot_fwi(data, year, medida, mes, absolute=absolute, figsize=fig_size)

st.pyplot(fig, use_container_width=False)

download_file(df, file_name=f"map_data_{medida}_{mes}_{year}.csv")

# choose provinces
province = st.selectbox("Selecciona la provincia", data['Provincia'].unique())

prov_data = get_datos_provincia(data, year, mes, province)

# 3 columns
col1, col2, col3 = st.columns(3)
with col1:
    st.write("### FWI Mínimo")
    st.write(prov_data['min'], '/ 5')
with col2:
    st.write("### FWI Media")
    st.write(prov_data['mean'], '/ 5')
with col3:
    st.write("### FWI Máximo")
    st.write(prov_data['max'], '/ 5')