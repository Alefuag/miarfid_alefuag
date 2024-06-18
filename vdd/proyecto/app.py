import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months
from plotter import plot_fwi, plot_lineplot, get_datos_provincia

# Importa las funciones de visualización
# from scripts.data_processing import load_data
# from scripts.visualizations import pandas_bar_chart, seaborn_scatter_plot

# Configuración de la página
st.set_page_config(
    page_title="FWI Visualization App",
    page_icon=":bar_chart:",
    layout="wide"
)

# Carga de datos
data = load_data('eimri_estadistica_basica_all.csv')

# Título y descripción
st.title("FWI (Forest Weather Index) Visualization")
st.markdown("## Análisis y visualización de datos")

# Sidebar para seleccionar el tipo de gráfico
st.sidebar.header("Opciones de visualización")


# year slider
year = st.sidebar.slider("Selecciona el año", min_value=data['year'].min(), max_value=data['year'].max(), value=data['year'].max(), step=1)

# mes selectbox
mes = st.sidebar.selectbox("Selecciona el mes", months[:-1])
# medida radio button
medida = st.sidebar.radio("Selecciona la medida", data['Estadisticos'].unique(), index=3)

# absolute values
st.sidebar.markdown("### Opciones adicionales")
absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)

# Mostrar el gráfico seleccionado
# if chart_type == "Pandas Bar Chart":
#     pandas_bar_chart(data)
# elif chart_type == "Seaborn Scatter Plot":
#     seaborn_scatter_plot(data)
st.write("Año seleccionado:", year)
st.write("Mes seleccionado:", mes)
st.write("Medida seleccionada:", medida)

fig_size = (1, 0.6)
scale = st.slider("Tamaño de la figura", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
scale *= 10
fig_size = (fig_size[0]*scale, fig_size[1]*scale)

# plot map
fig = plot_fwi(data, year, medida, mes, absolute=absolute, figsize=fig_size)

st.pyplot(fig, use_container_width=False)

# choose provinces
province = st.selectbox("Selecciona las provincias", data['Provincia'].unique())

prov_data = get_datos_provincia(data, year, mes, province)

# 3 columns
col1, col2, col3 = st.columns(3)
with col1:
    st.write("### Mínimo")
    st.write(prov_data['min'], '/ 5')
with col2:
    st.write("### Media")
    st.write(prov_data['mean'], '/ 5')
with col3:
    st.write("### Máximo")
    st.write(prov_data['max'], '/ 5')
    


# Footer
# why does not places the footer at the bottom of the page?

st.markdown("### Created by Alejandro Furió :sunglasses:", )
st.markdown('Github: [Alefuag](https://github.com/Alefuag/miarfid_alefuag/tree/main/vdd)')
