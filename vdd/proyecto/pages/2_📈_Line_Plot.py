import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months, download_file, get_fig_size
from plotter import plot_fwi, plot_lineplot


st.set_page_config(
    page_title="FWI Line Plot",
    page_icon="📈",
    layout="wide"
)

with st.spinner("Cargando datos..."):
    data = load_data('eimri_estadistica_basica_all.csv')

# Título y descripción
st.title("FWI Line Plot")

st.header("Gráficas de Líneas")

fig_size = get_fig_size(fig_size=(1, 0.6))


# Sidebar para seleccionar el tipo de gráfico
st.sidebar.header("Opciones de visualización")

provinces = st.sidebar.multiselect("Selecciona las provincias", data['Provincia'].unique())
if len(provinces) == 0:
    provinces.append('Valencia')    
# year slider
year = st.sidebar.slider("Selecciona el año", min_value=data['year'].min(), max_value=data['year'].max(), value=data['year'].max(), step=1)

# medida radio button
medida = st.sidebar.radio("Selecciona la medida", pd.Series(data['Estadisticos'].unique()).str.replace("_", " "), index=3).replace(" ", "_")

# absolute values
st.sidebar.markdown("### Opciones adicionales")

absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)

# plot map
fig, df = plot_lineplot(data, provinces, medida, year, absolute=absolute, figsize=fig_size)

st.pyplot(fig, use_container_width=False)

download_file(df, file_name=f"lineplot_data_{medida}_{year}.csv")


# Footer
st.markdown("---")
st.markdown("Código fuente en [GitHub](https://github.com/Alefuag/miarfid_alefuag/tree/main/vdd)")