import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importa las funciones de visualización
# from scripts.data_processing import load_data
# from scripts.visualizations import pandas_bar_chart, seaborn_scatter_plot

# Configuración de la página
st.set_page_config(
    page_title="Data Visualization App",
    page_icon=":bar_chart:",
    layout="wide"
)

# Carga de datos
# data = load_data()

# Título y descripción
st.title("FWI (Forest Weather Index) Visualization")
st.markdown("## Análisis y visualización de datos")

# Sidebar para seleccionar el tipo de gráfico
st.sidebar.header("Opciones de visualización")
chart_type = st.sidebar.selectbox("Selecciona el tipo de gráfica", ["Pandas Bar Chart", "Seaborn Scatter Plot"])

# Mostrar el gráfico seleccionado
# if chart_type == "Pandas Bar Chart":
#     pandas_bar_chart(data)
# elif chart_type == "Seaborn Scatter Plot":
#     seaborn_scatter_plot(data)
st.write("Gráfico seleccionado:", chart_type)


# Footer
# why does not places the footer at the bottom of the page?

st.markdown("### Created by Alejandro Furió :sunglasses:", )
st.markdown('Github: [Alefuag](https://github.com/Alefuag/miarfid_alefuag/tree/main/vdd)')
