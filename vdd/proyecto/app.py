import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months
from plotter import plot_fwi, plot_lineplot, plot_pie_chart, get_datos_provincia


# Configuración de la página
st.set_page_config(
    page_title="FWI Visualization App",
    page_icon=":bar_chart:",
    layout="wide"
)

# Carga de datos
with st.spinner("Cargando datos..."):
    data = load_data('eimri_estadistica_basica_all.csv')

def home_page():
    st.title("FWI (Forest Weather Index) Visualization")
    st.header("Visualización de Datos")

    st.write("Este es un proyecto de visualización de datos de FWI (Forest Weather Index) en España.")
    st.write("FWI es un índice que mide el riesgo de incendio forestal basado en las condiciones meteorológicas." + 
             "Este índice varía de 0 a 5, siendo 5 el valor más alto y por lo tanto el mayor riesgo de incendio.")

    st.write("El conjunto de datos contiene información sobre el FWI en diferentes provincias de España." + 
             "Puedes visualizar los datos en un mapa y en gráficas de barras.")
    
    st.write("Para navegar por la aplicación, utiliza el menú de la izquierda.")

def map_page():

    # Título y descripción
    st.title("FWI (Forest Weather Index) Visualization")

    st.header("Mapas")

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

    # st.write("Año seleccionado:", year)
    # st.write("Mes seleccionado:", mes)
    # st.write("Medida seleccionada:", medida)

    fig_size = (1, 0.6)
    scale = st.sidebar.slider("Tamaño de la figura", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    scale *= 10
    fig_size = (fig_size[0]*scale, fig_size[1]*scale)


    # plot map
    fig = plot_fwi(data, year, medida, mes, absolute=absolute, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)

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


def bar_chart_page():
    # Título y descripción
    st.title("FWI (Forest Weather Index) Visualization")

    st.header("Gráficas de Barras")

    fig_size = (1, 0.6)
    scale = st.sidebar.slider("Tamaño de la figura", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    scale *= 10
    fig_size = (fig_size[0]*scale, fig_size[1]*scale)

    st.sidebar.markdown("---")

    # Sidebar para seleccionar el tipo de gráfico
    st.sidebar.header("Opciones de visualización")
    # year slider
    year = st.sidebar.slider("Selecciona el año", min_value=data['year'].min(), max_value=data['year'].max(), value=data['year'].max(), step=1)

    # medida radio button
    medida = st.sidebar.radio("Selecciona la medida", data['Estadisticos'].unique(), index=3)

    # absolute values
    st.sidebar.markdown("### Opciones adicionales")
    absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)


    provinces = st.sidebar.multiselect("Selecciona las provincias", data['Provincia'].unique())
    if len(provinces) == 0:
        provinces.append('Valencia')

    # plot map
    fig = plot_lineplot(data, provinces, medida, year, absolute=absolute, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)


def pie_chart_page():
    # Título y descripción
    st.title("FWI (Forest Weather Index) Visualization")

    st.header("Gráficas de Barras")

    fig_size = (1, 0.5)
    scale = st.sidebar.slider("Tamaño de la figura", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    scale *= 10
    fig_size = (fig_size[0]*scale, fig_size[1]*scale)

    st.sidebar.markdown("---")

    # Sidebar para seleccionar el tipo de gráfico
    st.sidebar.header("Opciones de visualización")
    # medida radio button
    medida = st.sidebar.radio("Selecciona la medida", data['Estadisticos'].unique(), index=3)


    provinces = st.sidebar.selectbox("Selecciona las provincias", data['Provincia'].unique())
    if len(provinces) == 0:
        provinces.append('Valencia')

    # plot map
    fig = plot_pie_chart(data, provinces, medida, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)
    

PAGES = {
    "Home": home_page,
    "Mapa": map_page,
    "Gráficas de Barras": bar_chart_page,
    "Gráficas de Pastel": pie_chart_page
}


st.sidebar.title("Navegación")
selection = st.sidebar.radio("Ir a", list(PAGES.keys()))
st.sidebar.markdown("---")

page = PAGES[selection]
page()



# Footer
st.markdown("---")
st.markdown("Hecho con :heart: por Alejandro")
st.markdown("Puedes encontrar el código fuente en [GitHub](https://github.com/Alefuag/miarfid_alefuag/tree/main/vdd)")