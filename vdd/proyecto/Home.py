import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loader import load_data, months, download_file
from plotter import plot_fwi, plot_lineplot, plot_bar_chart, get_datos_provincia


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
    st.title("FWI Visualization Dashboard")
    # st.header("Visualización de Datos")

    st.markdown(
        """
        Bienvenido al **FWI Dashboard**, una plataforma interactiva diseñada para proporcionar una visualización clara y precisa del Fire Weather Index (FWI).
        
        El `FWI` es una herramienta para predecir y gestionar el riesgo de incendios forestales, basada en diversos factores meteorológicos.
        
        Este índice varía de `1` a `5`, siendo 5 el valor más alto y por lo tanto el mayor riesgo de incendio.
        """
    )

    st.markdown(
        """
        ## Funcionalidades del Dashboard
        - Visualización interactiva de los componentes del `FWI`.
        - Análisis temporal y espacial del riesgo de incendios.
        - Descarga de datos para análisis adicional.
        """
        )

    # st.write("")

    st.write("El conjunto de datos contiene información sobre el `FWI` en diferentes provincias de España. " + 
             "Puedes visualizar los datos en un mapa y en gráficas de barras.")
    
    st.write(
        """
        ## Comienza a explorar
        Utiliza el menú de la izquierda para navegar por las diferentes secciones del dashboard y comenzar tu análisis del `FWI`.
        """
    )

    st.page_link("Home.py", label="Inicio", icon="🏠")
    st.write("Donde podrás encontrar información sobre el proyecto y las funcionalidades del dashboard.")
    st.page_link("pages/1_🗺_Mapa.py", label="Mapa", icon="🗺")
    st.write("Visualiza el `FWI` en un mapa interactivo y explora los datos por provincia.")
    st.page_link("pages/2_📈_Grafico_de_Lineas.py", label="Gráfico de Lineas", icon="📈")
    st.write("Visualiza la evolución del `FWI` a lo largo del tiempo en diferentes provincias.")
    st.page_link("pages/3_📊_Grafico_de_Barras.py", label="Gráfico de Barras", icon="📊")
    st.write("Visualiza el `FWI` promedio por año en diferentes provincias.")




def map_page():

    # Título y descripción
    st.title("Forest Weather Index Visualization Tool")

    st.header("Mapas")

    fig_size = (1, 0.6)
    scale = st.sidebar.slider("Tamaño de la figura", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
    scale *= 10
    fig_size = (fig_size[0]*scale, fig_size[1]*scale)
    st.sidebar.markdown("---")

    # Sidebar para seleccionar el tipo de gráfico
    st.sidebar.header("Opciones de visualización")
    # year slider
    year = st.sidebar.slider("Selecciona el año", min_value=data['year'].min(), max_value=data['year'].max(), value=data['year'].max(), step=1)

    # mes selectbox
    mes = st.sidebar.selectbox("Selecciona el mes", months[:-1])
    # medida radio button
    medida = st.sidebar.radio("Selecciona la medida", pd.Series(data['Estadisticos'].unique()).str.replace("_", " "), index=3).replace(" ", "_")


    # absolute values
    st.sidebar.markdown("### Opciones adicionales")
    absolute = st.sidebar.toggle("Mostrar escala absoluta", value=False)

    # plot map
    fig, df = plot_fwi(data, year, medida, mes, absolute=absolute, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)

    download_file(df)
    
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


def line_chart_page():
    # Título y descripción
    st.title("Forest Weather Index Visualization Tool")

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
    fig, df = plot_lineplot(data, provinces, medida, year, absolute=absolute, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)

    download_file(df)


def bar_chart_page():
    # Título y descripción
    st.title("Forest Weather Index Visualization Tool")

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
    fig, df = plot_pie_chart(data, provinces, medida, figsize=fig_size)

    st.pyplot(fig, use_container_width=False)

    download_file(df)



PAGES = {
    "Home": home_page,
    "FWI por territorio - Mapa": map_page,
    "FWI Mensual - Gráfico de Líneas": line_chart_page,
    "FWI Anual - Gráficas de Barras": bar_chart_page
}


# st.sidebar.title("Navegación")
# selection = st.sidebar.radio("Ir a", list(PAGES.keys()))
# st.sidebar.markdown("---")

# page = PAGES[selection]
# page()

home_page()

# Footer
st.markdown("---")
st.markdown("Código fuente en [GitHub](https://github.com/Alefuag/miarfid_alefuag/tree/main/vdd)")