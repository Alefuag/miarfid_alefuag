

import pandas as pd
import geopandas as gpd
import glob
from unidecode import unidecode
import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
from tqdm import tqdm

import streamlit as st

current_dir = Path(__file__).parent
data_folder = current_dir.parent / 'datos'

months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre', 'Anual']

@st.cache_data
def load_data(file_name):
    """
    Load data from a CSV file and return a DataFrame.
    
    Args:
        file_name (str): The name of the CSV file to load.

    """
    merged_file = str(data_folder / file_name)
    gpd_file = data_folder / 'lineas_limite.zip!SHP_ETRS89/provincias'

    provincias = gpd.read_file(gpd_file)
    fwi_df = pd.read_csv(merged_file)

    fwi_prov = fwi_df['Provincia'].apply(clean_province_name).replace('Araba', 'Alava')
    geo_prov = provincias['NAMEUNIT'].apply(clean_province_name)

    common_provinces = set(fwi_prov.unique()).intersection(set(geo_prov.unique()))

    fwi_df_clean = fwi_df[fwi_df['Provincia'].apply(clean_province_name).replace('Araba', 'Alava').isin(common_provinces)]
    fwi_df_clean.loc[:, 'Provincia'] = fwi_df_clean['Provincia'].apply(clean_province_name).replace('Araba', 'Alava')
    geo_prov_clean = provincias[provincias['NAMEUNIT'].apply(clean_province_name).isin(common_provinces)]
    geo_prov_clean.loc[:, 'NAMEUNIT'] = geo_prov_clean['NAMEUNIT'].apply(clean_province_name)

    geo_prov_useful_cols = ['NAMEUNIT', 'geometry']

    # merge the two dataframes
    merged_df = geo_prov_clean[geo_prov_useful_cols].merge(fwi_df_clean, left_on='NAMEUNIT', right_on='Provincia', how='right')

    processed_df = merged_df.copy()

    for m in months:
        processed_df[m] = merged_df[m].str.replace(',', '.').astype('float64')

    interval_df = calculate_interval(processed_df)
    processed_df = processed_df.merge(interval_df, on=['year', 'Provincia'])

    return processed_df


def calculate_interval(df):
    # group by year and province, then substrac the max from the min
    return df.groupby(['year', 'Provincia']).agg({'Anual': lambda x: x.max() - x.min()}).rename(columns={'Anual': 'Intervalo'}).reset_index()


def clean_province_name(province):
    clean_name = unidecode(province)
    if '/' in clean_name:
        return clean_name.split('/')[1]
    return clean_name

