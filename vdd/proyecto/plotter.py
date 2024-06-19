from matplotlib import pyplot as plt
import geopandas as gpd
import seaborn as sns

from loader import months


def plot_fwi(province_df, year, medida, mes, absolute=False, figsize=(12, 12)):
    year_mask = province_df['year'] == year
    medida_mask = province_df['Estadisticos'] == medida

    filtered_df = province_df[year_mask & medida_mask]

    lower_bound = 1 if absolute else filtered_df[mes].min()
    upper_bound = 5 if absolute else filtered_df[mes].max()


    ax = filtered_df.plot(
        column=mes,
        legend=True,
        # figsize=(12, 12),
        cmap='coolwarm',
        edgecolor='black',
        linewidth=0.3,
        vmin=lower_bound, # cmap range min
        vmax=upper_bound, # cmap range max
        missing_kwds={'color': 'lightgrey'},
        figsize=figsize
        )

    ax.set_title(f'FWI {medida} {mes} {year}')
    # remove axis
    ax.axis('off')

    plt.ylabel('FWI')

    return ax.get_figure(), filtered_df


def plot_lineplot(province_df, provincias, medida, year, absolute=False, figsize=(12, 12)):
    year_mask = province_df['year'] == year
    medida_mask = province_df['Estadisticos'] == medida
    
    fig, ax = plt.subplots(figsize=figsize)

    for provincia in provincias:
        provincia_mask = province_df['Provincia'] == provincia
        filtered_df = province_df[provincia_mask & medida_mask & year_mask]
        value_list = [filtered_df[mes].values for mes in months[:-1]]
        # unpack the list of lists
        plt.plot(months[:-1], value_list, label=provincia, marker='o')
    
    if absolute:
        plt.ylim(0.9, 5)
    
    ax.set_title(f'{medida} {year}')

    plt.title(f'FWI {medida} {year}')
    plt.ylabel('FWI')
    plt.xlabel('Mes')
    # rotate x labels
    plt.xticks(rotation=45)
    plt.legend()

    return fig, filtered_df

def get_datos_provincia(province_df, year, mes, provincia):
    year_mask = province_df['year'] == year
    # medida_mask = province_df['Estadisticos'] == medida
    provincia_mask = province_df['Provincia'] == provincia

    filtered_df = province_df[year_mask & provincia_mask]
    filtered_df
    res = {
        'min' : filtered_df[filtered_df['Estadisticos'] == 'Minimo'][mes].iloc[0],
        'mean' : filtered_df[filtered_df['Estadisticos'] == 'Media'][mes].iloc[0],
        'max' : filtered_df[filtered_df['Estadisticos'] == 'Maximo'][mes].iloc[0],
    }
    res = {k: str(v) for (k, v) in res.items()}
    return res


def plot_bar_chart(province_df, provincia, medida, figsize=(12, 12)):
    medida_mask = province_df['Estadisticos'] == medida
    provincia_mask = province_df['Provincia'] == provincia

    fig, ax = plt.subplots(figsize=figsize)

    filtered_df = province_df[provincia_mask & medida_mask]

    # ax = filtered_df.plot(
    #     kind='pie',
    #     y='Anual',
    #     # ax=ax,
    #     labels=filtered_df['year'],
    #     autopct='%1.1f%%',
    #     startangle=90,
    #     legend=True,
    #     fontsize=11
    # )

    # plt.pie(filtered_df['Anual'], labels=filtered_df['year'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})

    # plt.bar(filtered_df['year'], filtered_df['Anual'], color='skyblue')

    sns.barplot(x='year', y='Anual', hue='year', data=filtered_df, palette='flare', legend=False)
    plt.title(f'{medida} en {provincia}. Años {filtered_df["year"].min()}-{filtered_df["year"].max()}')
    plt.ylabel('FWI Anual')

    # add legend
    # plt.legend(title='Años', loc='upper right', bbox_to_anchor=(1.2, 1))

    ax.set_title(f'{medida} en {provincia}. Años {filtered_df["year"].min()}-{filtered_df["year"].max()}')

    return fig, filtered_df