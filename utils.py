import streamlit as st

import pandas as pd
import numpy as np

# Librerias de visualización
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar
import seaborn as sns
import matplotlib.cm as cm

# Librerias de ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances

import os
import base64
from io import BytesIO

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def display_logo(width=180):
    #col1, col2, col3 = st.columns([1, 2, 1])
    #with col2:
    st.image("media/logos/Atleti.png", width=width)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def buscar_jugadoras(df, player=None, position=None, team=None, compet=None, minyear=None, maxyear=None, MP=None):
    # Creamos una máscara inicial que selecciona todas las filas
    mask = pd.Series(True, index=df.index)
    
    # Aplicamos los filtros solo si se proporcionan
    if player:
        mask &= df['Player'].str.contains(player, case=False)
    if position:
        mask &= df['Pos'] == position
    if team:
        mask &= df['Squad'] == team
    if compet:
        mask &= df['League'] == compet
    if minyear:
        mask &= df['Born'] >= minyear
    if maxyear:
        mask &= df['Born'] <= maxyear
    if MP:
        mask &= df['MP'] >= MP
    
    resultado = df[mask]
    resultado = resultado.drop(columns=['Pos'])

    #columnas = resultado.columns.tolist()
    #columnas[0], columnas[1] = columnas[1], columnas[0]
    #resultado = resultado[columnas]

    print("Resultado de la búsqueda:")
    print("Numero de jugadoras:", df[mask].shape[0])
    print(resultado.to_string())  # Imprime el DataFrame en formato de tabla
    
    return resultado

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def find_similar_players(df, player_name):
    # Implementa la lógica de búsqueda de jugadoras similares aquí
    # Este es un placeholder, deberás implementar tu propio algoritmo
    return df.sample(5)  # Retorna 5 jugadoras aleatorias como ejemplo

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def generar_radar(player1, player2, df, metrics):
    # Implementa la lógica de búsqueda de jugadoras similares aquí
    # Este es un placeholder, deberás implementar tu propio algoritmo
    return df
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def calcular_comparativa(df, metric, player_value, player_position, competition):
    """
    Calcula las medias de comparación y determina si el valor del jugador es mejor o peor.
    
    Args:
        df: DataFrame con todas las jugadoras
        metric: Nombre de la métrica a comparar
        player_value: Valor de la métrica para la jugadora seleccionada
        player_position: Posición de la jugadora
        competition: Competición de la jugadora
    
    Returns:
        dict: Diccionario con las medias y el estado de comparación
    """
    # Métricas donde un valor más bajo es mejor (invertir comparación)
    lower_is_better = ['GA', 'GA90', 'Err', 'CrdY', 'CrdR', '2CrdY', 'Off.1']
    
   
    # Calcular media de jugadoras de la misma posición en todas las ligas
    position_df = df[df['Posición Principal'] == player_position]
    liga_df = position_df[position_df['League'] == competition]
    position_metric_values = position_df[metric].dropna()
    liga_metric_values = liga_df[metric].dropna()

    
    # Obtener las medias
    liga_mean = liga_metric_values.mean() if not liga_metric_values.empty else 0
    position_mean = position_metric_values.mean() if not position_metric_values.empty else 0
    
    # Determinar si el valor es mejor que las medias
    if metric in lower_is_better:
        better_than_liga = player_value < liga_mean
        better_than_position = player_value < position_mean
    else:
        better_than_liga = player_value > liga_mean
        better_than_position = player_value > position_mean
    
    # Determinar el estado de comparación
    if better_than_liga and better_than_position:
        comparison_state = "better"
    elif better_than_liga or better_than_position:
        comparison_state = "mixed"
    else:
        comparison_state = "worse"
    
    return {
        'liga_mean': liga_mean,
        'position_mean': position_mean,
        'comparison_state': comparison_state}

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def cargar_datos_atm():
    try:
        # Cargar los datos de porteras
        df_keepers = pd.read_csv("data/data_gold/df_keepers_gold_1.csv")
        # Cargar los datos de jugadoras de campo
        df_players = pd.read_csv("data/data_gold/df_players_gold_1.csv")
        
        # Imprimir información sobre las columnas (para depuración)
        st.write("Columnas en df_keepers:", df_keepers.columns.tolist())
        st.write("Columnas en df_players:", df_players.columns.tolist())
        
        # Obtener todas las columnas únicas entre ambos dataframes
        all_columns = list(set(df_keepers.columns) | set(df_players.columns))
        
        # Agregar columnas faltantes a cada dataframe
        for col in all_columns:
            if col not in df_keepers.columns:
                # Si es columna numérica en df_players, inicializar con 0
                if col in df_players.columns and pd.api.types.is_numeric_dtype(df_players[col]):
                    df_keepers[col] = 0
                else:
                    df_keepers[col] = np.nan
            
            if col not in df_players.columns:
                # Si es columna numérica en df_keepers, inicializar con 0
                if col in df_keepers.columns and pd.api.types.is_numeric_dtype(df_keepers[col]):
                    df_players[col] = 0
                else:
                    df_players[col] = np.nan
        
        # Unir los dataframes
        df_combined = pd.concat([df_keepers, df_players], ignore_index=True)
        
        # Filtrar solo las jugadoras del Atlético de Madrid
        # Asegurar que la columna 'club' existe
        if 'club' in df_combined.columns:
            df_atletico = df_combined[df_combined['club'].str.contains('atlético', case=False, na=False)]
            
            # Verificar si tenemos resultados
            if len(df_atletico) == 0:
                st.warning("No se encontraron jugadoras del Atlético de Madrid. Verificando nombres de clubes disponibles...")
                st.write("Clubes en el dataframe:", df_combined['club'].unique())
            
            return df_atletico
        else:
            st.error("La columna 'club' no existe en el dataframe combinado.")
            return None
    
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}")
        st.info("Asegúrate de que los archivos existen en las rutas especificadas.")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.write(f"Detalles del error: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

def encontrar_jugadoras_similares(nombre_jugadora, df, n_similares=10):
    """
    Encuentra las jugadoras más similares a una jugadora específica usando PCA y K-means.
    
    Args:
        nombre_jugadora (str): Nombre de la jugadora de referencia
        df (DataFrame): DataFrame con los datos de todas las jugadoras
        n_similares (int): Número de jugadoras similares a devolver
        
    Returns:
        DataFrame: DataFrame con las jugadoras más similares
    """
    # Verificar que la jugadora existe en el DataFrame
    if nombre_jugadora not in df['Player'].values:
        print(f"La jugadora {nombre_jugadora} no se encuentra en el dataset.")
        return None
    
    # Obtener la posición de la jugadora
    jugadora_info = df[df['Player'] == nombre_jugadora].iloc[0]
    player_position = jugadora_info['Posición Principal']
    
    print(f"Analizando jugadora: {nombre_jugadora}")
    print(f"Posición: {player_position}")
    
    # Definir métricas relevantes según la posición
    position_metrics = {
        'GK': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 
               'GA', 'GA90', 'SoTA', 'Save%', 'CS%', 'Save%_PK', 'PSxG', 'PSxG/SoT', 'PSxG-GA', 
               'Pass_Cmp_+40y%', 'Pass_AvgLen', 'Stp%', '#OPA/90', 'AvgDist'],
               
        'DF': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 
               'Gls', 'Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', 'touch_Def Pen', 
               'touch_Def 3rd', 'touch_Mid 3rd', 'TO_Succ%', 'CrsPA', 'Tkl/90', 'Tkl%', 'Blocks', 
               'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1'],
               
        'MF': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 
               'Gls', 'Ast', 'G+A', 'SoT/90', 'G/Sh', 'Dist', 'SCA90', 'GCA90', 'Cmp%_short', 
               'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', 'xA', 'KP', 'pass_1/3', 'PPA', 
               'CrsPA', 'touch_Mid 3rd', 'touch_Att 3rd', 'touch_Att Pen', 'TO_Succ%', 
               'carries_TotDist', 'carries_PrgDist', 'PrgR', 'Tkl/90', 'Tkl%', 'Blocks', 
               'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1'],
               
        'FW': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 
               'Gls', 'Ast', 'G+A', 'SoT/90', 'G/Sh', 'Dist', 'xG', 'G-xG', 'SCA90', 'GCA90', 
               'xA', 'KP', 'pass_1/3', 'PPA', 'CrsPA', 'touch_Mid 3rd', 'touch_Att 3rd', 
               'touch_Att Pen', 'TO_Succ%', 'carries_TotDist', 'carries_PrgDist', 'PrgR', 
               'Tkl/90', 'Tkl%', 'Blocks', 'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1']
    }
    
    relevant_metrics = position_metrics.get(player_position, [])
    
    # Paso 1: Filtrar jugadoras por posición y seleccionar métricas relevantes
    df_pos = df[df['Posición Principal'] == player_position].copy()
    df_pos = df_pos[relevant_metrics].copy()
    
    # Paso 2: Preparación de datos
    # Establecer el nombre del jugador como índice
    df_prep = df_pos.copy()
    df_prep.index = df_prep['Player']
    
    # Eliminar columnas no numéricas o no relevantes para el análisis
    non_numeric_cols = ['Player', 'Squad', 'Born', 'MP']
    df_prep = df_prep.drop(columns=[col for col in non_numeric_cols if col in df_prep.columns])
    
    # Manejo de valores NaN (reemplazarlos con la media)
    df_prep = df_prep.fillna(df_prep.mean())
    
    # Escalado de datos
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_prep), 
        columns=df_prep.columns,
        index=df_prep.index
    )
    
    # Paso 3: PCA
    # Determinar el número óptimo de componentes (autovalores > 1)
    S = np.cov(df_scaled.T)
    autovalores, _ = np.linalg.eigh(S)
    n_components = sum(sorted(autovalores) > autovalores.mean())
    n_components = max(min(n_components, 8), 2)  # Entre 2 y 8 componentes
    
    print(f"Utilizando {n_components} componentes principales")
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(df_scaled)
    
    # Crear DataFrame con los scores de PCA
    df_pca = pd.DataFrame(
        pca_scores,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df_scaled.index
    )
    
    # Visualización de la varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = explained_variance.cumsum()
    print(f"Varianza explicada acumulada: {cum_explained_variance[-1]:.2%}")
    
    # Paso 4: Clustering personalizado para la jugadora objetivo
    # Primero, obtenemos el vector de características PCA de la jugadora objetivo
    target_player_pca = df_pca.loc[nombre_jugadora]
    
    # Calculamos la distancia euclidiana desde cada jugadora a la jugadora objetivo
    distances = pairwise_distances(
        df_pca,
        target_player_pca.values.reshape(1, -1),
        metric='euclidean'
    )
    
    # Creamos un DataFrame con las distancias
    distance_df = pd.DataFrame({
        'Player': df_pca.index,
        'Distance': distances.flatten()
    })
    
    # Ordenamos por distancia (las más similares primero, excluyendo la jugadora objetivo)
    similar_players = distance_df[distance_df['Player'] != nombre_jugadora].sort_values('Distance')
    
    # Limitamos a n_similares jugadoras
    similar_players = similar_players.head(n_similares)
    
    # Paso 5: Visualización
    # Creamos un DataFrame para visualización que incluya la jugadora objetivo y las similares
    vis_players = pd.concat([
        pd.DataFrame({'Player': [nombre_jugadora], 'Group': ['Target']}),
        pd.DataFrame({'Player': similar_players['Player'], 'Group': ['Similar'] * len(similar_players)})
    ])
    
    # Añadimos las coordenadas PCA
    vis_df = pd.merge(vis_players, df_pca.reset_index(), on='Player')
    
    # Visualización en 2D (primeras 2 componentes principales)
    plt.figure(figsize=(12, 8))
    
    # Primero dibujamos todas las jugadoras en gris claro
    plt.scatter(df_pca['PC1'], df_pca['PC2'], color='lightgray', alpha=0.5, label='Otras jugadoras')
    
    # Dibujamos las jugadoras similares
    similar_indices = vis_df[vis_df['Group'] == 'Similar'].index
    plt.scatter(
        vis_df.loc[similar_indices, 'PC1'],
        vis_df.loc[similar_indices, 'PC2'],
        color='blue',
        alpha=0.7,
        label='Jugadoras similares'
    )
    
    # Dibujamos la jugadora objetivo destacada
    target_index = vis_df[vis_df['Group'] == 'Target'].index[0]
    plt.scatter(
        vis_df.loc[target_index, 'PC1'],
        vis_df.loc[target_index, 'PC2'],
        color='red',
        s=100,
        label=nombre_jugadora
    )
    
    # Añadimos etiquetas a los puntos
    for i, row in vis_df.iterrows():
        plt.annotate(
            row['Player'],
            (row['PC1'], row['PC2']),
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title(f'Jugadoras más similares a {nombre_jugadora} (PCA)')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Devolver las jugadoras similares con información adicional
    result_df = pd.merge(
        similar_players,
        df[['Player', 'Squad', 'Born', 'Min', 'Posición Principal']],
        on='Player'
    )
    
    # Ordenar por distancia (más similares primero)
    result_df = result_df.sort_values('Distance')
    
    print("\nJugadoras más similares:")
    for i, row in result_df.iterrows():
        print(f"{row['Player']} ({row['Squad']}) - Distancia: {row['Distance']:.4f}")
    
    return result_df, plt.gcf()

# Ejemplo de uso:
# similar_players_df, fig = encontrar_jugadoras_similares("Alexia Putellas", df)
# fig.savefig("jugadoras_similares.png")
# similar_players_df.head(10)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

# Función para crear gráficos comparativos
def crear_graficos_comparativos(metrics1, metrics2, metrics_list, position, metric_names, player1_name, player2_name):
    if not metrics1 or not metrics2 or not metrics_list:
        st.warning("No hay suficientes datos para crear gráficos comparativos")
        return
    
    st.subheader("Comparación de Métricas")
    
    # Organizar las métricas en filas de 3
    metrics_per_row = 3
    
    # Procesar las métricas en grupos de 3
    for i in range(0, len(metrics_list), metrics_per_row):
        # Crear una fila con 3 columnas
        cols = st.columns(metrics_per_row)
        
        # Procesar cada métrica de este grupo
        for j, metric in enumerate(metrics_list[i:i+metrics_per_row]):
            if metric in metrics1 and metric in metrics2 and j < len(cols):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 5))
                    
                    # Datos para el gráfico
                    labels = [player1_name, player2_name]
                    values = [metrics1[metric], metrics2[metric]]
                    
                    # Definir colores
                    colors = ['#1f77b4', '#ff7f0e']
                    
                    # Crear el gráfico de barras
                    bars = ax.bar(labels, values, color=colors, width=0.6)
                    
                    # Añadir etiquetas y título
                    ax.set_ylabel(metric)
                    ax.set_title(f"{metric_names.get(metric, metric)}", fontsize=10)
                    
                    # Añadir valores sobre las barras
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
                    
                    # Ajustar los ejes y mostrar el gráfico
                    plt.tight_layout()
                    st.pyplot(fig)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->
