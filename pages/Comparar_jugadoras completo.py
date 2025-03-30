import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from utils import display_logo
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

display_logo(180)

# Configuración de la página
st.title("Comparación de Jugadoras")

# Aplicar estilos personalizados
st.markdown("""
<style>
    /* Fondo blanco para los scorecards */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Color rojo para las etiquetas de métricas */
    .stMetric label {
        color: #e63946 !important;
        font-size: 0.9rem;
    }
    
    /* Color azul oscuro y negrita para los valores */
    .stMetric [data-testid="stMetricValue"] {
        color: #1d3557 !important;
        font-weight: bold !important;
        font-size: 1.4rem;
    }
    
    /* Estilo para comparación */
    .comparison-container {
        display: flex;
        justify-content: space-between;
    }
    
    /* Estilo para títulos centrados */
    .centered-title {
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Estilos para el indicador de similitud */
    .similarity-container {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100px;
    }
    
    .similarity-value {
        font-size: 36px;
        font-weight: bold;
        color: #000000; /* Color negro para el texto */
    }
    
    /* Colores de fondo para el índice de similitud según rango */
    .similarity-high {
        background-color: #a8e6cf; /* Verde claro para >85% */
    }
    
    .similarity-medium {
        background-color: #fdfd96; /* Amarillo claro para 60-84.9% */
    }
    
    .similarity-low {
        background-color: #ffcc99; /* Naranja claro para <60% */
    }
    
    .similarity-label {
        font-size: 14px;
        color: #000000;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    /* Estilos para percentiles */
    .percentile-container {
        padding: 1px;
        margin: 5px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        # Cargar los datos de porteras
        df_keepers = pd.read_csv("data/data_gold/df_keepers_gold_1.csv")
        # Cargar los datos de jugadoras de campo
        df_players = pd.read_csv("data/data_gold/df_players_gold_1.csv")
        # Cargar datos de información adicional de jugadoras
        df_players_info = pd.read_csv("data/players_info/big/df_players_info_global.csv")
        # Cargar datos de clubes 
        df_teams_info = pd.read_csv("data/teams_info/big/df_teams_info_global.csv")
        # Cargar datos de fotos de ATM 
        df_atm_photos = pd.read_csv("data/players_info/atm_pics.csv")
        
        # Copiar la columna Pos a Posición Principal en df_keepers
        if 'Pos' in df_keepers.columns:
            if 'Posición Principal' not in df_keepers.columns:
                df_keepers['Posición Principal'] = df_keepers['Pos']
            else:
                # Si la columna ya existe, actualizar sus valores
                df_keepers['Posición Principal'] = df_keepers['Pos']
        
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
        
        # Devolver todos los dataframes
        return df_combined, df_players_info, df_teams_info, df_atm_photos
    
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}")
        st.info("Asegúrate de que los archivos existen en las rutas especificadas.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.write(f"Detalles del error: {str(e)}")
        return None, None, None, None

# Cargar los datos
df_combined, df_players_info, df_teams_info, df_atm_photos = cargar_datos()

# Definir métricas por posición y nivel (macro, meso, micro)
def get_position_metrics():
    # Métricas para cada posición
    position_metrics = {
        'GK': {
            'macro': ['MP', 'Starts', 'Min', 'Min%', 'GA', 'GA90', 'SoTA', 'Save%', 'CS%'],
            'meso': ['Save%_PK', 'PSxG', 'PSxG/SoT', 'PSxG-GA', 'Pass_Cmp_+40y%', 'Pass_AvgLen'],
            'micro': ['Stp%', '#OPA/90', 'AvgDist', 'Mn/Start', 'Mn/Sub', 'Mn/MP']
        },
        'DF': {
            'macro': ['MP', 'Starts', 'Min', 'Min%', 'Gls', 'CrdY', 'CrdR', 'Tkl+Int', 'Recov'],
            'meso': ['Tkl/90', 'Tkl%', 'Blocks', 'Int', 'touch_Def Pen', 'touch_Def 3rd', 'touch_Mid 3rd'],
            'micro': ['TO_Succ%', 'Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', '2CrdY', 'Off.1']
        },
        'MF': {
            'macro': ['MP', 'Starts', 'Min', 'Min%', 'Gls', 'Ast', 'G+A', 'SCA90', 'GCA90'],
            'meso': ['SoT/90', 'G/Sh', 'Dist', 'xA', 'KP', 'pass_1/3', 'crsPA', 'touch_Mid 3rd', 'touch_Att 3rd'],
            'micro': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', 'TO_Succ%', 'carries_TotDist', 
                     'carries_PrgDist', 'PrgR', 'Tkl/90', 'Tkl%', 'CrdY', 'CrdR']
        },
        'FW': {
            'macro': ['MP', 'Starts', 'Min', 'Min%', 'Gls', 'Ast', 'G+A', 'SoT/90', 'xG'],
            'meso': ['G/Sh', 'Dist', 'G-xG', 'SCA/90', 'GCA/90', 'touch_Att 3rd', 'touch_Att Pen'],
            'micro': ['TO_Succ%', 'carries_TotDist', 'carries_PrgDist', 'PrgR', 'Tkl/90', 'CrdY', 'CrdR', 'Off.1']
        }
    }
    return position_metrics

# Diccionario de nombres de métricas para mostrar
def get_metric_display_names():
    return {
        'MP': 'Partidos Jugados',
        'Starts': 'Partidos Titular',
        'Min': 'Minutos Jugados',
        'GA': 'Goles Encajados',
        'GA90': 'Goles Encajados (por 90)',
        'SoTA': 'Tiros a Puerta Recibidos',
        'Save%': '(%) Paradas',
        'CS%': '(%) Clean Sheets',
        'Save%_PK': '(%) Paradas en Penalty',
        'PSxG': 'xG tras de Tiro',
        'PSxG/SoT': 'xG tras Tiro por Tiro a Puerta',
        'PSxG-GA': 'xG tras Tiro menos Goles Encajado',
        'Pass_Cmp_+40y%': '(%) Pases exitosos +40yardas',
        'Pass_AvgLen': 'Longitud de Pases (Avg)',
        'Stp%': '(%) Centros al área bloqueados',
        '#OPA/90': 'Acciones fuera del área (por 90)',
        'AvgDist': 'Distancia de Acciones fuera del área (Avg)',
        'Gls': 'Goles',
        'Ast': 'Asistencias',
        'G+A': 'Goles + Asistencias',
        'G-PK': 'Goles (Sin Penalties)',
        'PK': 'Penalties Marcados',
        'PKatt': 'Penalties Intentados',
        'Sh': 'Tiros',
        'SoT': 'Tiros a Puerta',
        'G/Sh': 'Goles por Tiro',
        'G/SoT': 'Goles por Tiro a Puerta',
        'CrdY': 'Tarjetas Amarillas',
        'CrdR': 'Tarjetas Rojas',
        'Tkl': 'Entradas',
        'TklW': 'Entradas Ganadas',
        'Blocks': 'Bloqueos',
        'Int': 'Intercepciones',
        'Clr': 'Despejes',
        'Err': 'Errores',
        'Touches': 'Toques',
        'Succ%': '(%) Regates Exitosos',
        'Tkld%': '(%) Veces Regateado',
        'Att 3rd': 'Toques en Último Tercio',
        'Att Pen': 'Toques en Área Rival',
        'Live': 'Toques en Juego',
        'Prog Rec': 'Recepciones Progresivas',
        'Min%' : '(%) Minutos del Equipo jugados',
        'Mn/Start' : 'Minutos por partido de titular',
        'Mn/Sub' : 'Minutos por partido de suplente',
        'Mn/MP' : 'Minutos por partido jugado',
        'SoT/90' : 'Tiros a puerta (por 90)',
        'Dist' : 'Distancia de los tiros (Avg)',
        'xG' : 'Expected Goals scored',
        'G-xG' : 'Expected Goals Scored minus Goals Scored',
        'SCA90' : 'Acciones de tiro creadas (por 90)',
        'GCA90' : 'Acciones de gol creadas (por 90)',
        'touch_Def Pen' : 'Contactos al balón en Área propia', 
        'touch_Def 3rd' : 'Contactos al balón en 1/3 defensivo', 
        'touch_Mid 3rd' : 'Contactos al balón en 1/3 medio', 
        'touch_Att 3rd' : 'Contactos al balón en 1/3 ofensivo', 
        'touch_Att Pen' : 'Contactos al balón en Área rival', 
        'TO_Succ%' : '(%) Éxito de regates intentados',
        'Tkl/90' : 'Tackles (por90)',
        'Tkl%' : '(%) Éxito en tackles',
        'Blocks' : 'Tiros bloqueados',
        'Int' : 'Interceptaciones', 
        'Tkl+Int' : 'Tackles + Interceptaciones',
        'Recov' : 'Recuperaciones de Balón',
        '2CrdY' : '2ª Tarjeta Amarilla',
        'Off.1': 'Fueras de juego', 
        'carries_TotDist' : 'Distancia recorrida con el balón',
        'carries_PrgDist' : 'Distancia progresiva recorrida con el balón',
        'PrgR' : 'Recepciones de balón en progresión',
        'crsPA' : 'Pases de centro al área',
        'PPA' : 'Pases al área rival',
        'SCA/90': 'Acciones de tiro creadas (por 90)',
        'GCA/90': 'Acciones de gol creadas (por 90)',
        'xA': 'Expected Assists',
        'KP': 'Pases clave',
        'pass_1/3': 'Pases al último tercio',
        'TotDist': 'Distancia total de pases',
        'PrgDist': 'Distancia progresiva de pases'
    }

# Función para crear filtros de jugadora
def crear_filtros_jugadora(key_prefix, container):
    with container:
        # Variables para almacenar datos de los filtros
        all_ligas = []
        all_clubes = []
        all_posiciones = []
        filtered_players = []
        
        # Obtener ligas disponibles
        if 'League' in df_combined.columns:
            all_ligas = sorted(df_combined['League'].dropna().unique())
        else:
            all_ligas = []
        
        # Obtener clubes disponibles
        if 'Squad' in df_combined.columns:
            all_clubes = sorted(df_combined['Squad'].dropna().unique())
        
        # Obtener posiciones disponibles
        if 'Posición Principal' in df_combined.columns:
            all_posiciones = sorted(df_combined['Posición Principal'].dropna().unique())
        
        # Obtener nombres de jugadoras
        if 'Player' in df_combined.columns:
            all_players = sorted(df_combined['Player'].dropna().unique())
            filtered_players = all_players
        
        # Iniciar con filtros vacíos
        selected_liga = st.selectbox(
            "Liga:", 
            [""] + list(all_ligas), 
            index=0, 
            key=f"{key_prefix}_liga"
        )
        
        # Filtrar clubes según la liga seleccionada
        filtered_clubes = all_clubes
        if selected_liga:
            filtered_clubes = sorted(df_combined[df_combined['League'] == selected_liga]['Squad'].dropna().unique())
        
        selected_club = st.selectbox(
            "Club:", 
            [""] + list(filtered_clubes), 
            index=0, 
            key=f"{key_prefix}_club"
        )
        
        # Filtrar jugadoras según el club seleccionado
        if selected_club:
            filtered_players = df_combined[df_combined['Squad'] == selected_club]['Player'].dropna().unique()
        
        # Filtrar posiciones según el club seleccionado
        filtered_positions = all_posiciones
        if selected_club:
            filtered_positions = df_combined[df_combined['Squad'] == selected_club]['Posición Principal'].dropna().unique()
        
        selected_position = st.selectbox(
            "Posición:", 
            [""] + list(filtered_positions), 
            index=0, 
            key=f"{key_prefix}_pos"
        )
        
        # Filtrar jugadoras según la posición seleccionada
        if selected_position and filtered_players is not None:
            position_players = df_combined[df_combined['Posición Principal'] == selected_position]['Player'].dropna().unique()
            filtered_players = [player for player in filtered_players if player in position_players]
        
        # Selectbox final para jugadoras con todos los filtros aplicados
        selected_player = st.selectbox(
            "Jugadora:", 
            [""] + list(filtered_players), 
            index=0, 
            key=f"{key_prefix}_player"
        )
        
        return selected_player

# Función para mostrar los datos básicos de las jugadoras
def mostrar_datos_jugadora(player_name, container):
    with container:
        if not player_name:
            st.info("Selecciona una jugadora")
            return None
        
        # Filtrar datos de la jugadora
        df_view = df_combined[df_combined['Player'] == player_name]
        
        if df_view.empty:
            st.warning(f"No se encontraron datos para {player_name}")
            return None
        
        # Buscar información adicional de la jugadora
        jugadora_info = None
        try:
            if df_players_info is not None and 'Player' in df_players_info.columns:
                # Intenta encontrar la jugadora por nombre exacto
                jugadora_info = df_players_info[df_players_info['Player'] == player_name]
                
                # Si no encuentra por nombre exacto, intenta buscar si el nombre está contenido
                if jugadora_info.empty:
                    matching_indices = []
                    for idx, row in df_players_info.iterrows():
                        if (not pd.isna(row['Player']) and 
                            (player_name.lower() in row['Player'].lower() or 
                             row['Player'].lower() in player_name.lower())):
                            matching_indices.append(idx)
                            break  # Solo tomamos la primera coincidencia
                    
                    if matching_indices:
                        jugadora_info = df_players_info.loc[matching_indices]
        except Exception as e:
            st.error(f"Error al buscar información de la jugadora: {e}")
            jugadora_info = None
        
        # Mostrar foto y datos básicos
        st.markdown(f"<h2 style='text-align: center;'>{player_name}</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Mostrar foto si está disponible
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if not df_view.empty and df_view['Squad'].iloc[0] == "Atlético de Madrid":
                try:
                    jugadora_foto = df_atm_photos[df_atm_photos['Player'] == player_name]
                    if not jugadora_foto.empty:
                        photo_url_atm = jugadora_foto['url_photo'].iloc[0]
                        st.image(photo_url_atm, width=150)
                except Exception as e:
                    st.warning(f"No se pudo cargar la foto: {e}")
            elif (jugadora_info is not None and not jugadora_info.empty and 
                'Photo' in jugadora_info.columns and 
                len(jugadora_info) > 0 and not pd.isna(jugadora_info['Photo'].iloc[0])):
                photo_url = jugadora_info['Photo'].iloc[0]
                st.image(photo_url, width=150)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Buscar el logo del club
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            team_logo_url = None
            if df_teams_info is not None and 'Squad' in df_teams_info.columns and 'Shield URL' in df_teams_info.columns:
                club = df_view['Squad'].iloc[0] if not df_view.empty else ""
                club_team = df_teams_info[df_teams_info['Squad'] == club]
                if not club_team.empty and not pd.isna(club_team['Shield URL'].iloc[0]):
                    team_logo_url = club_team['Shield URL'].iloc[0]
                    st.image(team_logo_url, width=150)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Mostrar datos básicos
        datos_basicos = []
        
        # Datos del dataframe principal
        if not df_view.empty:
            if 'Posición Principal' in df_view.columns and not pd.isna(df_view['Posición Principal'].iloc[0]):
                datos_basicos.append(("Posición", df_view['Posición Principal'].iloc[0]))
            
            # Verificar que jugadora_info no esté vacío y tenga filas antes de acceder a iloc[0]
            if (jugadora_info is not None and not jugadora_info.empty and 
                'Birth_Date' in jugadora_info.columns and 
                len(jugadora_info) > 0 and not pd.isna(jugadora_info['Birth_Date'].iloc[0])):
                datos_basicos.append(("Fecha de Nacimiento", jugadora_info['Birth_Date'].iloc[0]))
                
            if 'Nation' in df_view.columns and not pd.isna(df_view['Nation'].iloc[0]):
                datos_basicos.append(("Nacionalidad", df_view['Nation'].iloc[0]))
            if 'Squad' in df_view.columns and not pd.isna(df_view['Squad'].iloc[0]):
                datos_basicos.append(("Club", df_view['Squad'].iloc[0]))
            if 'League' in df_view.columns and not pd.isna(df_view['League'].iloc[0]):
                datos_basicos.append(("Liga", df_view['League'].iloc[0]))
        
        # Mostrar los datos básicos en dos columnas
        col1, col2 = st.columns(2)
        for i, (nombre, valor) in enumerate(datos_basicos):
            if i % 2 == 0:
                with col1:
                    st.info(f"**{nombre}:** {valor}")
            else:
                with col2:
                    st.info(f"**{nombre}:** {valor}")
        
        return df_view

# Función para obtener métricas del jugador
def obtener_metricas_jugadora(df_view):
    if df_view is None or df_view.empty:
        return None, None, None
    
    # Obtener la posición de la jugadora
    player_position = df_view['Posición Principal'].iloc[0] if 'Posición Principal' in df_view.columns else ""
    
    # Obtener métricas por posición
    position_metrics = get_position_metrics()
    
    # Obtener métricas para cada nivel basado en la posición
    metrics_by_level = {
        'macro': position_metrics.get(player_position, {}).get('macro', []),
        'meso': position_metrics.get(player_position, {}).get('meso', []),
        'micro': position_metrics.get(player_position, {}).get('micro', [])
    }
    
    # Verificar que las métricas existan en el dataframe
    existing_metrics = {}
    for level, metrics in metrics_by_level.items():
        existing_metrics[level] = [metric for metric in metrics if metric in df_view.columns]
    
    # Obtener todas las métricas existentes juntas
    all_metrics = []
    for metrics in existing_metrics.values():
        all_metrics.extend(metrics)
    
    # Crear un diccionario con los valores de las métricas
    metrics_data = {}
    
    for metric in all_metrics:
        if metric in df_view.columns:
            metrics_data[metric] = df_view[metric].iloc[0]
    
    return metrics_data, existing_metrics, player_position

# Función para mostrar métricas en diferentes niveles
def mostrar_metricas_nivel(df_view, metrics_list, metric_names, nivel, container):
    with container:
        if not df_view is None and not metrics_list is None and len(metrics_list) > 0:
            # Mostrar 3 métricas por fila
            num_metrics = len(metrics_list)
            metrics_per_row = 3
            
            for i in range(0, num_metrics, metrics_per_row):
                cols = st.columns(metrics_per_row)
                
                for j, metric in enumerate(metrics_list[i:i+metrics_per_row]):
                    if j < len(cols) and metric in df_view.columns:
                        with cols[j]:
                            value = df_view[metric].iloc[0]
                            
                            # Formatear el valor según el tipo de métrica
                            if 'percentage' in metric or 'rate' in metric or 'completion' in metric or '%' in metric:
                                formatted_value = f"{value:.1f}%"
                            elif 'distance' in metric:
                                formatted_value = f"{value:.1f} km"
                            elif metric in ['xg', 'goals_conceded_per90', 'G/Sh', 'G/SoT']:
                                formatted_value = f"{value:.2f}"
                            else:
                                try:
                                    formatted_value = f"{value:.1f}"
                                except:
                                    formatted_value = str(value)
                            
                            # Mostrar la scorecard sin delta
                            st.metric(
                                label=metric_names.get(metric, metric),
                                value=formatted_value,
                                delta=None
                            )
        else:
            st.info(f"No hay métricas de nivel {nivel} disponibles para esta jugadora")

# Función para calcular percentiles de jugadoras
def calcular_percentiles(df_view, metrics_list):
    if df_view is None or df_view.empty or not metrics_list:
        return {}
    
    player_position = df_view['Posición Principal'].iloc[0] if 'Posición Principal' in df_view.columns else ""
    
    # Filtrar jugadoras de la misma posición
    df_position = df_combined[df_combined['Posición Principal'] == player_position]
    
    percentiles = {}
    
    for metric in metrics_list:
        if metric in df_view.columns and metric in df_position.columns:
            # Obtener el valor de la jugadora
            player_value = df_view[metric].iloc[0]
            
            # Calcular el percentil
            if not pd.isna(player_value):
                # Eliminar valores no numéricos y NaN
                metric_values = df_position[metric].dropna()
                
                if not metric_values.empty:
                    percentile = stats.percentileofscore(metric_values, player_value)
                    percentiles[metric] = percentile
    
    return percentiles

# Función para mostrar comparación de percentiles
def mostrar_percentiles(percentiles1, percentiles2, player1_name, player2_name, metric_names):
    if not percentiles1 or not percentiles2:
        st.info("No hay suficientes datos para calcular percentiles")
        return
    
    # Obtener métricas comunes
    common_metrics = [m for m in percentiles1.keys() if m in percentiles2]
    
    if not common_metrics:
        st.info("No hay métricas comunes para comparar percentiles")
        return
    
    # Mostrar gráfico de barras para comparar percentiles
    data = []
    for metric in common_metrics:
        data.append({
            'metric': metric_names.get(metric, metric),
            'Percentil ' + player1_name: percentiles1[metric],
            'Percentil ' + player2_name: percentiles2[metric]
        })
    
    # Convertir a DataFrame
    df_percentiles = pd.DataFrame(data)
    
    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=(12, len(common_metrics) * 0.5))
    
    # Ordenar por la diferencia entre percentiles
    df_percentiles['diff'] = abs(df_percentiles['Percentil ' + player1_name] - df_percentiles['Percentil ' + player2_name])
    df_percentiles = df_percentiles.sort_values('diff', ascending=False)
    
    # Crear gráfico
    bar_width = 0.35
    index = np.arange(len(df_percentiles))
    
    ax.barh(index, df_percentiles['Percentil ' + player1_name], bar_width, label=player1_name, color='#1f77b4')
    ax.barh(index + bar_width, df_percentiles['Percentil ' + player2_name], bar_width, label=player2_name, color='#ff7f0e')
    
    # Añadir etiquetas y leyenda
    ax.set_xlabel('Percentil')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(df_percentiles['metric'])
    ax.set_xlim(0, 100)
    ax.legend()
    
    # Añadir líneas de referencia
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=75, color='gray', linestyle='--', alpha=0.5)
    
    # Añadir valores sobre las barras
    for i, v in enumerate(df_percentiles['Percentil ' + player1_name]):
        ax.text(v + 1, i - 0.05, f"{v:.0f}%", color='#1f77b4', fontweight='bold')
    
    for i, v in enumerate(df_percentiles['Percentil ' + player2_name]):
        ax.text(v + 1, i + bar_width - 0.05, f"{v:.0f}%", color='#ff7f0e', fontweight='bold')
    
    plt.tight_layout()
    
    return fig

# Función para calcular la similitud entre dos jugadoras usando distancia euclidiana
def calcular_similitud(metrics1, metrics2, position_metrics, position):
    if not metrics1 or not metrics2 or not position_metrics or not position:
        return 0
    
    # Obtener métricas relevantes para la posición
    relevant_metrics = []
    for level_metrics in position_metrics.get(position, {}).values():
        relevant_metrics.extend(level_metrics)
    
    # Identificar métricas comunes
    common_metrics = [m for m in relevant_metrics if m in metrics1 and m in metrics2]
    
    if not common_metrics:
        return 0
    
    # Crear vectores para el cálculo de la distancia euclidiana
    vector1 = np.array([metrics1.get(m, 0) for m in common_metrics])
    vector2 = np.array([metrics2.get(m, 0) for m in common_metrics])
    
    # Normalizar cada métrica por su rango en el dataset para evitar que métricas 
    # con valores grandes dominen la distancia
    max_values = {}
    min_values = {}
    
    # Filtrar jugadoras de la misma posición
    df_position = df_combined[df_combined['Posición Principal'] == position]
    
    # Calcular min y max para normalización
    for metric in common_metrics:
        if metric in df_position.columns:
            values = df_position[metric].dropna()
            if not values.empty:
                max_values[metric] = values.max()
                min_values[metric] = values.min()
            else:
                max_values[metric] = max(metrics1.get(metric, 0), metrics2.get(metric, 0))
                min_values[metric] = min(metrics1.get(metric, 0), metrics2.get(metric, 0))
    
    # Normalizar vectores (escalar entre 0 y 1)
    normalized_vector1 = []
    normalized_vector2 = []
    
    for i, metric in enumerate(common_metrics):
        range_val = max_values[metric] - min_values[metric]
        if range_val > 0:
            normalized_vector1.append((vector1[i] - min_values[metric]) / range_val)
            normalized_vector2.append((vector2[i] - min_values[metric]) / range_val)
        else:
            normalized_vector1.append(0)
            normalized_vector2.append(0)
    
    # Calcular la distancia euclidiana normalizada
    normalized_vector1 = np.array(normalized_vector1)
    normalized_vector2 = np.array(normalized_vector2)
    
    euclidean_distance = np.sqrt(np.sum((normalized_vector1 - normalized_vector2) ** 2))
    
    # Normalizar la distancia a un rango de 0 a 1, donde 0 es la máxima distancia posible
    # (√n donde n es número de dimensiones) y 1 es distancia 0
    max_possible_distance = np.sqrt(len(common_metrics))
    
    # Convertir distancia a similitud (0 = diferentes, 100 = idénticas)
    similarity = (1 - (euclidean_distance / max_possible_distance)) * 100
    
    # Ajustar la escala para que sea más interpretable
    # Opcional: Podemos ajustar esta fórmula según lo que parezca más intuitivo
    # Ejemplo: un valor de similitud más bajo para considerar dos jugadoras "similares"
    adjusted_similarity = max(0, similarity)
    
    return adjusted_similarity

# Función para crear gráfico radar
def crear_grafico_radar(metrics1_data, metrics2_data, metrics_list, player1_name, player2_name, metric_names):
    if not metrics1_data or not metrics2_data or not metrics_list:
        return
    
    # Filtrar solo las métricas que existen en ambos jugadores
    common_metrics = [m for m in metrics_list if m in metrics1_data and m in metrics2_data]
    
    if not common_metrics:
        return
    
    # Obtener nombres descriptivos para las métricas
    labels = [metric_names.get(m, m) for m in common_metrics]
    
    # Extraer valores para cada jugadora
    values1 = [metrics1_data.get(m, 0) for m in common_metrics]
    values2 = [metrics2_data.get(m, 0) for m in common_metrics]
    
    # Normalizar valores para el gráfico radar (entre 0 y 1)
    normalized_values = []
    for i in range(len(common_metrics)):
        max_val = max(values1[i], values2[i]) * 1.1  # 10% más para el gráfico
        if max_val > 0:
            normalized_values.append((values1[i]/max_val, values2[i]/max_val))
        else:
            normalized_values.append((0, 0))
    
    # Preparar datos para el gráfico radar
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    # Cerrar el círculo
    angles += angles[:1]
    
    # Preparar valores normalizados para el gráfico
    values1_norm = [v[0] for v in normalized_values]
    values2_norm = [v[1] for v in normalized_values]
    # Cerrar los polígonos
    values1_norm += values1_norm[:1]
    values2_norm += values2_norm[:1]
    labels += labels[:1]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Dibujar líneas del radar
    ax.plot(angles, values1_norm, 'o-', linewidth=2, label=player1_name, color='#1f77b4')
    ax.plot(angles, values2_norm, 'o-', linewidth=2, label=player2_name, color='#ff7f0e')
    ax.fill(angles, values1_norm, alpha=0.25, color='#1f77b4')
    ax.fill(angles, values2_norm, alpha=0.25, color='#ff7f0e')
    
    # Añadir etiquetas
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=labels[:-1])
    
    # Ajustar límites y estilo
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    
    # Añadir leyenda
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Ajustar diseño
    plt.tight_layout()
    
    return fig

# Verificar si se cargaron los datos correctamente
if df_combined is not None and not df_combined.empty:
    # Crear filtros en la barra lateral
    st.sidebar.title("Selección de Jugadoras")
    
    # Sección para la Jugadora 1
    st.sidebar.markdown("<h3 class='centered-title'>Jugadora 1</h3>", unsafe_allow_html=True)
    player1 = crear_filtros_jugadora("player1", st.sidebar)
    
    # Separador
    st.sidebar.divider()
    
    # Sección para la Jugadora 2
    st.sidebar.markdown("<h3 class='centered-title'>Jugadora 2</h3>", unsafe_allow_html=True)
    player2 = crear_filtros_jugadora("player2", st.sidebar)
    
    # Mostrar información de las jugadoras seleccionadas
    col1, col2 = st.columns(2)
    
    # Mostrar datos de las jugadoras
    df_player1 = mostrar_datos_jugadora(player1, col1)
    df_player2 = mostrar_datos_jugadora(player2, col2)
    
    # Crear las tabs después de mostrar la información básica de las jugadoras
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Visión General", 
        "Métricas Macro", 
        "Métricas Meso", 
        "Métricas Micro",
        "Comparación por Percentiles",
        "Comparación gráfica por métricas"
    ])
    
    # Variables para almacenar métricas y posiciones
    metrics1_data = None
    metrics2_data = None
    existing_metrics1 = None
    existing_metrics2 = None
    player1_position = None
    player2_position = None
    metric_names = get_metric_display_names()
    position_metrics = get_position_metrics()
    
    # Obtener métricas si se seleccionaron jugadoras
    if df_player1 is not None and player1:
        metrics1_data, existing_metrics1, player1_position = obtener_metricas_jugadora(df_player1)
       
    if df_player2 is not None and player2:
        metrics2_data, existing_metrics2, player2_position = obtener_metricas_jugadora(df_player2)
    
    # TAB 1: Visión General
    with tab1:
        st.header("Visión General de la Comparativa")
        col1, col2, col3 = st.columns(3)
        with col2:
            if metrics1_data and metrics2_data and player1_position == player2_position:
                # Calcular similitud coseno
                similarity = calcular_similitud(metrics1_data, metrics2_data, position_metrics, player1_position)
                
                # Determinar la clase CSS según el valor de similitud
                similarity_class = ""
                if similarity > 75:  # Ajustamos los umbrales para la distancia euclidiana
                    similarity_class = "similarity-high"
                elif similarity >= 50:
                    similarity_class = "similarity-medium"
                else:
                    similarity_class = "similarity-low"
                
                # Mostrar porcentaje de similitud con la clase correspondiente
                st.markdown(f"""
                <div class='similarity-container {similarity_class}'>
                    <div class='similarity-label'>Índice de Similitud</div>
                    <div class='similarity-value'>{similarity:.1f}%</div>
                    <div style="font-size: 12px;">Basado en distancia euclidiana normalizada</div>
                </div>
                """, unsafe_allow_html=True)
        
        if metrics1_data and metrics2_data and player1_position == player2_position:
            # Crear gráfico radar para visualizar similitudes
            # Obtener todas las métricas para la posición
            all_position_metrics = []
            for metrics_list in position_metrics.get(player1_position, {}).values():
                all_position_metrics.extend(metrics_list)
            
            radar_fig = crear_grafico_radar(
                metrics1_data, 
                metrics2_data, 
                all_position_metrics, 
                player1, 
                player2, 
                metric_names
            )
            
            if radar_fig:
                st.pyplot(radar_fig)
                st.caption("Gráfico radar que muestra el perfil de rendimiento de ambas jugadoras")
                
        elif player1 and player2 and player1_position != player2_position:
            st.warning(f"Las jugadoras tienen posiciones diferentes ({player1_position} vs {player2_position}). No es posible realizar una comparación directa del perfil de juego.")
        else:
            st.info("Selecciona dos jugadoras para ver su comparativa")
    
    # TAB 2: Métricas Macro
    with tab2:
        st.header("Métricas Macro")
        st.markdown("Las métricas macro representan indicadores generales de rendimiento y participación")
        
        col1, col2 = st.columns(2)
        
        if df_player1 is not None and player1 and existing_metrics1:
            with col1:
                st.subheader(f"{player1}")
                mostrar_metricas_nivel(
                    df_player1, 
                    existing_metrics1.get('macro', []), 
                    metric_names, 
                    'macro', 
                    col1
                )
        
        if df_player2 is not None and player2 and existing_metrics2:
            with col2:
                st.subheader(f"{player2}")
                mostrar_metricas_nivel(
                    df_player2, 
                    existing_metrics2.get('macro', []), 
                    metric_names, 
                    'macro', 
                    col2
                )
    
    # TAB 3: Métricas Meso
    with tab3:
        st.header("Métricas Meso")
        st.markdown("Las métricas meso representan indicadores intermedios y específicos por área de juego")
        
        col1, col2 = st.columns(2)
        
        if df_player1 is not None and player1 and existing_metrics1:
            with col1:
                st.subheader(f"{player1}")
                mostrar_metricas_nivel(
                    df_player1, 
                    existing_metrics1.get('meso', []), 
                    metric_names, 
                    'meso', 
                    col1
                )
        
        if df_player2 is not None and player2 and existing_metrics2:
            with col2:
                st.subheader(f"{player2}")
                mostrar_metricas_nivel(
                    df_player2, 
                    existing_metrics2.get('meso', []), 
                    metric_names, 
                    'meso', 
                    col2
                )
    
    # TAB 4: Métricas Micro
    with tab4:
        st.header("Métricas Micro")
        st.markdown("Las métricas micro representan indicadores detallados y específicos de habilidades concretas")
        
        col1, col2 = st.columns(2)
        
        if df_player1 is not None and player1 and existing_metrics1:
            with col1:
                st.subheader(f"{player1}")
                mostrar_metricas_nivel(
                    df_player1, 
                    existing_metrics1.get('micro', []), 
                    metric_names, 
                    'micro', 
                    col1
                )
        
        if df_player2 is not None and player2 and existing_metrics2:
            with col2:
                st.subheader(f"{player2}")
                mostrar_metricas_nivel(
                    df_player2, 
                    existing_metrics2.get('micro', []), 
                    metric_names, 
                    'micro', 
                    col2
                )
    
    # TAB 5: Comparación por percentiles
    with tab5:
        st.header("Comparación por Percentiles")
        st.markdown("Esta vista muestra en qué percentil se encuentra cada jugadora en las diferentes métricas en comparación con otras jugadoras de su misma posición")
        
        if metrics1_data and metrics2_data and player1_position == player2_position:
            # Calcular todos los percentiles
            all_metrics = []
            for level_metrics in position_metrics.get(player1_position, {}).values():
                all_metrics.extend(level_metrics)
            
            percentiles1 = calcular_percentiles(df_player1, all_metrics)
            percentiles2 = calcular_percentiles(df_player2, all_metrics)
            
            # Mostrar gráfico de comparación de percentiles
            percentile_fig = mostrar_percentiles(percentiles1, percentiles2, player1, player2, metric_names)
            
            if percentile_fig:
                st.pyplot(percentile_fig)
                st.caption(f"Comparación de percentiles entre {player1} y {player2} respecto a otras jugadoras de su posición")
                
                # Explicación de los percentiles
                st.info("""
                **Interpretación de los percentiles:**
                - Un percentil de 90% significa que la jugadora supera al 90% de las otras jugadoras en esa métrica.
                - Un percentil de 50% indica un rendimiento medio en comparación con otras jugadoras.
                - Un percentil de 10% indica que el 90% de las otras jugadoras tienen mejores valores en esa métrica.
                """)
            else:
                st.info("No hay suficientes datos para calcular percentiles")
                
        elif player1 and player2 and player1_position != player2_position:
            st.warning("Las jugadoras tienen posiciones diferentes. Los percentiles solo son comparables entre jugadoras de la misma posición.")
        else:
            st.info("Selecciona dos jugadoras para ver la comparación por percentiles")
    
    # TAB 6: Comparación gráfica por métricas
    with tab6:
        st.header("Comparación gráfica por métricas")
        st.markdown("Esta vista permite comparar directamente las métricas individuales entre ambas jugadoras")
        
        # Crear gráficos comparativos si ambas jugadoras tienen datos
        if metrics1_data and metrics2_data:
            # Determinar si tienen la misma posición
            if player1_position == player2_position:
                # Si las jugadoras tienen la misma posición, podemos usar todas las métricas
                # Obtener todas las métricas para la posición
                all_position_metrics = []
                for metrics_list in position_metrics.get(player1_position, {}).values():
                    all_position_metrics.extend(metrics_list)
                
                # Filtrar para obtener solo métricas que existen en ambas jugadoras
                common_metrics = [m for m in all_position_metrics if m in metrics1_data and m in metrics2_data]
                
                # Organizar las métricas en filas de 3
                metrics_per_row = 3
                
                # Procesar las métricas en grupos de 3
                for i in range(0, len(common_metrics), metrics_per_row):
                    # Crear una fila con 3 columnas
                    cols = st.columns(metrics_per_row)
                    
                    # Procesar cada métrica de este grupo
                    for j, metric in enumerate(common_metrics[i:i+metrics_per_row]):
                        if j < len(cols):
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(4, 5))
                                
                                # Datos para el gráfico
                                labels = [player1, player2]
                                values = [metrics1_data[metric], metrics2_data[metric]]
                                
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
            else:
                st.warning(f"Las jugadoras tienen posiciones diferentes ({player1_position} vs {player2_position}). Para una comparación más significativa, selecciona jugadoras de la misma posición.")
        else:
            st.info("Selecciona dos jugadoras para ver la comparación gráfica de métricas")
else:
    st.error("No se pudieron cargar los datos. Verifica que los archivos existen en las rutas especificadas.")