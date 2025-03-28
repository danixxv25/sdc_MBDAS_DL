import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import display_logo

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
        return None, None, None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.write(f"Detalles del error: {str(e)}")
        return None, None, None

# Cargar los datos
df_combined, df_players_info, df_teams_info, df_atm_photos = cargar_datos()

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
            if df_view['Squad'].iloc[0] == "Atlético de Madrid":
                jugadora_foto = df_atm_photos[df_atm_photos['Player'] == player_name]
                photo_url_atm = jugadora_foto['url_photo'].iloc[0]
                st.image(photo_url_atm, use_container_width=True)

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
                    st.image(team_logo_url, width=100)
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

# Función para mostrar métricas de una jugadora
def mostrar_metricas_jugadora(df_view, container):
    with container:
        if df_view is None or df_view.empty:
            return None
        
        # Obtener la posición de la jugadora
        player_position = df_view['Posición Principal'].iloc[0] if 'Pos' in df_view.columns else ""
        
        # Definir métricas relevantes por posición
        position_metrics = {
            'GK': ['MP','Starts','Min','GA','GA90','SoTA','Save%','CS%','Save%_PK','PSxG','PSxG/SoT','PSxG-GA','Pass_Cmp_+40y%','Pass_AvgLen','Stp%','#OPA/90','AvgDist'],
            'DF': ['MP', 'Starts', 'Min', 'CrdY', 'CrdR', 'Tkl', 'TklW', 'Blocks', 'Int', 'Clr', 'Err', 'Touches', 'Succ%', 'Tkld%', 'Att 3rd', 'Att Pen'],
            'MF': ['MP', 'Starts', 'Min', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'Sh', 'SoT', 'Touches', 'Succ%', 'Att 3rd', 'Att Pen', 'Prog Rec'],
            'FW': ['MP', 'Starts', 'Min', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'Sh', 'SoT', 'G/Sh', 'G/SoT', 'Touches', 'Att Pen', 'Live']
        }
        
        # Mapeo de nombres técnicos a nombres descriptivos para el usuario
        metric_display_names = {
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
                    'Gls' : 'Goles marcados',
                    'Ast' : 'Asistencias',
                    'G+A' : 'Goles + Asistencias',
                    'SoT/90' : 'Tiros a puerta (por 90)',
                    'G/Sh' : 'Goles por cada tiro',
                    'Dist' : 'Distancia de los tiros (Avg)',
                    'xG' : 'Expected Goals scored',
                    'G-xG' : 'Expected Goals Scored minus Goals Scored',
                    'SCA90' : 'Acciones de tiro creadas (por 90)',
                    'GCA90' : 'Acciones de gol creadas (por 90)',
                    'touch_Def Pen' : 'Contactos al balón en Área propia', 
                    'touch_Def 3rd' : 'Contactos al balón en 1/3 defensivo' , 
                    'touch_Mid 3rd' : 'Contactos al balón en 1/3 medio' , 
                    'touch_Att 3rd' : 'Contactos al balón en 1/3 ofensivo', 
                    'touch_Att Pen' : 'Contactos al balón en Área rival', 
                    'TO_Succ%' : '(%) Éxito de regates intentados',
                    'Tkl/90' : 'Tackles (por90)',
                    'Tkl%' : '(%) Éxito en tackles',
                    'Blocks' : 'Tiros bloqueados',
                    'Int' : 'Interceptaciones', 
                    'Tkl+Int' : 'Tackles + Interceptaciones',
                    'Recov' : 'Recuperaciones de Balón',
                    'CrdY' : 'Tarjetas Amarillas', 
                    'CrdR' : 'Tarjetas Rojas',
                    '2CrdY' : '2ª Tarjeta Amarilla',
                    'Off.1': 'Fueras de juego', 
                    'carries_TotDist' : 'Distancia recorrida con el balón',
                    'carries_PrgDist' : 'Distancia progresiva recorrida con el balón',
                    'PrgR' : 'Recepciones de balón en progresión',
                    'CrsPA' : 'Pases de centro al área',
                    'PPA' : 'Pases al área rival'
        }
        
        # Obtener métricas relevantes para la posición
        relevant_metrics = position_metrics.get(player_position, [])
        
        # Verificar que las métricas existan en el dataframe
        existing_metrics = [metric for metric in relevant_metrics if metric in df_view.columns]
        
        # Crear un diccionario con los valores de las métricas para devolver
        metrics_data = {}
        
        if existing_metrics:
            # Mostrar 3 métricas por fila
            num_metrics = len(existing_metrics)
            metrics_per_row = 3
            
            for i in range(0, num_metrics, metrics_per_row):
                cols = st.columns(metrics_per_row)
                
                for j, metric in enumerate(existing_metrics[i:i+metrics_per_row]):
                    if j < len(cols):
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
                            
                            # Guardar el valor para comparaciones
                            metrics_data[metric] = value
                            
                            # Mostrar la scorecard sin delta
                            st.metric(
                                label=metric_display_names.get(metric, metric),
                                value=formatted_value,
                                delta=None
                            )
        
        return metrics_data, existing_metrics, player_position, metric_display_names

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

# Verificar si se cargaron los datos correctamente
if df_combined is not None and not df_combined.empty:
    # Crear dos columnas para los filtros
    col1, col2 = st.columns(2)
    
    # Título para cada sección de filtros
    col1.markdown("<h3 class='centered-title'>Jugadora 1</h3>", unsafe_allow_html=True)
    col2.markdown("<h3 class='centered-title'>Jugadora 2</h3>", unsafe_allow_html=True)
    
    # Crear filtros para cada jugadora
    player1 = crear_filtros_jugadora("player1", col1)
    player2 = crear_filtros_jugadora("player2", col2)
    
    # Separador
    st.divider()
    
    # Mostrar información de las jugadoras seleccionadas
    col1, col2 = st.columns(2)
    
    # Mostrar datos de las jugadoras
    df_player1 = mostrar_datos_jugadora(player1, col1)
    df_player2 = mostrar_datos_jugadora(player2, col2)
    
    # Separador
    st.divider()
    
    # Título para sección de métricas
    #st.markdown(f"<h1 style='text-align: center;'>Estadísticas Individuales</h1>", unsafe_allow_html=True)
    
    # Mostrar métricas de cada jugadora
    col1, col2 = st.columns(2)
    
    # Variables para almacenar métricas y posiciones
    metrics1_data = None
    metrics2_data = None
    metrics_list = None
    player_position = None
    metric_names = None
    
    # Mostrar métricas si se seleccionaron jugadoras
    if df_player1 is not None and player1:
        col1.subheader(f"Métricas de {player1}")
        metrics1_data, metrics_list1, position1, metric_names = mostrar_metricas_jugadora(df_player1, col1)
    
    if df_player2 is not None and player2:
        col2.subheader(f"Métricas de {player2}")
        metrics2_data, metrics_list2, position2, metric_names = mostrar_metricas_jugadora(df_player2, col2)
    
    # Separador
    st.divider()
    
    # Crear gráficos comparativos si ambas jugadoras tienen datos
    if metrics1_data and metrics2_data:
        # Usar solo métricas que existen para ambas jugadoras
        if position1 == position2:
            # Si las jugadoras tienen la misma posición, podemos usar todas las métricas
            metrics_list = [m for m in metrics_list1 if m in metrics_list2]
            player_position = position1
        else:
            # Si tienen posiciones diferentes, buscar métricas comunes
            metrics_list = [m for m in metrics_list1 if m in metrics_list2]
            st.warning(f"Las jugadoras tienen posiciones diferentes ({position1} vs {position2}). Mostrando solo métricas compatibles.")
        
        # Crear los gráficos comparativos
        crear_graficos_comparativos(metrics1_data, metrics2_data, metrics_list, player_position, metric_names, player1, player2)
    elif player1 and player2:
        st.info("Selecciona dos jugadoras para ver la comparación gráfica")
else:
    st.error("No se pudieron cargar los datos. Verifica que los archivos existen en las rutas especificadas.")
