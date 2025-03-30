import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import display_logo


# Aplicar estilos personalizados para los scorecards
st.markdown("""
<style>
    /* Fondo blanco para los scorecards */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Color rojo para las etiquetas de métricas */
    .stMetric label {
        color: #e63946 !important;
        font-size: 0.9rem;
    }
    
    /* Color azul oscuro y negrita para los valores */
    .stMetric [data-testid="stMetricValue"] {
        font-weight: bold !important;
        font-size: 1.4rem;
    }
    
    /* Valor mejor que ambas medias - Verde */
    .metric-better [data-testid="stMetricValue"] {
        color: #2e8b57 !important;
    }
    
    /* Valor mejor que una media pero peor que otra - Amarillo */
    .metric-mixed [data-testid="stMetricValue"] {
        color: #ffa500 !important;
    }
    
    /* Valor peor que ambas medias - Rojo */
    .metric-worse [data-testid="stMetricValue"] {
        color: #dc3545 !important;
    }
    
    /* Estilo para las medias de comparación */
    .metric-comparison {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 5px;
    }
            
    /* Tamaño fijo para los scorecards */
    .stMetric {
        height: 165px;
        width: 100%;
        min-width: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Asegurar que el contenedor de valor tenga altura fija */
    .stMetric [data-testid="stMetricValue"] {
        margin: 8px 0;
        height: 30px;
        display: flex;
        align-items: center;
    }

    /* Asegurar que la sección de comparativas tenga altura fija */
    .metric-comparison {
        height: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Función para calcular medias y comparar valores
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

#display_logo(100)
#st.title("Atlético de Madrid Femenino")

@st.cache_data
def cargar_datos():
    try:
        # Cargar los datos de porteras
        df_keepers = pd.read_csv("data/data_gold/df_keepers_gold_1.csv")
        # Cargar los datos de jugadoras de campo
        df_players = pd.read_csv("data/data_gold/df_players_gold_1.csv")
        # Cargar datos de inrormación adicional de jugadoras
        df_players_info = pd.read_csv("data/players_info/df_230_players_info.csv")
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
        
        # Filtrar solo las jugadoras del Atlético de Madrid
        # Asegurar que la columna 'club' existe
        if 'Squad' in df_combined.columns:
            df_atletico = df_combined[df_combined['Squad'].str.contains('atlético', case=False, na=False)]
            
            # Verificar si tenemos resultados
            if len(df_atletico) == 0:
                st.warning("No se encontraron jugadoras del Atlético de Madrid. Verificando nombres de clubes disponibles...")
                st.write("Clubes en el dataframe:", df_combined['Squad'].unique())
            
            # Devolver tanto el dataframe del equipo como el de información adicional
            return df_atletico, df_players_info, df_teams_info, df_combined, df_atm_photos
        else:
            st.error("La columna 'Squad' no existe en el dataframe combinado.")
            return None, None
    
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}")
        st.info("Asegúrate de que los archivos existen en las rutas especificadas.")
        return None, None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.write(f"Detalles del error: {str(e)}")
        return None, None
    
# Cargar los datos
df_atletico, df_players_info, df_teams_info, df_combined, df_atm_photos = cargar_datos()

# Verificar si se cargaron los datos correctamente
if df_atletico is not None and not df_atletico.empty:
    # Añadir selector de jugadoras
    if 'Player' in df_atletico.columns:
        # Ordenar nombres alfabéticamente para el selector
        sorted_names = sorted(df_atletico['Player'].unique())
        
        # Añadir "" como primera opción
        selected_player = st.sidebar.selectbox(
            "Selecciona una jugadora:",
            [""] + list(sorted_names)
        )
        
        # Filtrar por jugadora seleccionada
        if selected_player != "":
            df_view = df_atletico[df_atletico['Player'] == selected_player]
            
            # Buscar información adicional de la jugadora
            jugadora_info = None
            if df_players_info is not None and 'Player' in df_players_info.columns:
                # Intenta encontrar la jugadora por nombre exacto
                jugadora_info = df_players_info[df_players_info['Player'] == selected_player]
                
                # Si no encuentra por nombre exacto, intenta buscar si el nombre está contenido
                if jugadora_info.empty:
                    for idx, row in df_players_info.iterrows():
                        if selected_player.lower() in row['Player'].lower() or row['Player'].lower() in selected_player.lower():
                            jugadora_info = df_players_info.iloc[[idx]]
                            break
            
            # Mostrar foto y datos básicos
            st.markdown(f"<h1 style='text-align: center;'>{selected_player}</h1>", unsafe_allow_html=True)
            st.divider()
            col1, col2, col3 = st.columns([3, 2, 3])

            with col1:
                        # Centrar foto horizontalmente
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        
                        # Mostrar foto si está disponible
                        if df_atm_photos is not None and not df_atm_photos.empty and 'url_photo' in df_atm_photos.columns:
                            jugadora_foto = df_atm_photos[df_atm_photos['Player'] == selected_player]
                            photo_url = jugadora_foto['url_photo'].iloc[0]
                            st.image(photo_url, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                # Centrar logo horizontalmente
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                
                # Buscar el logo del club
                team_logo_url = None
                if df_teams_info is not None and 'Squad' in df_teams_info.columns and 'Shield URL' in df_teams_info.columns:
                    club = df_view['Squad'].iloc[0]
                    club_team = df_teams_info[df_teams_info['Squad'] == club]
                    if not club_team.empty and not pd.isna(club_team['Shield URL'].iloc[0]):
                        team_logo_url = club_team['Shield URL'].iloc[0]
                        st.image(team_logo_url, width=150)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Mostrar datos básicos de la jugadora (a la izquierda)
            col_datos, col_stats = st.columns([1, 2])
            
            with col3:
                
                # Combinar datos de ambos dataframes
                datos_basicos = []
                
                # Datos del dataframe principal
                if not df_view.empty:
                    if 'Posición Principal' in df_view.columns and not pd.isna(df_view['Posición Principal'].iloc[0]):
                        datos_basicos.append(("Posición Principal", df_view['Posición Principal'].iloc[0]))
                    if 'Posición Secundaria' in df_view.columns and not pd.isna(df_view['Posición Secundaria'].iloc[0]):
                        datos_basicos.append(("Posición Secundaria", df_view['Posición Secundaria'].iloc[0]))
                    if 'Birth_Date' in jugadora_info.columns and not pd.isna(jugadora_info['Birth_Date'].iloc[0]):
                        datos_basicos.append(("Fecha de Nacimiento", jugadora_info['Birth_Date'].iloc[0]))
                    if 'Nation' in df_view.columns and not pd.isna(df_view['Nation'].iloc[0]):
                        datos_basicos.append(("Nacionalidad", df_view['Nation'].iloc[0]))
                    if 'Height' in jugadora_info.columns and not pd.isna(jugadora_info['Height'].iloc[0]):
                        datos_basicos.append(("Altura", f"{jugadora_info['Height'].iloc[0]}"))
                    if 'Weight' in jugadora_info.columns and not pd.isna(jugadora_info['Weight'].iloc[0]):
                        datos_basicos.append(("Peso", f"{jugadora_info['Weight'].iloc[0]} kg"))
                               
                # Mostrar los datos básicos
                for nombre, valor in datos_basicos:
                    st.info(f"**{nombre}:** {valor}")
            
st.divider()

col1, col2, col3 = st.columns([3, 6, 3])

# Comprobar si el dataframe está vacío
if df_atletico.empty:
    st.warning("No se pudo cargar el dataframe.")
else:
    # Suponiendo que existe una columna que indica la posición
    # Ajusta este nombre según la columna real en tu dataframe
    position_column = 'Posición Principal'
    if selected_player:
        st.markdown(f"<h1 style='text-align: center;'>Estadísticas de {selected_player}</h1>", unsafe_allow_html=True)


    
    # Buscar la posición de la jugadora seleccionada en df_atletico
    if 'Player' in df_atletico.columns and 'Posición Principal' in df_atletico.columns:
        player_info = df_atletico[df_atletico['Player'] == selected_player]
        
        if not player_info.empty:
            player_position = player_info['Posición Principal'].iloc[0]
            player_position2 = player_info['Posición Secundaria'].iloc[0]
            #st.subheader(f"Posición: {player_position}")
            
            # Filtrar dataframe por la posición seleccionada
            position_df = df_atletico[df_atletico[position_column] == player_position]
            
            # Definir métricas relevantes por posición
            position_metrics = {
                    'GK': ['MP','Starts','Min','GA','GA90','SoTA','Save%','CS%','Save%_PK','PSxG','PSxG/SoT','PSxG-GA','Pass_Cmp_+40y%','Pass_AvgLen','Stp%','#OPA/90','AvgDist'],
                    'DF': ['MP',
                        'Starts',
                        'Min',
                        'Min%',
                        'Mn/Start',
                        'Mn/Sub',
                        'Mn/MP', 
                        'Gls',
                        'Cmp%_short',
                        'Cmp%_med',
                        'Cmp%_long',
                        'TotDist',
                        'PrgDist',
                        'touch_Def Pen',
                        'touch_Def 3rd', 
                        'touch_Mid 3rd',
                        'TO_Succ%',
                        'CrsPA',
                        'Tkl/90',
                        'Tkl%',
                        'Blocks',
                        'Int', 
                        'Tkl+Int',
                        'Recov',
                        'CrdY', 
                        'CrdR',
                        '2CrdY',
                        'Off.1'
                    ],
                    'MF': ['MP',
                        'Starts',
                        'Min',
                        'Min%',
                        'Mn/Start',
                        'Mn/Sub',
                        'Mn/MP', 
                        'Gls', 'Ast',
                        'G+A',
                        'SoT/90',
                        'G/Sh',
                        'Dist', 
                        'SCA90',
                        'GCA90',
                        'Cmp%_short',
                        'Cmp%_med',
                        'Cmp%_long',
                        'TotDist',
                        'PrgDist',
                        'xA',
                        'KP',
                        'pass_1/3',
                        'PPA',
                        'CrsPA',
                        'touch_Mid 3rd', 
                        'touch_Att 3rd', 
                        'touch_Att Pen',
                        'TO_Succ%',
                        'carries_TotDist',
                        'carries_PrgDist',
                        'PrgR', 
                        'Tkl/90',
                        'Tkl%',
                        'Blocks',
                        'Int', 
                        'Tkl+Int',
                        'Recov',
                        'CrdY', 
                        'CrdR',
                        '2CrdY',
                        'Off.1'
                    ],
                    'FW': ['MP',
                        'Starts',
                        'Min',
                        'Min%',
                        'Mn/Start',
                        'Mn/Sub',
                        'Mn/MP',
                        'Gls',
                        'Ast',
                        'G+A',
                        'SoT/90',
                        'G/Sh',
                        'Dist',
                        'xG',
                        'G-xG',
                        'SCA90',
                        'GCA90',
                        'xA',
                        'KP',
                        'pass_1/3',
                        'PPA',
                        'CrsPA',
                        'touch_Mid 3rd', 
                        'touch_Att 3rd', 
                        'touch_Att Pen', 
                        'TO_Succ%',
                        'carries_TotDist',
                        'carries_PrgDist',
                        'PrgR', 
                        'Tkl/90',
                        'Tkl%',
                        'Blocks',
                        'Int', 
                        'Tkl+Int',
                        'Recov',
                        'CrdY', 
                        'CrdR',
                        '2CrdY',
                        'Off.1'
                    ]
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
                'SoT/90' : 'Tiros a puerta (por 90)',
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
                'Tkl+Int' : 'Tackles + Interceptaciones',
                'Recov' : 'Recuperaciones de Balón',
                'CrdY' : 'Tarjetas Amarillas', 
                'CrdR' : 'Tarjetas Rojas',
                '2CrdY' : '2ª Tarjeta Amarilla',
                'Off.1': 'Fueras de juego', 
                'carries_TotDist' : 'Distancia recorrida con el balón',
                'carries_PrgDist' : 'Distancia progresiva recorrida con el balón',
                'PrgR' : 'Recepciones de balón en progresión',
                'PPA' : 'Pases al área rival',
                'Cmp%_short' : 'Precisión en pases cortos',
                'Cmp%_med' : 'Precisión en pases medios',
                'Cmp%_long' : 'Precisión en pases largos',
                'TotDist' : 'Distancia total de pases',
                'PrgDist' : 'Distancia progresiva de pases',
                'xA' : 'Expected Assists',
                'KP' : 'Pases clave',
                'pass_1/3' : 'Pases al último tercio',
                'CrsPA' : 'Pases de centro al área'
                }


            # Filtrar dataframe para obtener las estadísticas de la jugadora seleccionada
            player_stats = df_atletico[df_atletico['Player'] == selected_player]
            
            if not player_stats.empty:
                # Obtener métricas relevantes para la posición de la jugadora
                relevant_metrics = position_metrics.get(player_position, [])
                
                # Verificar que las métricas existan en el dataframe
                existing_metrics = [metric for metric in relevant_metrics if metric in player_stats.columns]
                
                if existing_metrics:
                    # Obtener la competición de la jugadora seleccionada
                    player_competition = player_stats['League'].iloc[0] if 'League' in player_stats.columns else None
                    
                    # Calcular el número de columnas necesarias (máximo 5 columnas)
                    num_metrics = len(existing_metrics)
                    num_columns = min(5, num_metrics)
                    
                    # Crear las columnas para las scorecards
                    cols = st.columns(num_columns)
                    
                    # Llenar las columnas con las métricas
                    for i, metric in enumerate(existing_metrics[:num_columns]):
                        with cols[i]:
                            # Obtener el valor de la métrica para esta jugadora
                            value = player_stats[metric].iloc[0]
                            
                            # Calcular comparativas
                            comparativa = calcular_comparativa(
                                df_combined, 
                                metric, 
                                value, 
                                player_position, 
                                player_competition
                            )
                            
                            # Formatear el valor según el tipo de métrica
                            if 'percentage' in metric or 'rate' in metric or 'completion' in metric or '%' in metric:
                                formatted_value = f"{value:.1f}%"
                                formatted_liga = f"{comparativa['liga_mean']:.1f}%"
                                formatted_position = f"{comparativa['position_mean']:.1f}%"
                            elif 'distance' in metric:
                                formatted_value = f"{value:.1f} km"
                                formatted_liga = f"{comparativa['liga_mean']:.1f} km"
                                formatted_position = f"{comparativa['position_mean']:.1f} km"
                            elif metric in ['xg', 'goals_conceded_per90', 'G/Sh', 'G/SoT', 'PSxG/SoT', 'PSxG-GA']:
                                formatted_value = f"{value:.2f}"
                                formatted_liga = f"{comparativa['liga_mean']:.2f}"
                                formatted_position = f"{comparativa['position_mean']:.2f}"
                            else:
                                try:
                                    formatted_value = f"{value:.1f}"
                                    formatted_liga = f"{comparativa['liga_mean']:.1f}"
                                    formatted_position = f"{comparativa['position_mean']:.1f}"
                                except:
                                    formatted_value = str(value)
                                    formatted_liga = str(comparativa['liga_mean'])
                                    formatted_position = str(comparativa['position_mean'])
                            
                            # Crear un div contenedor con la clase según la comparación
                            comparison_class = f"metric-{comparativa['comparison_state']}"
                            
                            # Mostrar la métrica con HTML personalizado
                            st.markdown(
                                f"""
                                <div class="stMetric {comparison_class}">
                                    <label>{metric_display_names.get(metric, metric)}</label>
                                    <div data-testid="stMetricValue">{formatted_value}</div>
                                    <div class="metric-comparison">
                                        Media Liga: {formatted_liga}<br>
                                        Media Top 5 Ligas: {formatted_position}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Si hay más de 5 métricas, crear filas adicionales
                    if num_metrics > 5:
                        # Calcular cuántas filas adicionales se necesitan
                        remaining_metrics = num_metrics - 5
                        num_additional_rows = (remaining_metrics + 4) // 5  # División con techo para obtener el número de filas
                        
                        # Crear cada fila adicional
                        for row in range(num_additional_rows):
                            st.write("")  # Espacio entre filas
                            start_idx = 5 + (row * 5)  # Índice inicial para esta fila
                            end_idx = min(start_idx + 5, num_metrics)  # Índice final para esta fila
                            
                            # Crear columnas para esta fila
                            cols_row = st.columns(min(5, end_idx - start_idx))
                            
                            # Llenar las columnas con métricas
                            for i, metric in enumerate(existing_metrics[start_idx:end_idx]):
                                with cols_row[i]:
                                    # Obtener el valor de la métrica para esta jugadora
                                    value = player_stats[metric].iloc[0]
                                    
                                    # Calcular comparativas
                                    comparativa = calcular_comparativa(
                                        df_combined, 
                                        metric, 
                                        value, 
                                        player_position, 
                                        player_competition
                                    )
                                    
                                    # Formatear el valor según el tipo de métrica
                                    if 'percentage' in metric or 'rate' in metric or 'completion' in metric or '%' in metric:
                                        formatted_value = f"{value:.1f}%"
                                        formatted_liga = f"{comparativa['liga_mean']:.1f}%"
                                        formatted_position = f"{comparativa['position_mean']:.1f}%"
                                    elif 'distance' in metric:
                                        formatted_value = f"{value:.1f} km"
                                        formatted_liga = f"{comparativa['liga_mean']:.1f} km"
                                        formatted_position = f"{comparativa['position_mean']:.1f} km"
                                    elif metric in ['xg', 'goals_conceded_per90', 'G/Sh', 'G/SoT', 'PSxG/SoT', 'PSxG-GA']:
                                        formatted_value = f"{value:.2f}"
                                        formatted_liga = f"{comparativa['liga_mean']:.2f}"
                                        formatted_position = f"{comparativa['position_mean']:.2f}"
                                    else:
                                        try:
                                            formatted_value = f"{value:.1f}"
                                            formatted_liga = f"{comparativa['liga_mean']:.1f}"
                                            formatted_position = f"{comparativa['position_mean']:.1f}"
                                        except:
                                            formatted_value = str(value)
                                            formatted_liga = str(comparativa['liga_mean'])
                                            formatted_position = str(comparativa['position_mean'])
                                    
                                    # Crear un div contenedor con la clase según la comparación
                                    comparison_class = f"metric-{comparativa['comparison_state']}"
                                    
                                    # Mostrar la métrica con HTML personalizado
                                    st.markdown(
                                        f"""
                                        <div class="stMetric {comparison_class}">
                                            <label>{metric_display_names.get(metric, metric)}</label>
                                            <div data-testid="stMetricValue">{formatted_value}</div>
                                            <div class="metric-comparison">
                                                Media Liga: {formatted_liga}<br>
                                                Media Top 5 Ligas: {formatted_position}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                else:
                    st.warning(f"No se encontraron métricas relevantes para la posición {player_position}.")
            else:
                st.warning(f"No se encontraron estadísticas para {selected_player} en el dataframe de jugadoras.")
        else:
            # Interfaz cuando no hay jugadora seleccionada
            st.info("Elige una jugadora en la barra lateral")
            
            # Mostrar resumen de datos disponibles
            col1, col2, col3 = st.columns(3)
            
            with col2:
                display_logo(360)
            
            #with col2:
                #st.subheader("Instrucciones")
                #st.write("1. Selecciona una jugadora en el menú desplegable en la barra lateral")
                #st.write("2. Puedes filtrar por liga, club, posición o año de nacimiento")
                #st.write("3. Los filtros se actualizan dinámicamente según tus selecciones")
                #st.write("4. Selecciona una jugadora para ver sus estadísticas detalladas")


