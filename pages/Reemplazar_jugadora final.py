import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import time

from utils import display_logo

display_logo(100)

# Configuración de la página y tema
#st.set_page_config(
#    page_title="Análisis de Similitud de Jugadoras",
#    page_icon="⚽",
#    layout="wide",
#    initial_sidebar_state="expanded"
#)

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

# Título de la aplicación
st.title("Análisis de Similitud de Jugadoras")
st.markdown("Esta aplicación te permite encontrar jugadoras similares a una jugadora seleccionada utilizando PCA y K-Means.")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        # Cargar los datos de porteras
        df_keepers = pd.read_csv("data/data_gold/df_keepers_gold_1.csv")
        # Cargar los datos de jugadoras de campo
        df_players = pd.read_csv("data/data_gold/df_players_gold_1.csv")
        # Cargar datos de información adicional de jugadoras
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

# Definición del diccionario de métricas por posición
position_metrics = {
    'GK': ['Player', 'Squad', 'Born', 'MP','Starts','Min', 'GA','GA90','SoTA','Save%','CS%','Save%_PK','PSxG','PSxG/SoT','PSxG-GA','Pass_Cmp_+40y%','Pass_AvgLen','Stp%','#OPA/90','AvgDist'],
    'DF': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 'Gls', 'Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', 'touch_Def Pen', 'touch_Def 3rd', 'touch_Mid 3rd', 'TO_Succ%', 'CrsPA', 'Tkl/90', 'Tkl%', 'Blocks', 'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1'],
    'MF': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 'Gls', 'Ast', 'G+A', 'SoT/90', 'G/Sh', 'Dist', 'SCA90', 'GCA90', 'Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist', 'PrgDist', 'xA', 'KP', 'pass_1/3', 'PPA', 'CrsPA', 'touch_Mid 3rd', 'touch_Att 3rd', 'touch_Att Pen', 'TO_Succ%', 'carries_TotDist', 'carries_PrgDist', 'PrgR', 'Tkl/90', 'Tkl%', 'Blocks', 'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1'],
    'FW': ['Player', 'Squad', 'Born', 'MP', 'Starts', 'Min', 'Min%', 'Mn/Start', 'Mn/Sub', 'Mn/MP', 'Gls', 'Ast', 'G+A', 'SoT/90', 'G/Sh', 'Dist', 'xG', 'G-xG', 'SCA90', 'GCA90', 'xA', 'KP', 'pass_1/3', 'PPA', 'CrsPA', 'touch_Mid 3rd', 'touch_Att 3rd', 'touch_Att Pen', 'TO_Succ%', 'carries_TotDist', 'carries_PrgDist', 'PrgR', 'Tkl/90', 'Tkl%', 'Blocks', 'Int', 'Tkl+Int', 'Recov', 'CrdY', 'CrdR', '2CrdY', 'Off.1']
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

# Cargar los datos
df_combined, df_players_info, df_teams_info, df_atm_photos = cargar_datos()

# Verificar si se cargaron los datos correctamente
if df_combined is not None and not df_combined.empty:
    # Filtrar jugadoras del Atlético de Madrid para el sidebar
    atm_players = df_combined[df_combined['Squad'] == 'Atlético de Madrid']['Player'].unique()
    
    # Sidebar para seleccionar jugadora a analizar
    st.sidebar.title("Selección de Jugadora")
    
    jugadora_seleccionada = st.sidebar.selectbox(
        "Selecciona una jugadora del Atlético de Madrid:",
        [""] +list(atm_players),
        index=0
    )
    
    df_view = df_combined[df_combined['Player'] == jugadora_seleccionada]


    def get_position_metrics():
        # Métricas para cada posición
        position_metrics = {
                    'GK': ['MP','Starts','Min','Min%','Mn/Start','Mn/Sub','Mn/MP', 'GA','GA90','SoTA','Save%','CS%','Save%_PK','PSxG','PSxG/SoT','PSxG-GA','Pass_Cmp_+40y%','Pass_AvgLen','Stp%','#OPA/90','AvgDist'],
                    'DF': ['MP',
                        'Starts',
                        'Min',
                        'Min%',
                        'Mn/Start',
                        'Mn/Sub'
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
                        'Mn/Sub'
                        'Mn/MP', 
                        'Gls',
                        'Ast',
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
                        'crsPA',
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
                        'Mn/Sub'
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
        
        position_metrics_lev = {
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
        
    # Ejecutar análisis al hacer clic en el botón
    if st.sidebar.button("Analizar Similitudes"):
        with st.spinner("Realizando análisis de similitud..."):
            # Obtener posición de la jugadora seleccionada
            position = df_combined[df_combined['Player'] == jugadora_seleccionada]['Posición Principal'].iloc[0]
           
            # Filtrar el DataFrame según la posición usando las métricas específicas
            if position in position_metrics:
                # Usamos las métricas específicas para la posición
                columnas_filtradas = position_metrics[position]
                df_filtered = df_combined[columnas_filtradas]
                #st.info(f"Usando {len(columnas_filtradas)} métricas específicas para la posición {position}")
            else:
                # Si la posición no está en nuestras categorías, usamos todas las columnas
                st.warning(f"Posición '{position}' no reconocida. Usando todas las métricas disponibles.")
                df_filtered = df_combined
            
            # Seleccionamos las columnas numéricas para el análisis (métricas)
            columnas_metricas = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Preparamos los datos para el análisis
            # Normalizamos los datos para que todas las métricas tengan la misma escala
            scaler = StandardScaler()
            datos_normalizados = scaler.fit_transform(df_filtered[columnas_metricas])
            
            # Creamos un DataFrame con los datos normalizados
            df_normalizado = pd.DataFrame(datos_normalizados, columns=columnas_metricas)
            df_normalizado['Player'] = df_filtered['Player'].values  # Añadimos el nombre para identificación
            
            # Aplicamos PCA para reducir dimensionalidad
            n_componentes = min(10, len(columnas_metricas))  # Usamos hasta 10 componentes principales
            
            pca = PCA(n_components=n_componentes)
            componentes_principales = pca.fit_transform(datos_normalizados)
            
            # Creamos un DataFrame con los componentes principales
            columnas_pca = [f'PC{i+1}' for i in range(n_componentes)]
            df_pca = pd.DataFrame(data=componentes_principales, columns=columnas_pca)
            df_pca['Player'] = df_filtered['Player'].values
            
            # Determinamos cuántos componentes explican al menos el 80% de la varianza
            varianza_explicada = pca.explained_variance_ratio_
            varianza_acumulada = np.cumsum(varianza_explicada)
            n_optimo = np.argmax(varianza_acumulada >= 0.8) + 1
            
            # Aplicamos K-Means para agrupar jugadoras similares
            kmeans = KMeans(n_clusters=min(10, len(df_filtered)), random_state=42)
            clusters = kmeans.fit_predict(componentes_principales[:, :n_optimo])
            
            # Añadimos el cluster asignado a cada jugadora
            df_pca['cluster'] = clusters
            
            # Identificamos el cluster de nuestra jugadora seleccionada
            cluster_jugadora = df_pca.loc[df_pca['Player'] == jugadora_seleccionada, 'cluster'].values[0]
            
            # Encontramos las jugadoras más similares
            # Obtenemos las coordenadas PCA de la jugadora seleccionada
            coords_jugadora = df_pca.loc[df_pca['Player'] == jugadora_seleccionada, columnas_pca[:n_optimo]].values[0]
            
            # Calculamos la distancia euclidiana entre todas las jugadoras y la jugadora seleccionada
            distancias = []
            for idx, row in df_pca.iterrows():
                if row['Player'] != jugadora_seleccionada:
                    coords = row[columnas_pca[:n_optimo]].values
                    distancia = np.linalg.norm(coords - coords_jugadora)
                    # Obtener Squad y Position
                    player_info = df_combined[df_combined['Player'] == row['Player']]
                    squad = player_info['Squad'].iloc[0] if not player_info.empty else "Desconocido"
                    pos = player_info['Posición Principal'].iloc[0] if not player_info.empty else "Desconocido"
                    
                    # Solo incluimos jugadoras de la misma posición para mejorar la relevancia
                    if pos == position:
                        distancias.append((row['Player'], distancia, row['cluster'], squad, pos))
            
            # Ordenamos por distancia para encontrar las más similares
            distancias_ordenadas = sorted(distancias, key=lambda x: x[1])
            
            # Creamos tres pestañas para mostrar diferentes visualizaciones
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Información Básica", "Clustering y Radar", "Análisis", "Índices Compuestos", "Comparativa de Métricas"])
            
            # Pestaña 1: Información básica de las jugadoras similares
            with tab1:
                st.header(f"Información de Jugadoras Similares a {jugadora_seleccionada}")
                
                # Información de la jugadora seleccionada
                st.subheader("Jugadora Seleccionada")
                col1, col2, col3 = st.columns([1,1, 1])
                
                with col1:
                    # Mostrar foto si está disponible
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    jugadora_info = df_combined[df_combined['Player'] == jugadora_seleccionada]
                    if jugadora_info['Squad'].iloc[0] == "Atlético de Madrid" and df_atm_photos is not None:
                        try:
                            jugadora_foto = df_atm_photos[df_atm_photos['Player'] == jugadora_seleccionada]
                            if not jugadora_foto.empty and 'url_photo' in jugadora_foto.columns:
                                photo_url_atm = jugadora_foto['url_photo'].iloc[0]
                                st.image(photo_url_atm, width=150)
                        except:
                            st.write("Foto no disponible")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    display_logo(150)

                with col3:
                    # Datos básicos de la jugadora seleccionada
                    info_seleccionada = df_combined[df_combined['Player'] == jugadora_seleccionada]
                    st.write(f"**Posición:** {info_seleccionada['Posición Principal'].iloc[0]}")
                    st.write(f"**Club:** {info_seleccionada['Squad'].iloc[0]}")
                    
                    # Agregar más información disponible
                    if 'Nation' in info_seleccionada.columns and not pd.isna(info_seleccionada['Nation'].iloc[0]):
                        st.write(f"**Nacionalidad:** {info_seleccionada['Nation'].iloc[0]}")
                    if 'League' in info_seleccionada.columns and not pd.isna(info_seleccionada['League'].iloc[0]):
                        st.write(f"**Liga:** {info_seleccionada['League'].iloc[0]}")
                    if 'Born' in info_seleccionada.columns and not pd.isna(info_seleccionada['Born'].iloc[0]):
                        st.write(f"**Año de nacimiento:** {info_seleccionada['Born'].iloc[0]}")
                
                # Separador
                st.divider()
                
                # Métricas clave de la jugadora seleccionada basadas en su posición
                st.subheader("Métricas destacadas")
                
                # Seleccionamos métricas clave según la posición específica usando position_metrics
                if position in position_metrics:
                    # Filtrar solo métricas numéricas
                    pos_metricas = [m for m in position_metrics[position] 
                                if m in df_combined.select_dtypes(include=['float64', 'int64']).columns 
                                and m not in ['Player', 'Squad', 'Born']]
                    
                    # Seleccionar 4-5 métricas clave para esta posición
                    metricas_clave = []
                    if position == 'GK':
                        metricas_clave = ['GA90', 'Save%', 'CS%', 'PSxG-GA', '#OPA/90']
                    elif position == 'DF':
                        metricas_clave = ['Tkl+Int', 'Blocks', 'Recov', 'touch_Def Pen', 'Tkl%']
                    elif position == 'MF':
                        metricas_clave = ['SCA90', 'GCA90', 'pass_1/3', 'PPA', 'touch_Mid 3rd']
                    elif position == 'FW':
                        metricas_clave = ['Gls', 'G/Sh', 'xG', 'SoT/90', 'touch_Att Pen']
                    
                    # Asegurarse de que las métricas seleccionadas existen en los datos
                    metricas_clave = [m for m in metricas_clave if m in pos_metricas]
                    
                    # Mostrar métricas clave en forma de tarjetas
                    if metricas_clave:
                        cols = st.columns(len(metricas_clave))
                        
                        for i, metrica in enumerate(metricas_clave):
                            if metrica in info_seleccionada.columns and not pd.isna(info_seleccionada[metrica].iloc[0]):
                                valor = info_seleccionada[metrica].iloc[0]
                                cols[i].metric(
                                    metric_display_names.get(metrica, metrica),
                                    f"{valor:.2f}"
                                )
                
                # Información detallada de cada jugadora similar
                st.subheader("Jugadoras similares - Información detallada")
                
                # Crear un acordeón para cada jugadora similar
                for i, (nombre, distancia, cluster, squad, pos) in enumerate(distancias_ordenadas[:10], 1):
                    with st.expander(f"{i}. {nombre} - Distancia: {distancia:.4f} - Club: {squad}"):
                        # Dividir en dos columnas para cada jugadora
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Mostrar foto si está disponible
                            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                            # Buscar foto de la jugadora
                            if squad == "Atlético de Madrid" and df_atm_photos is not None:
                                try:
                                    jugadora_foto = df_atm_photos[df_atm_photos['Player'] == nombre]
                                    if not jugadora_foto.empty and 'url_photo' in jugadora_foto.columns:
                                        photo_url_atm = jugadora_foto['url_photo'].iloc[0]
                                        st.image(photo_url_atm, width=150)
                                except:
                                    pass
                            
                            # Si hay información disponible en df_players_info
                            if df_players_info is not None:
                                try:
                                    jugadora_info_adicional = df_players_info[df_players_info['Player'] == nombre]
                                    if not jugadora_info_adicional.empty and 'Photo' in jugadora_info_adicional.columns:
                                        if not pd.isna(jugadora_info_adicional['Photo'].iloc[0]):
                                            photo_url = jugadora_info_adicional['Photo'].iloc[0]
                                            st.image(photo_url, width=150)
                                except:
                                    pass
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            # Datos básicos de la jugadora similar
                            info_jugadora = df_combined[df_combined['Player'] == nombre]
                            
                            # Mostrar distancia respecto a la jugadora seleccionada
                            st.metric("Distancia", f"{distancia:.4f}", 
                                    delta=f"{distancia - distancias_ordenadas[0][1]:.4f}" if i > 1 else "Más similar")
                            
                            st.write(f"**Posición:** {pos}")
                            st.write(f"**Club:** {squad}")
                            
                            # Agregar más información disponible
                            if 'Nation' in info_jugadora.columns and not pd.isna(info_jugadora['Nation'].iloc[0]):
                                st.write(f"**Nacionalidad:** {info_jugadora['Nation'].iloc[0]}")
                            if 'League' in info_jugadora.columns and not pd.isna(info_jugadora['League'].iloc[0]):
                                st.write(f"**Liga:** {info_jugadora['League'].iloc[0]}")
                            if 'Born' in info_jugadora.columns and not pd.isna(info_jugadora['Born'].iloc[0]):
                                st.write(f"**Año de nacimiento:** {info_jugadora['Born'].iloc[0]}")
                        
                        # Sección para mostrar algunos indicadores clave según la posición
                        st.subheader("Métricas destacadas")
                        
                        # Usar las mismas métricas clave definidas anteriormente para consistencia
                        if pos in position_metrics:
                            if pos == 'GK':
                                metricas_clave = ['GA90', 'Save%', 'CS%', 'PSxG-GA', '#OPA/90']
                            elif pos == 'DF':
                                metricas_clave = ['Tkl+Int', 'Blocks', 'Recov', 'touch_Def Pen', 'Tkl%']
                            elif pos == 'MF':
                                metricas_clave = ['SCA90', 'GCA90', 'pass_1/3', 'PPA', 'touch_Mid 3rd']
                            elif pos == 'FW':
                                metricas_clave = ['Gls', 'G/Sh', 'xG', 'SoT/90', 'touch_Att Pen']
                            
                            # Filtrar solo las métricas disponibles
                            metricas_clave = [m for m in metricas_clave if m in info_jugadora.columns]
                            
                            # Crear columnas para las métricas clave
                            if metricas_clave:
                                cols = st.columns(len(metricas_clave))
                                
                                for j, metrica in enumerate(metricas_clave):
                                    if metrica in info_jugadora.columns and not pd.isna(info_jugadora[metrica].iloc[0]):
                                        valor = info_jugadora[metrica].iloc[0]
                                        cols[j].metric(
                                            metric_display_names.get(metrica, metrica),
                                            f"{valor:.2f}"
                                        )
                        
                        # Comparación directa con la jugadora seleccionada
                        st.subheader("Comparación directa con " + jugadora_seleccionada)

                        # Usar exactamente las métricas definidas en position_metrics para esta posición
                        if pos in position_metrics:
                            metricas_posicion = position_metrics[pos]
                            
                            # Filtrar identificadores y mantener solo métricas numéricas
                            metricas_posicion = [m for m in metricas_posicion 
                                                if m not in ['Player', 'Squad', 'Born', 'Pos', 'Nation', 'Comp', 'Age']
                                                and m in df_combined.select_dtypes(include=['float64', 'int64']).columns]
                            
                            # Preparar datos para el radar
                            fig_radar_full = plt.figure(figsize=(10, 10))
                            ax_full = fig_radar_full.add_subplot(111, polar=True)
                            
                            try:
                                # Obtener datos para ambas jugadoras
                                datos_seleccionada = []
                                datos_similar = []
                                metricas_validas = []
                                
                                # Filtrar solo métricas disponibles para ambas jugadoras
                                for metrica in metricas_posicion:
                                    try:
                                        # Verificar que la métrica está disponible para ambas jugadoras
                                        val_original = df_combined[df_combined['Player'] == jugadora_seleccionada][metrica].iloc[0]
                                        val_similar = info_jugadora[metrica].iloc[0]
                                        
                                        if pd.notna(val_original) and pd.notna(val_similar):
                                            datos_seleccionada.append(val_original)
                                            datos_similar.append(val_similar)
                                            metricas_validas.append(metrica)
                                    except:
                                        # Si hay error, ignoramos esta métrica
                                        pass
                                
                                if len(metricas_validas) > 0:
                                    # Normalizar datos (usando valores máximos de todas las jugadoras)
                                    max_valores = []
                                    for m in metricas_validas:
                                        max_val = df_combined[m].max() if df_combined[m].max() > 0 else 1
                                        max_valores.append(max_val)
                                    
                                    datos_seleccionada_norm = [datos_seleccionada[i]/max_valores[i] for i in range(len(datos_seleccionada))]
                                    datos_similar_norm = [datos_similar[i]/max_valores[i] for i in range(len(datos_similar))]
                                    
                                    # Mostrar cantidad de métricas en gráfico
                                    st.caption(f"Comparando {len(metricas_validas)} métricas definidas para posición {pos}")
                                    
                                    # Calcular ángulos
                                    angulos = np.linspace(0, 2*np.pi, len(metricas_validas), endpoint=False).tolist()
                                    # Cerrar el círculo
                                    datos_seleccionada_norm = np.append(datos_seleccionada_norm, datos_seleccionada_norm[0])
                                    datos_similar_norm = np.append(datos_similar_norm, datos_similar_norm[0])
                                    angulos += angulos[:1]
                                    
                                    # Dibujar el radar
                                    ax_full.plot(angulos, datos_seleccionada_norm, color='red', linewidth=2, label=jugadora_seleccionada)
                                    ax_full.fill(angulos, datos_seleccionada_norm, color='red', alpha=0.1)
                                    ax_full.plot(angulos, datos_similar_norm, color='blue', linewidth=2, label=nombre)
                                    ax_full.fill(angulos, datos_similar_norm, color='blue', alpha=0.1)
                                    
                                    # Etiquetas y leyenda (usando nombres descriptivos)
                                    plt.xticks(angulos[:-1], [metric_display_names.get(m, m) for m in metricas_validas], size=7)
                                    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
                                    plt.ylim(0, 1)
                                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
                                    plt.title(f"Comparación de métricas de {pos}: {jugadora_seleccionada} vs {nombre}", size=12)
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig_radar_full)
                                    
                                    # Incluir tabla con los valores exactos de todas las métricas
                                    st.write(f"#### Valores exactos de las métricas para posición {pos}")
                                    
                                    # Crear lista de métricas con sus valores y descripciones
                                    datos_tabla_completa = {
                                        "Métrica": [metric_display_names.get(m, m) for m in metricas_validas],
                                        jugadora_seleccionada: [round(float(df_combined[df_combined['Player'] == jugadora_seleccionada][m].iloc[0]), 2) 
                                                            for m in metricas_validas],
                                        nombre: [round(float(info_jugadora[m].iloc[0]), 2) for m in metricas_validas]
                                    }
                                    
                                    # Mostrar tabla completa
                                    st.dataframe(pd.DataFrame(datos_tabla_completa), use_container_width=True)
                                    
                                else:
                                    st.warning(f"No hay suficientes métricas válidas disponibles para la comparación")
                            
                            except Exception as e:
                                st.error(f"Error al generar la comparación: {e}")
                                st.write("Detalles:", str(e))
                        else:
                            st.warning(f"No se encontraron métricas definidas para la posición {pos}")
                
                # Información adicional sobre la interpretación
                st.info("""
                **¿Cómo interpretar estos resultados?** 
                - La distancia es una medida de similitud: valores más bajos indican mayor similitud 
                - Las métricas destacadas te muestran los indicadores clave para la posición
                - La comparación directa te permite identificar fortalezas y debilidades relativas
                - El gráfico radar muestra el perfil general de la jugadora respecto a la seleccionada
                """)
            
            # Pestaña 2: Visualización de clustering y gráfico radar
            with tab2:
                st.header("Análisis de Clustering y Radar de Métricas")
                
                # Visualización de componentes principales y clusters
                st.subheader("Visualización PCA y Clustering")
                
                fig_pca = plt.figure(figsize=(10, 8))
                # Graficamos todos los puntos, coloreados por cluster
                scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], 
                                    cmap='viridis', alpha=0.6, s=50)
                
                # Destacamos la jugadora seleccionada
                idx_jugadora = df_pca.index[df_pca['Player'] == jugadora_seleccionada][0]
                plt.scatter(df_pca.loc[idx_jugadora, 'PC1'], df_pca.loc[idx_jugadora, 'PC2'], 
                            color='red', s=200, edgecolors='black', marker='*', label=jugadora_seleccionada)
                
                # Destacamos las 10 jugadoras más similares
                for nombre, distancia, _, _, _ in distancias_ordenadas[:10]:
                    idx = df_pca.index[df_pca['Player'] == nombre][0]
                    plt.scatter(df_pca.loc[idx, 'PC1'], df_pca.loc[idx, 'PC2'], 
                                color='orange', s=100, edgecolors='black', alpha=0.8)
                    plt.annotate(nombre, (df_pca.loc[idx, 'PC1'], df_pca.loc[idx, 'PC2']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.colorbar(scatter, label='Cluster')
                plt.title(f'Visualización PCA de Jugadoras Similares a {jugadora_seleccionada}')
                plt.xlabel('Componente Principal 1')
                plt.ylabel('Componente Principal 2')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                # Guardar figura PCA para el PDF
                st.session_state.fig_pca_saved = fig_pca
                
                st.pyplot(fig_pca)
                
                # Explicación de los componentes principales
                st.markdown("### Explicación de la Varianza")
                
                # Gráfico de varianza explicada
                fig_var = plt.figure(figsize=(10, 6))
                plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.7, label='Varianza Individual')
                plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', label='Varianza Acumulada')
                plt.axhline(y=0.8, color='r', linestyle='--', label='Umbral 80%')
                plt.xlabel('Componentes Principales')
                plt.ylabel('Ratio de Varianza Explicada')
                plt.title('Varianza Explicada por Componentes Principales')
                plt.legend()
                plt.grid(True)
                
                st.pyplot(fig_var)
                
                st.write(f"Se necesitan {n_optimo} componentes para explicar al menos el 80% de la varianza.")
                
                # Gráfico de radar para comparar métricas
                st.subheader("Gráfico Radar de Métricas")
                
                # Usar las métricas específicas para la posición desde position_metrics
                if position in position_metrics:
                    # Obtener métricas numéricas para esta posición
                    relevant_metrics = [m for m in position_metrics[position] 
                                    if m in df_combined.select_dtypes(include=['float64', 'int64']).columns
                                    and m not in ['Player', 'Squad', 'Born']]
                    
                    st.write(f"Métricas seleccionadas para el radar: métricas específicas para {position}")
                    st.write(f"Número de métricas a visualizar: {len(relevant_metrics)}")
                    
                    # Seleccionamos las 5 jugadoras más similares para el gráfico de radar
                    jugadoras_radar = [jugadora_seleccionada] + [nombre for nombre, _, _, _, _ in distancias_ordenadas[:8]]
                    
                    # Verificar que las jugadoras existen en el DataFrame
                    jugadoras_disponibles = [j for j in jugadoras_radar if j in df_combined['Player'].values]
                    if len(jugadoras_disponibles) < len(jugadoras_radar):
                        st.warning(f"Algunas jugadoras no se encontraron en el DataFrame: {set(jugadoras_radar) - set(jugadoras_disponibles)}")
                    
                    # Preparar datos para el gráfico de radar
                    try:
                        # Configurar el gráfico de radar
                        fig_radar = plt.figure(figsize=(10, 10))
                        ax = fig_radar.add_subplot(111, polar=True)
                        
                        # Calcular el ángulo para cada métrica
                        angulos = np.linspace(0, 2*np.pi, len(relevant_metrics), endpoint=False).tolist()
                        
                        # Crear una paleta de colores para diferenciar las jugadoras
                        colores = plt.cm.tab10(np.linspace(0, 1, len(jugadoras_disponibles)))
                        
                        # Matriz para almacenar los valores normalizados de todas las jugadoras
                        all_values = []
                        
                        # Primero recopilamos todos los valores para normalización global
                        for jugadora in jugadoras_disponibles:
                            # Obtenemos datos cuidadosamente, una columna a la vez
                            valores_jugadora = []
                            for metrica in relevant_metrics:
                                try:
                                    # Extraer el valor específico para esta jugadora y métrica
                                    valor = df_combined.loc[df_combined['Player'] == jugadora, metrica].iloc[0]
                                    valores_jugadora.append(valor)
                                except Exception as e:
                                    valores_jugadora.append(0)  # Valor por defecto en caso de error
                            
                            all_values.append(valores_jugadora)
                        
                        # Normalizar todos los valores conjuntamente
                        normalized_values = []
                        for i, metrica in enumerate(relevant_metrics):
                            # Obtener todos los valores para esta métrica
                            metric_values = [values[i] for values in all_values]
                            min_val = min(metric_values)
                            max_val = max(metric_values)
                            
                            # Evitar división por cero
                            if max_val > min_val:
                                normalized_metric = [(v - min_val) / (max_val - min_val) for v in metric_values]
                            else:
                                normalized_metric = [0.5 for _ in metric_values]  # Valor medio si todos son iguales
                            
                            # Almacenar valores normalizados por métrica
                            for j in range(len(all_values)):
                                normalized_values[j].append(normalized_metric[j])
                        
                        # Dibujar cada jugadora en el gráfico de radar
                        for i, jugadora in enumerate(jugadoras_disponibles):
                            valores = normalized_values[i]
                            
                            # Completar el círculo repitiendo el primer valor
                            valores_completos = valores + [valores[0]]
                            angulos_completos = angulos + [angulos[0]]
                            
                            # Destacar la jugadora seleccionada con línea más gruesa
                            linewidth = 3 if jugadora == jugadora_seleccionada else 1.5
                            ax.plot(angulos_completos, valores_completos, linewidth=linewidth, linestyle='solid', label=jugadora, color=colores[i])
                            ax.fill(angulos_completos, valores_completos, alpha=0.1, color=colores[i])
                        
                        # Añadir las etiquetas para cada métrica (usando nombres descriptivos)
                        plt.xticks(angulos, [metric_display_names.get(m, m) for m in relevant_metrics], size=8)
                        
                        # Añadir las líneas de la red para cada nivel
                        ax.set_rlabel_position(0)
                        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
                        plt.ylim(0, 1)
                        
                        # Ajustar la leyenda y el título
                        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        plt.title(f'Comparación de Métricas: {jugadora_seleccionada} vs Jugadoras Similares', size=15)
                        
                        # Guardar figura Radar para el PDF
                        st.session_state.fig_radar_saved = fig_radar
                        
                        st.pyplot(fig_radar)
                    
                    except Exception as e:
                        st.error(f"Error al crear el gráfico de radar: {e}")
                        st.write("Detalles del error:", str(e))
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.warning(f"No se encontraron métricas definidas para la posición {position} en position_metrics")
                
                # Información sobre la interpretación del gráfico de radar
                st.info("""
                **Interpretación del gráfico de radar:**
                - Cada eje representa una métrica diferente, normalizada en una escala de 0 a 1
                - La jugadora seleccionada se destaca con una línea más gruesa
                - Cuanto mayor sea el área cubierta, mejor es el rendimiento general
                - Diferentes formas en el radar indican diferentes estilos de juego
                """)
            
            # Pestaña 4: Gráficos de barras comparativos por métricas individuales
            with tab5:
                st.header("Comparativa de Métricas Individuales")
                st.write(f"Comparando a {jugadora_seleccionada} ({position}) con las jugadoras más similares")
                
                # Usamos position_metrics para el análisis completo 
                
                # Número de jugadoras a comparar (5 por defecto, ajustable)
                #num_jugadoras = st.slider("Número de jugadoras a comparar:", min_value=2, max_value=5, value=5)
                
                # Obtener todas las métricas relevantes según la posición desde position_metrics
                if position in position_metrics:
                    metrics_to_display = position_metrics[position]
                    
                    # Excluir columnas de identificación
                    exclude_cols = ['Player', 'Squad', 'Born', 'Pos', 'Nation', 'Comp', 'Age', 'Born']
                    metrics_to_display = [m for m in metrics_to_display if m not in exclude_cols]
                    
                    # Verificar que las métricas existen en el dataframe
                    metrics_to_display = [m for m in metrics_to_display if m in df_combined.columns]
                    
                    # Mostrar el título con la posición
                    st.subheader(f"Métricas completas para la posición: {position} ({len(metrics_to_display)} métricas)")
                    
                    # Seleccionar las jugadoras a comparar
                    jugadoras_comparar = [jugadora_seleccionada] + [nombre for nombre, _, _, _, _ in distancias_ordenadas[:8]]
                    
                    # Crear un contenedor con scroll para los gráficos
                    graph_container = st.container()
                    
                    with graph_container:
                        # Organizar en una grid: 2 columnas
                        col1, col2 = st.columns(2)
                        cols = [col1, col2]
                        
                        # Crear un gráfico para cada métrica
                        for i, metric in enumerate(metrics_to_display):
                            # Alternar entre columnas
                            col = cols[i % 2]
                            
                            with col:
                                # Título descriptivo de la métrica
                                metric_name = metric_display_names.get(metric, metric)
                                st.write(f"### {metric_name}")
                                
                                # Crear figura para el gráfico de barras
                                fig_bar = plt.figure(figsize=(10, 5))
                                
                                # Extraer valores para las jugadoras seleccionadas
                                nombres = []
                                valores = []
                                
                                for jugadora in jugadoras_comparar:
                                    if jugadora in df_combined['Player'].values:
                                        try:
                                            valor = df_combined[df_combined['Player'] == jugadora][metric].iloc[0]
                                            # Verificar si el valor es numérico
                                            if not pd.isna(valor):
                                                # Truncar los nombres largos para mejor visualización
                                                nombre_corto = jugadora[:15] + '...' if len(jugadora) > 15 else jugadora
                                                nombres.append(nombre_corto)
                                                valores.append(valor)
                                        except Exception as e:
                                            st.error(f"Error al obtener datos para {jugadora} en la métrica {metric}: {e}")
                                
                                if nombres and valores:
                                    # Crear paleta de colores: destacar la jugadora seleccionada
                                    colores = ['#ff7f0e' if nombre.startswith(jugadora_seleccionada[:15]) else '#1f77b4' for nombre in nombres]
                                    
                                    # Crear el gráfico de barras
                                    bars = plt.bar(nombres, valores, color=colores)
                                    
                                    # Añadir etiquetas y título
                                    plt.title(f"{metric_name}", fontsize=12)
                                    plt.ylabel(metric)
                                    plt.xticks(rotation=45, ha='right')
                                    
                                    # Añadir valores sobre las barras
                                    for bar in bars:
                                        height = bar.get_height()
                                        if height != 0:  # Evitar errores con valores cero
                                            max_value = max(valores) if max(valores) > 0 else 1
                                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max_value,
                                                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
                                    
                                    plt.tight_layout()
                                    plt.grid(axis='y', alpha=0.3)
                                    
                                    # Mostrar el gráfico
                                    st.pyplot(fig_bar)
                                else:
                                    st.warning(f"No hay datos disponibles para la métrica {metric}")
                    
                    # Información sobre la interpretación
                    st.info("""
                    **Comparación de métricas individuales:**
                    - Las barras naranjas representan a la jugadora seleccionada
                    - Las barras azules representan a las jugadoras similares
                    - Estas comparaciones te permiten identificar fortalezas y debilidades específicas
                    - Puedes usar esta información para decidir qué jugadoras podrían ser buenas alternativas o complementos
                    """)
                else:
                    st.warning(f"No se encontraron métricas definidas para la posición {position} en position_metrics")
            
            # Pestaña 3: Análisis IA con DAFO y recomendaciones
            with tab3:
                st.header(f"Análisis para {jugadora_seleccionada}")
            
                # Contenedor para el análisis DAFO
                dafo_container = st.container()
                
                with dafo_container:
                    st.write("### Análisis DAFO")
                    
                    # Función para generar el análisis DAFO basado en los datos y métricas
                    def generar_dafo(jugadora, posicion, metricas_jugadora, metricas_similares, metricas_promedio):
                        """
                        Genera un análisis DAFO para la jugadora basado en sus métricas y comparándola con jugadoras similares.
                        
                        Args:
                            jugadora: Nombre de la jugadora
                            posicion: Posición de la jugadora
                            metricas_jugadora: Dict con las métricas de la jugadora
                            metricas_similares: Dict con las métricas de jugadoras similares
                            metricas_promedio: Dict con los promedios por posición
                        
                        Returns:
                            Dict con el análisis DAFO (debilidades, amenazas, fortalezas, oportunidades)
                        """
                        fortalezas = []
                        debilidades = []
                        oportunidades = []
                        amenazas = []
                        
                        # Verificamos si la posición existe en nuestro mapeo
                        if posicion in position_metrics:
                            metricas_posicion = [m for m in position_metrics[posicion] 
                                    if m not in ['Player', 'Squad', 'Born', 'Pos', 'Nation', 'Comp', 'Age']]
                            
                            # Analizamos fortalezas y debilidades basadas en métricas específicas de posición
                            for metrica in metricas_posicion:
                                if metrica in metricas_jugadora and metrica in metricas_promedio:
                                    # Obtenemos el valor de la jugadora y el promedio del equipo
                                    valor_jugadora = metricas_jugadora.get(metrica)
                                    promedio_posicion = metricas_promedio.get(metrica)
                                    
                                    if pd.notna(valor_jugadora) and pd.notna(promedio_posicion) and promedio_posicion > 0:
                                        # Calculamos el porcentaje de diferencia
                                        diff_porcentaje = ((valor_jugadora - promedio_posicion) / promedio_posicion) * 100
                                        metrica_nombre = metric_display_names.get(metrica, metrica)
                                        
                                        # Determinamos si es una fortaleza o debilidad
                                        if diff_porcentaje >= 15:  # 15% mejor que el promedio
                                            fortalezas.append(f"**{metrica_nombre}**: Destaca con un {valor_jugadora:.2f} (un {abs(diff_porcentaje):.1f}% superior al promedio de su posición)")
                                        elif diff_porcentaje <= -15:  # 15% peor que el promedio
                                            debilidades.append(f"**{metrica_nombre}**: Por debajo con un {valor_jugadora:.2f} (un {abs(diff_porcentaje):.1f}% inferior al promedio de su posición)")
                            
                            # Analizamos oportunidades y amenazas basadas en comparaciones y tendencias
                            for metrica in metricas_posicion:
                                if metrica in metricas_jugadora:
                                    valores_similares = [s.get(metrica, 0) for s in metricas_similares if metrica in s]
                                    if valores_similares:
                                        mejor_similar = max(valores_similares)
                                        valor_jugadora = metricas_jugadora.get(metrica)
                                        
                                        if pd.notna(valor_jugadora) and pd.notna(mejor_similar) and mejor_similar > 0 and valor_jugadora > 0:
                                            diff_porcentaje = ((mejor_similar - valor_jugadora) / valor_jugadora) * 100
                                            metrica_nombre = metric_display_names.get(metrica, metrica)
                                            
                                            if diff_porcentaje >= 20:  # 20% mejor que nuestra jugadora
                                                oportunidades.append(f"**{metrica_nombre}**: Potencial para mejorar un {abs(diff_porcentaje):.1f}% hasta {mejor_similar:.2f} (referencia de jugadoras similares)")
                                            
                                            # Identificar métricas donde está muy por encima de similares (posible riesgo de regresión)
                                            if (valor_jugadora - mejor_similar) / mejor_similar > 0.3:
                                                amenazas.append(f"**{metrica_nombre}**: Rendimiento actual de {valor_jugadora:.2f} podría ser difícil de mantener (un {((valor_jugadora - mejor_similar) / mejor_similar * 100):.1f}% superior a jugadoras similares)")
                        
                        # Agregar análisis específicos por posición basados en position_metrics
                        if posicion == 'GK':
                            if 'GA90' in metricas_jugadora and 'Save%' in metricas_jugadora:
                                if metricas_jugadora['GA90'] > 1.2:
                                    amenazas.append("Alto ratio de goles encajados podría indicar vulnerabilidad ante ciertos tipos de ataque")
                                if metricas_jugadora['Save%'] < 65:
                                    oportunidades.append("Mejorar en porcentaje de paradas para aumentar la solidez defensiva")
                        
                        elif posicion == 'DF':
                            if 'Tkl+Int' in metricas_jugadora and 'Blocks' in metricas_jugadora:
                                if metricas_jugadora['Tkl+Int'] < 3:
                                    oportunidades.append("Aumentar acciones defensivas como tackles e intercepciones")
                                if metricas_jugadora['Blocks'] > 2:
                                    fortalezas.append("Buena capacidad para bloquear tiros y pases peligrosos")
                        
                        elif posicion == 'MF':
                            if 'pass_1/3' in metricas_jugadora and 'SCA90' in metricas_jugadora:
                                if metricas_jugadora.get('pass_1/3', 0) > 5:
                                    fortalezas.append("Excelente capacidad para hacer progresar el balón al último tercio")
                                if metricas_jugadora.get('SCA90', 0) < 2:
                                    oportunidades.append("Potencial para mejorar en la creación de oportunidades de tiro")
                        
                        elif posicion == 'FW':
                            if 'G/Sh' in metricas_jugadora and 'G-xG' in metricas_jugadora:
                                if metricas_jugadora.get('G/Sh', 0) < 0.1:
                                    oportunidades.append("Mejorar la eficiencia en la finalización de oportunidades")
                                if metricas_jugadora.get('G-xG', 0) > 0:
                                    fortalezas.append("Sobrerrendimiento en goles respecto a lo esperado por las oportunidades")
                        
                        # Si no tenemos suficientes puntos, agregamos algunos genéricos
                        if len(fortalezas) < 3:
                            fortalezas.append("Jugadora con potencial para desarrollarse en su posición")
                        if len(debilidades) < 2:
                            debilidades.append("Datos insuficientes para identificar áreas de mejora específicas")
                        if len(oportunidades) < 3:
                            oportunidades.append("Analizar jugadoras de élite en la misma posición para adoptar mejores prácticas")
                        if len(amenazas) < 2:
                            amenazas.append("La competencia en la misma posición podría limitar las oportunidades de juego")
                        
                        return {
                            "debilidades": debilidades, 
                            "amenazas": amenazas, 
                            "fortalezas": fortalezas, 
                            "oportunidades": oportunidades
                        }
                    
                    # Obtener datos de la jugadora seleccionada
                    jugadora_info = df_combined[df_combined['Player'] == jugadora_seleccionada]
                    
                    # Obtener métricas de la jugadora
                    metricas_jugadora = {}
                    # Usar las métricas específicas de la posición
                    if position in position_metrics:
                        metricas_numericas = [m for m in position_metrics[position] 
                                            if m in jugadora_info.select_dtypes(include=['float64', 'int64']).columns]
                    else:
                        metricas_numericas = jugadora_info.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    
                    for metrica in metricas_numericas:
                        if metrica in jugadora_info.columns:
                            metricas_jugadora[metrica] = jugadora_info[metrica].iloc[0]
                    
                    # Obtener datos de jugadoras similares
                    metricas_similares = []
                    for nombre, _, _, _, _ in distancias_ordenadas[:5]:
                        similar_info = df_combined[df_combined['Player'] == nombre]
                        metricas_similar = {}
                        for metrica in metricas_numericas:
                            if metrica in similar_info.columns:
                                try:
                                    metricas_similar[metrica] = similar_info[metrica].iloc[0]
                                except:
                                    pass  # Si hay error, omitimos esta métrica
                        metricas_similares.append(metricas_similar)
                    
                    # Obtener promedios por posición
                    position = jugadora_info['Posición Principal'].iloc[0]
                    jugadoras_misma_posicion = df_combined[df_combined['Posición Principal'] == position]
                    
                    metricas_promedio = {}
                    for metrica in metricas_numericas:
                        if metrica in jugadoras_misma_posicion.columns:
                            metricas_promedio[metrica] = jugadoras_misma_posicion[metrica].mean()
                    
                    # Generar el DAFO
                    try:
                        dafo = generar_dafo(
                            jugadora_seleccionada, 
                            position, 
                            metricas_jugadora, 
                            metricas_similares, 
                            metricas_promedio
                        )
                        
                        # Mostrar DAFO en una presentación visual clara
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Fortalezas")
                            st.markdown('<div style="background-color: #d4edda; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                            for fortaleza in dafo["fortalezas"]:
                                st.markdown(f"✅ {fortaleza}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("#### Debilidades")
                            st.markdown('<div style="background-color: #f8d7da; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                            for debilidad in dafo["debilidades"]:
                                st.markdown(f"❌ {debilidad}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### Oportunidades")
                            st.markdown('<div style="background-color: #cce5ff; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                            for oportunidad in dafo["oportunidades"]:
                                st.markdown(f"🚀 {oportunidad}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("#### Amenazas")
                            st.markdown('<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
                            for amenaza in dafo["amenazas"]:
                                st.markdown(f"⚠️ {amenaza}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error al generar el análisis DAFO: {e}")
                        st.info("Intenta con otra jugadora o verifica los datos disponibles.")
                    
                    # Separador para la siguiente sección
                    st.divider()

                
                        
                # Información de interpretación
                st.info("""
                **Nota sobre el análisis IA:**
                - El análisis se basa únicamente en datos estadísticos disponibles
                - Las recomendaciones son generales y deben ser evaluadas por el cuerpo técnico
                - El análisis DAFO y las métricas a mejorar son herramientas orientativas para la toma de decisiones
                - Se recomienda complementar este análisis con la observación directa de los partidos
                """)
                
                # Sección de ayuda y metodología
                with st.expander("ℹ️ Metodología del análisis IA"):
                    st.markdown("""
                    ### Metodología del análisis IA
                    
                    Este análisis utiliza técnicas de inteligencia artificial para interpretar datos estadísticos y generar conclusiones significativas sobre el rendimiento de las jugadoras.
                    
                    **Proceso del análisis:**
                    
                    1. **Recopilación de datos**: Se analizan las métricas disponibles de la jugadora seleccionada.
                    2. **Análisis comparativo**: Se comparan estas métricas con:
                    - Jugadoras similares identificadas mediante algoritmos de clustering
                    - Promedios por posición en la misma liga/competición
                    3. **Identificación de patrones**: Se detectan fortalezas, debilidades, oportunidades y amenazas.
                    
                    **Limitaciones a considerar:**
                    
                    - El análisis está limitado a las métricas disponibles en la base de datos
                    - No considera factores cualitativos como liderazgo, comunicación o inteligencia táctica
                    - Las recomendaciones son generales y deben ser adaptadas al contexto específico del equipo
                    - La interpretación final debe realizarse por profesionales con conocimiento del contexto
                    
                    **Uso recomendado:**
                    
                    Este análisis debe utilizarse como herramienta complementaria en el proceso de toma de decisiones, no como sustituto del criterio técnico profesional.
                    """)
                
                # Sección de posibles próximos pasos
                with st.expander("🔄 Evolución y seguimiento"):
                    st.markdown("""
                    ### Seguimiento y evolución
                    
                    Para un análisis más completo, se recomienda:
                    
                    1. **Establecer métricas de seguimiento** específicas para la jugadora basadas en las áreas de mejora identificadas
                    2. **Crear un plan de desarrollo personalizado** con objetivos a corto, medio y largo plazo
                    3. **Realizar revisiones periódicas** para evaluar el progreso y ajustar el plan según sea necesario
                    4. **Comparar tendencias temporales** para identificar patrones de mejora o áreas de estancamiento
                    
                    Un enfoque integral debería combinar:
                    
                    - **Análisis de datos**: Métricas cuantitativas y tendencias
                    - **Evaluación técnica**: Observación directa de habilidades y técnica
                    - **Feedback cualitativo**: Aportaciones del cuerpo técnico y compañeras
                    - **Autoevaluación**: Percepción de la propia jugadora sobre su rendimiento
                    
                    La visualización periódica de estos informes puede ayudar tanto al cuerpo técnico como a la jugadora a entender mejor su evolución y potencial.
            
                    """)

            # Pestaña 5: Análisis de Índices Compuestos

            with tab4:
                st.header("Análisis por Índices Compuestos")
                st.write(f"Comparando a {jugadora_seleccionada} ({position}) con jugadoras similares utilizando índices especializados")
                
                # Obtener las 10 jugadoras más similares de la clusterización inicial
                jugadoras_similares = [nombre for nombre, _, _, _, _ in distancias_ordenadas[:10]]
                
                # Definir índices compuestos por posición
                indices_por_posicion = {
                    'GK': {
                        'Índice de Paradas': {
                            'metricas': ['Save%', 'CS%', 'PSxG-GA'],
                            'pesos': [0.5, 0.3, 0.2],
                            'descripcion': 'Evalúa la capacidad de evitar goles combinando porcentaje de paradas, porterías a cero y rendimiento vs. expectativa'
                        },
                        'Índice de Juego Aéreo': {
                            'metricas': ['Stp%', '#OPA/90'],
                            'pesos': [0.6, 0.4],
                            'descripcion': 'Mide la dominancia en el juego aéreo y salidas de portería'
                        },
                        'Índice de Distribución': {
                            'metricas': ['Pass_Cmp_+40y%', 'Pass_AvgLen'],
                            'pesos': [0.7, 0.3],
                            'descripcion': 'Evalúa la capacidad para distribuir el balón con precisión y rango de pase'
                        },
                        'Índice de Rendimiento Bajo Presión': {
                            'metricas': ['Save%_PK', 'PSxG-GA', 'GA90'],
                            'pesos': [0.4, 0.4, 0.2],
                            'descripcion': 'Evalúa el rendimiento en situaciones de alta presión como penaltis y momentos críticos'
                        }
                    },
                    'DF': {
                        'Índice de Solidez Defensiva': {
                            'metricas': ['Tkl%', 'Blocks', 'Int', 'Recov'],
                            'pesos': [0.3, 0.2, 0.3, 0.2],
                            'descripcion': 'Evalúa la capacidad para interceptar ataques y recuperar balones'
                        },
                        'Índice de Construcción': {
                            'metricas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'PrgDist'],
                            'pesos': [0.2, 0.3, 0.2, 0.3],
                            'descripcion': 'Mide la contribución al juego ofensivo desde posiciones defensivas'
                        },
                        'Índice de Posicionamiento': {
                            'metricas': ['touch_Def Pen', 'touch_Def 3rd', 'touch_Mid 3rd'],
                            'pesos': [0.3, 0.4, 0.3],
                            'descripcion': 'Evalúa el posicionamiento y cobertura de espacios en campo'
                        },
                        'Índice de Seguridad': {
                            'metricas': ['CrdY', 'CrdR', 'Tkl%', 'Recov'],
                            'pesos': [0.25, 0.25, 0.25, 0.25],
                            'descripcion': 'Mide la fiabilidad y disciplina defensiva, minimizando errores y amonestaciones'
                        },
                        'Índice de Juego Aéreo': {
                            'metricas': ['Blocks', 'Int', 'Recov'],
                            'pesos': [0.4, 0.3, 0.3],
                            'descripcion': 'Evalúa específicamente la dominancia en el juego aéreo y situaciones de balón alto'
                        }
                    },
                    'MF': {
                        'Índice de Control': {
                            'metricas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'PrgDist', 'pass_1/3'],
                            'pesos': [0.15, 0.25, 0.20, 0.15, 0.25],
                            'descripcion': 'Mide la capacidad para controlar el juego y hacer progresar el balón'
                        },
                        'Índice de Presión': {
                            'metricas': ['Tkl/90', 'Tkl%', 'Int', 'Recov'],
                            'pesos': [0.25, 0.25, 0.25, 0.25],
                            'descripcion': 'Evalúa la contribución defensiva y capacidad de recuperación'
                        },
                        'Índice de Creación': {
                            'metricas': ['xA', 'KP', 'SCA90', 'GCA90'],
                            'pesos': [0.3, 0.2, 0.25, 0.25],
                            'descripcion': 'Mide la contribución a la creación de oportunidades de gol'
                        },
                        'Índice de Versatilidad': {
                            'metricas': ['G+A', 'Tkl+Int', 'pass_1/3', 'Recov'],
                            'pesos': [0.25, 0.25, 0.25, 0.25],
                            'descripcion': 'Identifica mediocampistas completas que contribuyen en ambas fases del juego'
                        },
                        'Índice de Progresión': {
                            'metricas': ['carries_PrgDist', 'PrgDist', 'TO_Succ%', 'PrgR'],
                            'pesos': [0.3, 0.3, 0.2, 0.2],
                            'descripcion': 'Mide la capacidad para hacer avanzar al equipo hacia posiciones ofensivas'
                        }
                    },
                    'FW': {
                        'Índice de Efectividad Ofensiva': {
                            'metricas': ['Gls', 'G/Sh', 'xG', 'G-xG'],
                            'pesos': [0.3, 0.25, 0.25, 0.2],
                            'descripcion': 'Evalúa la capacidad goleadora y eficiencia en la finalización'
                        },
                        'Índice de Creación': {
                            'metricas': ['Ast', 'xA', 'SCA90', 'GCA90'],
                            'pesos': [0.3, 0.25, 0.25, 0.2],
                            'descripcion': 'Mide la contribución a la creación de ocasiones para compañeras'
                        },
                        'Índice de Movimiento Ofensivo': {
                            'metricas': ['touch_Att 3rd', 'touch_Att Pen', 'carries_PrgDist', 'PrgR'],
                            'pesos': [0.2, 0.3, 0.2, 0.3],
                            'descripcion': 'Evalúa el posicionamiento y movimiento en zonas ofensivas'
                        },
                        'Índice de Presión Ofensiva': {
                            'metricas': ['Tkl/90', 'Recov', 'touch_Att 3rd'],
                            'pesos': [0.4, 0.3, 0.3],
                            'descripcion': 'Evalúa la contribución a la presión alta y al trabajo defensivo en ataque'
                        },
                        'Índice de Autonomía': {
                            'metricas': ['TO_Succ%', 'G+A', 'carries_PrgDist'],
                            'pesos': [0.4, 0.3, 0.3],
                            'descripcion': 'Mide la capacidad para generar peligro por sí misma sin depender de compañeras'
                        }
                    }
                }
                
                # Índices generales para todas las posiciones
                indices_generales = {
                    'Índice de Impacto por Minuto': {
                        'metricas': ['Min', 'MP', 'G+A', 'Tkl+Int', 'Recov'],
                        'pesos': [0.2, 0.2, 0.2, 0.2, 0.2],
                        'descripcion': 'Evalúa eficiencia y rendimiento independientemente del tiempo de juego'
                    }
                }
                
                # Añadir los índices generales a cada posición
                for posicion in indices_por_posicion:
                    for nombre, definicion in indices_generales.items():
                        indices_por_posicion[posicion][nombre] = definicion
                
                # Función para calcular índices normalizados
                def calcular_indices(df, jugadoras, indices_definidos, posicion):
                    # Crear diccionario para almacenar resultados
                    resultados_indices = {}
                    
                    if posicion not in indices_definidos:
                        return {}
                    
                    # Iterar por cada índice definido para esta posición
                    for nombre_indice, definicion in indices_definidos[posicion].items():
                        metricas = definicion['metricas']
                        pesos = definicion['pesos']
                        
                        # Verificar que todas las métricas existen en el dataframe
                        metricas_disponibles = [m for m in metricas if m in df.columns]
                        pesos_disponibles = [pesos[i] for i, m in enumerate(metricas) if m in metricas_disponibles]
                        
                        # Normalizar pesos si cambiaron
                        if pesos_disponibles and sum(pesos_disponibles) != 1.0:
                            suma_pesos = sum(pesos_disponibles)
                            pesos_disponibles = [p/suma_pesos for p in pesos_disponibles]
                        
                        if not metricas_disponibles or not pesos_disponibles:
                            continue
                            
                        # Calcular valores crudos del índice para todas las jugadoras
                        resultados_indices[nombre_indice] = {}
                        
                        # Obtener valores máximos y mínimos para normalización
                        maximos = {}
                        minimos = {}
                        
                        for metrica in metricas_disponibles:
                            # Filtrar jugadoras de la misma posición para obtener max y min
                            valores_posicion = df[df['Posición Principal'] == posicion][metrica]
                            valores_validos = valores_posicion.dropna()
                            
                            if not valores_validos.empty:
                                maximos[metrica] = valores_validos.max()
                                minimos[metrica] = valores_validos.min()
                            else:
                                maximos[metrica] = 1
                                minimos[metrica] = 0
                        
                        # Calcular el índice para cada jugadora
                        for jugadora in jugadoras:
                            # Obtener datos de la jugadora
                            datos_jugadora = df[df['Player'] == jugadora]
                            
                            if datos_jugadora.empty:
                                continue
                            
                            # Calcular valor del índice
                            valor_indice = 0
                            for i, metrica in enumerate(metricas_disponibles):
                                if metrica in datos_jugadora.columns and not pd.isna(datos_jugadora[metrica].iloc[0]):
                                    valor_crudo = datos_jugadora[metrica].iloc[0]
                                    
                                    # Normalizar entre 0 y 1 (evitando división por cero)
                                    if maximos[metrica] > minimos[metrica]:
                                        valor_norm = (valor_crudo - minimos[metrica]) / (maximos[metrica] - minimos[metrica])
                                    else:
                                        valor_norm = 0.5  # Si todos los valores son iguales
                                    
                                    # Acumular valor ponderado
                                    valor_indice += valor_norm * pesos_disponibles[i]
                            
                            # Almacenar resultado
                            resultados_indices[nombre_indice][jugadora] = valor_indice * 100  # Escala 0-100 para facilitar interpretación
                    
                    return resultados_indices
                
                # Obtener jugadoras a comparar
                jugadoras_a_comparar = [jugadora_seleccionada] + jugadoras_similares
                
                # Calcular los índices
                indices_calculados = calcular_indices(df_combined, jugadoras_a_comparar, indices_por_posicion, position)
                
                # Si hay índices calculados, mostrarlos
                if indices_calculados:
                    # Sección para visualizar resultados
                    st.write("### Índices Compuestos para " + position)
                    
                    # Preparar datos para visualización
                    datos_visualizacion = {}
                    
                    for nombre_indice, valores in indices_calculados.items():
                        datos_visualizacion[nombre_indice] = {
                            'Jugadoras': list(valores.keys()),
                            'Valores': list(valores.values())
                        }
                    
                    # 1. VISUALIZACIÓN EN GRÁFICO RADAR
                    st.write("### Visualización en Radar de Índices Compuestos")
                    st.write(f"Comparando el perfil completo de {jugadora_seleccionada} con jugadoras similares")
                    
                    # Preparar datos para el radar
                    try:
                        # Seleccionar las jugadoras para el radar (limitamos a 5 para mejor visibilidad)
                        jugadoras_radar = [jugadora_seleccionada]
                        otras_jugadoras = [j for j in jugadoras_a_comparar if j != jugadora_seleccionada][:4]  # Las 4 más similares
                        jugadoras_radar.extend(otras_jugadoras)
                        
                        # Obtener solo los nombres de los índices
                        nombres_indices = list(indices_calculados.keys())
                        
                        if len(nombres_indices) >= 3:  # Necesitamos al menos 3 índices para un radar significativo
                            # Crear figura para el radar
                            fig_radar = plt.figure(figsize=(10, 10))
                            ax = fig_radar.add_subplot(111, polar=True)
                            
                            # Calcular el ángulo para cada índice
                            angulos = np.linspace(0, 2*np.pi, len(nombres_indices), endpoint=False).tolist()
                            
                            # Crear una paleta de colores para diferenciar las jugadoras
                            colores = plt.cm.tab10(np.linspace(0, 1, len(jugadoras_radar)))
                            
                            # Dibujar cada jugadora en el gráfico de radar
                            for i, jugadora in enumerate(jugadoras_radar):
                                # Obtener valores de los índices para esta jugadora
                                valores = []
                                for indice in nombres_indices:
                                    valor = indices_calculados[indice].get(jugadora, 0)
                                    valores.append(valor / 100)  # Normalizar a escala 0-1 (ya que los índices están en 0-100)
                                
                                # Completar el círculo repitiendo el primer valor
                                valores_completos = valores + [valores[0]]
                                angulos_completos = angulos + [angulos[0]]
                                
                                # Destacar la jugadora seleccionada con línea más gruesa
                                linewidth = 3 if jugadora == jugadora_seleccionada else 1.5
                                ax.plot(angulos_completos, valores_completos, linewidth=linewidth, linestyle='solid', 
                                    label=jugadora, color=colores[i])
                                ax.fill(angulos_completos, valores_completos, alpha=0.1, color=colores[i])
                            
                            # Añadir las etiquetas para cada índice
                            plt.xticks(angulos, nombres_indices, size=9)
                            
                            # Añadir las líneas de la red para cada nivel
                            ax.set_rlabel_position(0)
                            plt.yticks([0.2, 0.4, 0.6, 0.8], ["20", "40", "60", "80"], color="grey", size=8)
                            plt.ylim(0, 1)
                            
                            # Ajustar la leyenda y el título
                            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                            plt.title(f'Perfil de Índices Compuestos: {jugadora_seleccionada} vs Jugadoras Similares', size=14)
                            
                            # Mostrar el gráfico
                            st.pyplot(fig_radar)
                            
                            # Mostrar explicación del gráfico radar
                            st.info("""
                            **Interpretación del gráfico radar:**
                            - Cada eje representa un índice compuesto diferente
                            - La jugadora seleccionada se destaca con una línea más gruesa
                            - Mayor valor en un índice indica mejor rendimiento en ese aspecto
                            - La forma del polígono revela el perfil de fortalezas y debilidades
                            - Perfiles similares indican jugadoras con características parecidas
                            """)
                        else:
                            st.warning("Se necesitan al menos 3 índices para crear un gráfico radar. Algunos índices no pudieron calcularse con los datos disponibles.")
                    
                    except Exception as e:
                        st.error(f"Error al generar el gráfico radar: {e}")
                        st.info("Este error puede deberse a falta de datos suficientes para algunos índices. Intenta con otra jugadora o posición.")
                    
                    # 2. TABLA COMPARATIVA DE ÍNDICES
                    with st.expander("### Tabla Comparativa de Índices"):
                    
                        # Preparar datos para la tabla
                        tabla_datos = {'Jugadora': jugadoras_a_comparar}
                        
                        for nombre_indice, valores in indices_calculados.items():
                            tabla_datos[nombre_indice] = [valores.get(j, float('nan')) for j in jugadoras_a_comparar]
                        
                        # Crear DataFrame
                        df_tabla = pd.DataFrame(tabla_datos)
                        
                        # Mostrar tabla con formato
                        st.dataframe(df_tabla.style.format({col: "{:.1f}" for col in df_tabla.columns if col != 'Jugadora'}), 
                                    use_container_width=True)
                    
                    # 3. GRÁFICOS DE BARRAS (EN DESPLEGABLE)
                    with st.expander("Ver comparativas detalladas por índice (gráficos de barras)", expanded=False):
                        # Sección para visualizar resultados
                        st.write("### Índices Compuestos para " + position)
                        
                        # Preparar datos para visualización
                        datos_visualizacion = {}
                        
                        for nombre_indice, valores in indices_calculados.items():
                            datos_visualizacion[nombre_indice] = {
                                'Jugadoras': list(valores.keys()),
                                'Valores': list(valores.values())
                            }
                        
                        # Visualizar los índices en gráficos de barras
                        for nombre_indice, datos in datos_visualizacion.items():
                            st.write(f"#### {nombre_indice}")
                            st.write(indices_por_posicion[position][nombre_indice]['descripcion'])
                            
                            # Crear el gráfico
                            fig = plt.figure(figsize=(10, 5))
                            
                            # Ordenar los datos de mayor a menor valor
                            indices_ordenados = sorted(zip(datos['Jugadoras'], datos['Valores']), 
                                                    key=lambda x: x[1], reverse=True)
                            
                            nombres_ordenados = [n[:15] + '...' if len(n) > 15 else n for n, _ in indices_ordenados]
                            valores_ordenados = [v for _, v in indices_ordenados]
                            
                            # Crear colores, destacando la jugadora seleccionada
                            colores = ['#ff7f0e' if nombre.startswith(jugadora_seleccionada[:15]) else '#1f77b4' 
                                    for nombre in nombres_ordenados]
                            
                            # Crear gráfico de barras
                            bars = plt.bar(nombres_ordenados, valores_ordenados, color=colores)
                            
                            # Añadir valores sobre las barras
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
                            
                            plt.title(nombre_indice)
                            plt.ylabel("Puntuación (0-100)")
                            plt.xticks(rotation=45, ha='right')
                            plt.ylim(0, 105)  # Espacio para las etiquetas
                            plt.grid(axis='y', alpha=0.3)
                            plt.tight_layout()
                            
                            # Mostrar el gráfico
                            st.pyplot(fig)
                    
                    # Explicación específica por posición
                    st.write("## Interpretación de Índices para " + position)
                    
                    # Explicaciones específicas por posición
                    explicaciones = {
                        'GK': """
                        ### Interpretación de Índices para Porteras
                        
                        Los índices compuestos para porteras evalúan cuatro aspectos fundamentales del juego:
                        
                        **Índice de Paradas**: Este índice refleja la capacidad fundamental de una portera para evitar goles. 
                        Combina el porcentaje de paradas realizadas, la frecuencia de porterías a cero (clean sheets) y el rendimiento 
                        respecto a los goles esperados. Un valor alto indica una portera que constantemente realiza más paradas de las 
                        esperadas y mantiene su portería imbatida.
                        
                        **Índice de Juego Aéreo**: Evalúa el dominio de la portera en situaciones aéreas, particularmente en centros 
                        y balones colgados. Combina el porcentaje de centros interceptados con la frecuencia de acciones fuera del área.
                        Las porteras con valores altos son proactivas y dominantes en el juego aéreo.
                        
                        **Índice de Distribución**: Mide la contribución de la portera al juego ofensivo del equipo a través de su 
                        distribución del balón. Valora tanto la precisión en pases largos como la capacidad para variar distancias.
                        Un valor alto indica una portera moderna que participa activamente en la fase de construcción.
                        
                        **Índice de Rendimiento Bajo Presión**: Evalúa la capacidad de la portera para responder en situaciones críticas,
                        como penaltis, y su rendimiento general en momentos de alta presión. Un valor alto indica una portera con temple
                        y capacidad para decidir partidos en momentos clave.
                        
                        **Índice de Impacto por Minuto**: Mide la eficiencia de la portera normalizando su rendimiento respecto al tiempo
                        jugado, permitiendo comparar jugadoras con diferente cantidad de minutos.
                        
                        Estos índices permiten identificar diferentes perfiles de porteras y ayudan a evaluar qué aspectos 
                        específicos podría mejorar una jugadora para elevar su rendimiento general.
                        """,
                        
                        'DF': """
                        ### Interpretación de Índices para Defensas
                        
                        Los índices compuestos para defensas evalúan cinco dimensiones clave del juego defensivo:
                        
                        **Índice de Solidez Defensiva**: Este índice mide la eficacia de una defensa para detener ataques rivales.
                        Combina la efectividad en entradas, bloqueos, intercepciones y recuperaciones. Las defensas con valores 
                        altos son muy efectivas neutralizando amenazas y recuperando la posesión.
                        
                        **Índice de Construcción**: Evalúa la contribución de la defensa a la fase ofensiva del equipo.
                        Combina la precisión en diferentes rangos de pase y la progresión del balón. Un valor alto indica 
                        una defensa moderna capaz de iniciar ataques y participar en la construcción del juego.
                        
                        **Índice de Posicionamiento**: Mide la ocupación inteligente del espacio y la capacidad para cubrir 
                        zonas críticas del campo. Refleja la presencia en zonas defensivas clave y la transición al mediocampo.
                        Las defensas con valores altos demuestran excelente lectura del juego y posicionamiento.
                        
                        **Índice de Seguridad**: Evalúa la fiabilidad defensiva considerando disciplina (tarjetas), porcentaje de
                        duelos ganados y recuperaciones. Las defensas con valores altos son confiables y raramente cometen errores
                        que comprometan al equipo.
                        
                        **Índice de Juego Aéreo**: Mide específicamente la dominancia en situaciones de balón aéreo, un aspecto
                        crucial para las defensas, especialmente centrales. Combina acciones defensivas en situaciones aéreas.
                        
                        **Índice de Impacto por Minuto**: Evalúa la eficiencia normalizando el rendimiento respecto al tiempo
                        jugado, especialmente útil para comparar titulares con suplentes.
                        
                        Estos índices ayudan a identificar si una defensa tiene un perfil más orientado a la destrucción de juego 
                        rival o a la construcción del propio, permitiendo evaluar tanto sus fortalezas como áreas de mejora.
                        """,
                        
                        'MF': """
                        ### Interpretación de Índices para Mediocampistas
                        
                        Los índices compuestos para mediocampistas evalúan seis facetas fundamentales del juego en el centro del campo:
                        
                        **Índice de Control**: Este índice refleja la capacidad de una mediocampista para gestionar el ritmo y la dirección 
                        del juego. Combina la precisión en pases de diferentes distancias con la progresión del balón y los pases al último 
                        tercio. Las mediocampistas con valores altos son excelentes organizadoras que dictan el tempo del partido.
                        
                        **Índice de Presión**: Evalúa la faceta defensiva de una mediocampista, midiendo su contribución a la recuperación 
                        del balón y la presión sobre rivales. Combina entradas, intercepciones y recuperaciones. Un valor alto indica una 
                        mediocampista que trabaja intensamente para reconquistar la posesión.
                        
                        **Índice de Creación**: Mide la capacidad para generar oportunidades ofensivas. Combina asistencias esperadas, 
                        pases clave y acciones que conducen a tiros y goles. Las mediocampistas con valores altos son grandes generadoras 
                        de juego ofensivo y catalizadoras de las oportunidades del equipo.
                        
                        **Índice de Versatilidad**: Evalúa el equilibrio entre contribuciones ofensivas y defensivas, identificando
                        mediocampistas completas que aportan en ambas fases del juego. Un valor alto indica jugadoras "box-to-box" que
                        participan activamente en todo el campo.
                        
                        **Índice de Progresión**: Mide la capacidad para hacer avanzar al equipo hacia zonas ofensivas, ya sea mediante
                        pases, conducciones o recibiendo en zonas avanzadas. Un valor alto identifica a jugadoras que rompen líneas y
                        hacen progresar el juego.
                        
                        **Índice de Impacto por Minuto**: Evalúa la eficiencia normalizando el rendimiento respecto al tiempo
                        jugado, permitiendo comparaciones justas entre jugadoras con diferente participación.
                        
                        Estos índices permiten clasificar a las mediocampistas según sus perfiles (organizadoras, destructoras, creadoras, 
                        box-to-box) y evaluar el equilibrio de sus habilidades en las diferentes facetas del juego.
                        """,
                        
                        'FW': """
                        ### Interpretación de Índices para Delanteras
                        
                        Los índices compuestos para delanteras evalúan seis aspectos esenciales del juego ofensivo:
                        
                        **Índice de Efectividad Ofensiva**: Este índice mide la capacidad goleadora y la eficiencia en la finalización. 
                        Combina goles marcados, conversión de oportunidades y rendimiento respecto a goles esperados. Un valor alto 
                        indica una delantera clínica que maximiza sus oportunidades de gol.
                        
                        **Índice de Creación**: Evalúa la contribución de la delantera a la generación de oportunidades para sus 
                        compañeras. Combina asistencias, asistencias esperadas y acciones de creación de tiros y goles. Las delanteras 
                        con valores altos no solo finalizan jugadas sino que también las crean.
                        
                        **Índice de Movimiento Ofensivo**: Mide el posicionamiento, desmarque y capacidad para ocupar espacios 
                        peligrosos. Analiza la presencia en zonas avanzadas, especialmente el área rival, y la progresión con balón. 
                        Un valor alto refleja una delantera con excelente inteligencia posicional y capacidad para encontrar espacios.
                        
                        **Índice de Presión Ofensiva**: Evalúa la contribución defensiva de la delantera en zonas avanzadas, midiendo
                        su participación en la presión alta y recuperaciones en campo rival. Un valor alto indica delanteras que
                        trabajan intensamente en la primera línea defensiva.
                        
                        **Índice de Autonomía**: Mide la capacidad de la delantera para generar peligro por sí misma, mediante regates,
                        conducciones y finalización. Un valor alto identifica a jugadoras que pueden desequilibrar y resolver
                        situaciones sin depender tanto del juego colectivo.
                        
                        **Índice de Impacto por Minuto**: Evalúa la eficiencia normalizando el rendimiento respecto al tiempo
                        jugado, especialmente relevante para comparar titulares y suplentes o jugadoras con lesiones.
                        
                        Estos índices ayudan a distinguir entre diferentes tipos de delanteras (definidoras, generadoras, combinativas, 
                        presionadoras) y proporcionan una visión integral de sus contribuciones ofensivas más allá de los simples goles marcados.
                        """
                    }
                    
                    # Mostrar la explicación correspondiente a la posición
                    if position in explicaciones:
                        st.markdown(explicaciones[position])
                    
                    # Información adicional sobre cómo se utilizan los índices
                    st.info("""
                    **¿Cómo utilizar estos índices en el análisis?**
                    
                    - **Visión integral**: Los índices combinan múltiples métricas en un solo valor, facilitando comparaciones holísticas
                    - **Identificación de perfiles**: Permiten categorizar jugadoras según sus fortalezas específicas
                    - **Áreas de mejora**: Diferencias significativas en ciertos índices pueden indicar aspectos a desarrollar
                    - **Complementariedad**: Útil para identificar jugadoras que complementan las fortalezas/debilidades de otras
                    - **Evolución temporal**: Monitorizando estos índices a lo largo del tiempo se puede evaluar el desarrollo de jugadoras
                    
                    Todos los índices están normalizados en una escala de 0-100 para facilitar la comparación, donde 100 representa 
                    el rendimiento óptimo teórico en esa categoría.
                    """)
                    
                else:
                    st.warning(f"No se pudieron calcular índices para la posición {position}. Verifica que existen métricas suficientes en los datos.")
    
    else:
        # Mensaje cuando no se ha realizado el análisis
        st.info("""
        👈 Selecciona una jugadora del Atlético de Madrid en el panel lateral y haz clic en "Analizar Similitudes" para comenzar.
        
        Este análisis te permitirá:
        1. Encontrar las 10 jugadoras más similares basadas en sus métricas de rendimiento
        2. Ver visualizaciones de clustering y análisis de componentes principales
        3. Comparar métricas específicas entre jugadoras
        """)

# En caso de que no se carguen los datos correctamente
else:
    st.error("""
    No se pudieron cargar los datos necesarios. 
    
    Verifica que los siguientes archivos existen en las rutas especificadas:
    - data/data_gold/df_keepers_gold_1.csv
    - data/data_gold/df_players_gold_1.csv
    - data/players_info/df_230_players_info.csv
    - data/teams_info/big/df_teams_info_global.csv
    - data/players_info/atm_pics.csv
    """)
    
    # Opción de cargar datos de ejemplo para pruebas
    if st.button("Cargar datos de ejemplo para demostración"):
        st.info("Esta función crearía datos sintéticos para demostrar la funcionalidad de la aplicación.")
        # Aquí podrías implementar la generación de datos sintéticos si fuera necesario
