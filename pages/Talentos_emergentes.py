import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
from utils import display_logo

display_logo(100)

# Configuración de estilos personalizados
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
    
    /* Estilo para tarjetas de jugadoras */
    .jugadora-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .jugadora-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Badge para mostrar la posición */
    .position-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 5px;
    }
    
    .position-gk {
        background-color: #ffccd5;
        color: #d90429;
    }
    
    .position-df {
        background-color: #cce3ff;
        color: #0466c8;
    }
    
    .position-mf {
        background-color: #d8f3dc;
        color: #2d6a4f;
    }
    
    .position-fw {
        background-color: #ffddd2;
        color: #e76f51;
    }
    
    /* Badge para el ranking */
    .ranking-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 1rem;
        font-weight: bold;
        background-color: #ffe66d;
        color: #333;
        margin-left: 10px;
    }
    
    /* Estilo para los filtros */
    .filtros-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Estilo para gráficos */
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Estilo para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("Talentos Emergentes")
st.markdown("### Descubre las jóvenes promesas del fútbol femenino")
st.write("Esta herramienta te permite identificar y analizar las jugadoras menores de 23 años con mayor potencial, basándose en métricas adaptadas por posición y normalizadas según su tiempo de juego.")

# Función para cargar datos (usando la misma función que en otras páginas)
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

# Diccionario de métricas por posición
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

# Definición de los índices de talento por posición
talent_indices = {
    'GK': {
        'Índice de Potencial Defensivo': {
            'metricas': ['Save%', 'CS%', 'PSxG-GA'],
            'pesos': [0.4, 0.3, 0.3],
            'min_threshold': {'MP': 5, 'Min': 270},  # Mínimo 5 partidos o 270 minutos
            'descripcion': 'Evalúa la capacidad fundamental para evitar goles, combinando porcentaje de paradas, porterías a cero y rendimiento vs. expectativa.'
        },
        'Índice de Distribución': {
            'metricas': ['Pass_Cmp_+40y%', 'Pass_AvgLen'],
            'pesos': [0.6, 0.4],
            'min_threshold': {'MP': 4, 'Min': 225},
            'descripcion': 'Mide la capacidad para distribuir el balón con precisión y rango de pase.'
        },
        'Índice de Juego Aéreo': {
            'metricas': ['Stp%', '#OPA/90', 'AvgDist'],
            'pesos': [0.4, 0.3, 0.3],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Evalúa la dominancia en el juego aéreo y la proactividad fuera del área.'
        },
        'Potencial General': {
            'metricas': ['Save%', 'PSxG-GA', 'Pass_Cmp_+40y%', 'Stp%'],
            'pesos': [0.3, 0.3, 0.2, 0.2],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Valoración global del potencial de la portera combinando aspectos defensivos y con balón.'
        }
    },
    'DF': {
        'Índice de Solidez Defensiva': {
            'metricas': ['Tkl%', 'Blocks', 'Int', 'Recov', 'Tkl/90'],
            'pesos': [0.3, 0.15, 0.2, 0.2, 0.15],
            'min_threshold': {'MP': 5, 'Min': 270},
            'descripcion': 'Evalúa la capacidad para interceptar ataques y recuperar balones.'
        },
        'Índice de Construcción': {
            'metricas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'PrgDist'],
            'pesos': [0.2, 0.3, 0.2, 0.3],
            'min_threshold': {'MP': 4, 'Min': 225},
            'descripcion': 'Mide la contribución al juego ofensivo desde posiciones defensivas.'
        },
        'Índice de Posicionamiento': {
            'metricas': ['touch_Def Pen', 'touch_Def 3rd', 'touch_Mid 3rd'],
            'pesos': [0.3, 0.4, 0.3],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Evalúa el posicionamiento y cobertura de espacios en campo.'
        },
        'Potencial General': {
            'metricas': ['Tkl+Int', 'Cmp%_long', 'Recov', 'PrgDist', 'Blocks'],
            'pesos': [0.25, 0.2, 0.2, 0.2, 0.15],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Valoración global del potencial de la defensa combinando aspectos defensivos y ofensivos.'
        }
    },
    'MF': {
        'Índice de Control': {
            'metricas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'PrgDist', 'pass_1/3'],
            'pesos': [0.15, 0.25, 0.2, 0.15, 0.25],
            'min_threshold': {'MP': 5, 'Min': 270},
            'descripcion': 'Mide la capacidad para controlar el juego y hacer progresar el balón.'
        },
        'Índice de Creación': {
            'metricas': ['xA', 'KP', 'SCA90', 'GCA90'],
            'pesos': [0.3, 0.2, 0.25, 0.25],
            'min_threshold': {'MP': 4, 'Min': 225},
            'descripcion': 'Evalúa la contribución a la creación de oportunidades de gol.'
        },
        'Índice de Presión': {
            'metricas': ['Tkl/90', 'Tkl%', 'Int', 'Recov'],
            'pesos': [0.3, 0.2, 0.25, 0.25],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Valora la contribución defensiva y capacidad de recuperación.'
        },
        'Potencial General': {
            'metricas': ['pass_1/3', 'Tkl+Int', 'xA', 'carries_PrgDist', 'SCA90'],
            'pesos': [0.2, 0.2, 0.2, 0.2, 0.2],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Valoración global del potencial de la mediocampista combinando aspectos de creación, control y defensa.'
        }
    },
    'FW': {
        'Índice de Definición': {
            'metricas': ['Gls', 'G/Sh', 'xG', 'G-xG', 'SoT/90'],
            'pesos': [0.25, 0.25, 0.2, 0.15, 0.15],
            'min_threshold': {'MP': 5, 'Min': 270},
            'descripcion': 'Evalúa la capacidad goleadora y eficiencia en la finalización.'
        },
        'Índice de Creación': {
            'metricas': ['Ast', 'xA', 'SCA90', 'GCA90', 'KP'],
            'pesos': [0.25, 0.2, 0.2, 0.2, 0.15],
            'min_threshold': {'MP': 4, 'Min': 225},
            'descripcion': 'Mide la contribución a la creación de ocasiones para compañeras.'
        },
        'Índice de Movimiento': {
            'metricas': ['touch_Att 3rd', 'touch_Att Pen', 'PrgR', 'TO_Succ%'],
            'pesos': [0.25, 0.3, 0.25, 0.2],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Evalúa el posicionamiento, desmarque y capacidad para ocupar espacios peligrosos.'
        },
        'Potencial General': {
            'metricas': ['G+A', 'SoT/90', 'touch_Att Pen', 'xG', 'PrgR'],
            'pesos': [0.25, 0.2, 0.2, 0.2, 0.15],
            'min_threshold': {'MP': 3, 'Min': 180},
            'descripcion': 'Valoración global del potencial ofensivo combinando definición, creación y movimiento.'
        }
    }
}

# Función para calcular los índices de talento
def calcular_indices_talento(df, edad_maxima=23):
    # Filtrar jugadoras menores de la edad máxima
    # Suponiendo que 'Born' contiene el año de nacimiento y estamos en 2025
    año_actual = 2025
    df_jovenes = df[df['Born'] >= (año_actual - edad_maxima)].copy()
    
    # Diccionario para almacenar los resultados
    resultados = {}
    
    # Para cada posición
    for posicion, indices in talent_indices.items():
        # Filtrar jugadoras de esta posición
        df_posicion = df_jovenes[df_jovenes['Posición Principal'] == posicion].copy()
        
        # Inicializar resultados para esta posición
        resultados[posicion] = {}
        
        # Para cada índice definido
        for nombre_indice, definicion in indices.items():
            metricas = definicion['metricas']
            pesos = definicion['pesos']
            min_threshold = definicion['min_threshold']
            
            # Verificar que las métricas existen y filtrar las jugadoras que cumplen con el umbral mínimo
            df_valido = df_posicion.copy()
            
            # Aplicar umbrales mínimos
            for metrica, valor in min_threshold.items():
                if metrica in df_valido.columns:
                    df_valido = df_valido[df_valido[metrica] >= valor]
            
            # Si no hay jugadoras que cumplan el criterio, continuar con el siguiente índice
            if df_valido.empty:
                continue
                
            # Inicializar columna para este índice
            resultados[posicion][nombre_indice] = {}
            
            # Calculamos máximos y mínimos para normalización
            max_vals = {}
            min_vals = {}
            
            for metrica in metricas:
                if metrica in df_valido.columns:
                    valores_metrica = df_valido[metrica].dropna()
                    if not valores_metrica.empty:
                        max_vals[metrica] = valores_metrica.max()
                        min_vals[metrica] = valores_metrica.min()
            
            # Calcular el índice para cada jugadora
            for _, row in df_valido.iterrows():
                jugadora = row['Player']
                squad = row['Squad']
                born = row['Born']
                mp = row['MP'] if 'MP' in row else 0
                mins = row['Min'] if 'Min' in row else 0
                
                # Normalizar y calcular el valor del índice
                valor_indice = 0
                valores_por_metrica = {}
                
                for i, metrica in enumerate(metricas):
                    if metrica in row and not pd.isna(row[metrica]) and metrica in max_vals and metrica in min_vals:
                        # Verificar si el rango es válido para evitar división por cero
                        if max_vals[metrica] > min_vals[metrica]:
                            # Normalizar valor entre 0 y 1
                            valor_norm = (row[metrica] - min_vals[metrica]) / (max_vals[metrica] - min_vals[metrica])
                            # Almacenar para detalles
                            valores_por_metrica[metrica] = valor_norm
                            # Acumular valor ponderado
                            valor_indice += valor_norm * pesos[i]
                        else:
                            # Si todos los valores son iguales, usar 0.5 como valor normalizado
                            valor_norm = 0.5
                            valores_por_metrica[metrica] = valor_norm
                            valor_indice += valor_norm * pesos[i]
                
                # Escalar a 0-100 para mejor interpretación
                valor_final = valor_indice * 100
                
                # Almacenar resultado con datos adicionales
                resultados[posicion][nombre_indice][jugadora] = {
                    'valor': valor_final,
                    'squad': squad,
                    'born': born,
                    'edad': año_actual - born,
                    'partidos': mp,
                    'minutos': mins,
                    'detalles': valores_por_metrica
                }
    
    return resultados

# Cargar datos
df_combined, df_players_info, df_teams_info, df_atm_photos = cargar_datos()

# Verificar si se cargaron los datos correctamente
if df_combined is not None and not df_combined.empty:
    # Sidebar para controles
    st.sidebar.title("Controles")
    
    # Selección de edad máxima
    edad_maxima = st.sidebar.slider("Edad máxima (años)", min_value=18, max_value=25, value=23, step=1)
    
    # Filtros de liga
    ligas_disponibles = ['Todas'] + sorted(df_combined['League'].dropna().unique().tolist())
    liga_seleccionada = st.sidebar.selectbox("Competición", ligas_disponibles)
    
    # Número de jugadoras a mostrar
    num_jugadoras = st.sidebar.slider("Número de jugadoras en el ranking", min_value=5, max_value=25, value=10, step=5)
    
    # Selección de posición
    posiciones = ['Todas', 'GK', 'DF', 'MF', 'FW']
    posicion_seleccionada = st.sidebar.selectbox("Posición", posiciones)
    
    # Selección de índice de talento
    indices_seleccionables = ['Potencial General']
    
    if posicion_seleccionada in talent_indices:
        indices_seleccionables = list(talent_indices[posicion_seleccionada].keys())
    elif posicion_seleccionada == 'Todas':
        # Para "Todas", usamos solo el índice general
        indices_seleccionables = ['Potencial General']
        
    indice_seleccionado = st.sidebar.selectbox("Índice de talento", indices_seleccionables)
    
    st.sidebar.markdown("---")
    st.sidebar.write("### Información")
    st.sidebar.info("""
    Los índices de talento están diseñados para identificar jugadoras jóvenes prometedoras incluso con pocos minutos.
    
    Cada índice normaliza las estadísticas por tiempo de juego y aplica umbrales mínimos para garantizar suficientes datos.
    
    La puntuación va de 0 a 100, donde valores más altos indican mayor potencial.
    """)
    
    # Añadir explicación del cálculo
    with st.sidebar.expander("¿Cómo se calculan los índices?"):
        st.write("""
        1. Se aplican **umbrales mínimos** de partidos/minutos para garantizar una muestra significativa.
        2. Se **normalizan** las métricas de 0 a 1 en comparación con las jugadoras de la misma posición.
        3. Se aplican **pesos específicos** según la importancia de cada métrica para el índice.
        4. Se escala el resultado final a una puntuación de 0-100 para facilitar la interpretación.
        
        Las métricas utilizadas varían según posición e índice, centrándose en aspectos clave del rendimiento.
        """)
    
    # Calcular los índices de talento
    with st.spinner("Calculando índices de talento..."):
        # Filtramos por liga si es necesario
        if liga_seleccionada != 'Todas':
            df_filtered = df_combined[df_combined['League'] == liga_seleccionada].copy()
        else:
            df_filtered = df_combined.copy()
            
        # Calculamos los índices
        resultados_indices = calcular_indices_talento(df_filtered, edad_maxima)
    
    # Crear pestañas para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["Ranking de Talentos", "Comparativa Detallada", "Comparativa con Equipo Propio", "Visualización por Edad"])
    
    with tab1:
        st.header("Ranking de Talentos Emergentes")
        
        # Preparar datos para el ranking según la selección
        jugadoras_ranking = []
        
        if posicion_seleccionada != 'Todas':
            # Si se seleccionó una posición específica
            if posicion_seleccionada in resultados_indices and indice_seleccionado in resultados_indices[posicion_seleccionada]:
                for jugadora, datos in resultados_indices[posicion_seleccionada][indice_seleccionado].items():
                    jugadoras_ranking.append({
                        'Player': jugadora,
                        'Posición': posicion_seleccionada,
                        'Valor': datos['valor'],
                        'Squad': datos['squad'],
                        'Edad': datos['edad'],
                        'Partidos': datos['partidos'],
                        'Minutos': datos['minutos']
                    })
        else:
            # Si se seleccionaron todas las posiciones, combinamos los resultados
            for posicion in talent_indices.keys():
                if posicion in resultados_indices and 'Potencial General' in resultados_indices[posicion]:
                    for jugadora, datos in resultados_indices[posicion]['Potencial General'].items():
                        jugadoras_ranking.append({
                            'Player': jugadora,
                            'Posición': posicion,
                            'Valor': datos['valor'],
                            'Squad': datos['squad'], 
                            'Edad': datos['edad'],
                            'Partidos': datos['partidos'],
                            'Minutos': datos['minutos']
                        })
        
        # Ordenar por valor del índice
        jugadoras_ranking_sorted = sorted(jugadoras_ranking, key=lambda x: x['Valor'], reverse=True)
        
        # Limitar al número seleccionado
        jugadoras_ranking_top = jugadoras_ranking_sorted[:num_jugadoras]
        
        if not jugadoras_ranking_top:
            st.warning(f"No se encontraron suficientes jugadoras que cumplan con los criterios mínimos para el índice seleccionado. Prueba con otros filtros o reduce los umbrales mínimos.")
        else:
            # Mostrar métricas generales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de talentos", f"{len(jugadoras_ranking)} jugadoras")
            with col2:
                st.metric("Edad promedio", f"{sum(j['Edad'] for j in jugadoras_ranking) / len(jugadoras_ranking):.1f} años")
            with col3:
                if posicion_seleccionada != 'Todas':
                    mejor_equipo = max(set([j['Squad'] for j in jugadoras_ranking]), 
                                      key=lambda x: sum(1 for j in jugadoras_ranking if j['Squad'] == x))
                    st.metric("Equipo con más talentos", mejor_equipo)
                else:
                    mejor_posicion = max(set([j['Posición'] for j in jugadoras_ranking]), 
                                        key=lambda x: sum(1 for j in jugadoras_ranking if j['Posición'] == x))
                    st.metric("Posición con más talentos", mejor_posicion)
            
            # Mostrar información detallada sobre el índice seleccionado
            if posicion_seleccionada in talent_indices and indice_seleccionado in talent_indices[posicion_seleccionada]:
                st.info(f"**{indice_seleccionado}**: {talent_indices[posicion_seleccionada][indice_seleccionado]['descripcion']}")
                
                # Mostrar métricas utilizadas
                metricas_utilizadas = talent_indices[posicion_seleccionada][indice_seleccionado]['metricas']
                pesos = talent_indices[posicion_seleccionada][indice_seleccionado]['pesos']
                
                # Crear gráfico de distribución de pesos
                fig, ax = plt.subplots(figsize=(5, 2))
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(metricas_utilizadas)))
                bars = ax.barh(
                    [metric_display_names.get(m, m) for m in metricas_utilizadas], 
                    pesos, 
                    color=colors
                )
                
                # Añadir valores
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                            va='center', fontsize=9)
                
                ax.set_title(f'Ponderación de métricas para {indice_seleccionado}')
                ax.set_xlabel('Peso')
                ax.set_xlim(0, max(pesos) + 0.1)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Crear tarjetas para las jugadoras del ranking
            st.subheader("Top Jugadoras")
            
            # Crear grid de tarjetas
            num_cols = 2  # Número de columnas en el grid
            rows = [st.columns(num_cols) for _ in range((len(jugadoras_ranking_top) + num_cols - 1) // num_cols)]
            
            for i, jugadora in enumerate(jugadoras_ranking_top):
                col = rows[i // num_cols][i % num_cols]
                
                with col:
                    # Definir clase CSS para la posición
                    position_class = f"position-{jugadora['Posición'].lower()}"
                    
                    # Construir el HTML de la tarjeta
                    html = f"""
                    <div class="jugadora-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h3 style="margin: 0;">{i+1}. {jugadora['Player']}</h3>
                            <span class="position-badge {position_class}">{jugadora['Posición']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <div>{jugadora['Squad']}</div>
                            <div style="display: flex; align-items: center;">
                                <strong style="font-size: 1.2rem; margin-right: 5px;">{jugadora['Valor']:.1f}</strong>
                                <span style="color: #666; font-size: 0.8rem;">/100</span>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div style="color: #666;">Edad:</div>
                            <div><strong>{jugadora['Edad']}</strong> años</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div style="color: #666;">Partidos:</div>
                            <div><strong>{jugadora['Partidos']}</strong></div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div style="color: #666;">Minutos:</div>
                            <div><strong>{jugadora['Minutos']}</strong></div>
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
            
            # Mostrar tabla completa
            with st.expander("Ver tabla completa"):
                df_ranking = pd.DataFrame(jugadoras_ranking_top)
                if not df_ranking.empty:
                    # Cambiar nombres de columnas para la tabla
                    df_ranking.columns = ['Jugadora', 'Posición', 'Puntuación', 'Equipo', 'Edad', 'Partidos', 'Minutos']
                    # Formatear la puntuación para mostrar solo un decimal
                    df_ranking['Puntuación'] = df_ranking['Puntuación'].round(1)
                    st.dataframe(df_ranking, use_container_width=True)
    
    with tab2:
        st.header("Comparativa Detallada")
        
        # Seleccionar jugadoras para comparar (máximo 5)
        if jugadoras_ranking:
            opciones_jugadoras = [j['Player'] for j in jugadoras_ranking_sorted[:30]]  # Limitamos a top 30 para selección
            
            jugadoras_seleccionadas = st.multiselect(
                "Selecciona jugadoras para comparar (máximo 5):",
                opciones_jugadoras,
                default=opciones_jugadoras[:min(3, len(opciones_jugadoras))]
            )
            
            if len(jugadoras_seleccionadas) > 5:
                st.warning("Has seleccionado más de 5 jugadoras. Solo se mostrarán las primeras 5.")
                jugadoras_seleccionadas = jugadoras_seleccionadas[:5]
            
            if jugadoras_seleccionadas:
                # Obtener datos detallados de las jugadoras seleccionadas
                detalles_jugadoras = []
                
                for jugadora_nombre in jugadoras_seleccionadas:
                    # Buscar la posición de la jugadora
                    for jugadora in jugadoras_ranking:
                        if jugadora['Player'] == jugadora_nombre:
                            posicion = jugadora['Posición']
                            break
                    else:
                        continue  # Si no se encuentra la jugadora, continuar con la siguiente
                    
                    # Obtener todos los índices para esta posición y jugadora
                    detalles = {'Nombre': jugadora_nombre, 'Posición': posicion, 'Valores': {}}
                    
                    for nombre_indice in talent_indices[posicion]:
                        if nombre_indice in resultados_indices[posicion] and jugadora_nombre in resultados_indices[posicion][nombre_indice]:
                            detalles['Valores'][nombre_indice] = resultados_indices[posicion][nombre_indice][jugadora_nombre]['valor']
                            if nombre_indice == 'Potencial General':
                                detalles['Squad'] = resultados_indices[posicion][nombre_indice][jugadora_nombre]['squad']
                                detalles['Edad'] = resultados_indices[posicion][nombre_indice][jugadora_nombre]['edad']
                    
                    if detalles['Valores']:  # Solo añadir si hay valores
                        detalles_jugadoras.append(detalles)
                
                if detalles_jugadoras:
                    # Crear gráfico de radar
                    # Primero, determinar los índices comunes entre todas las jugadoras seleccionadas
                    indices_comunes = set()
                    for i, detalle in enumerate(detalles_jugadoras):
                        if i == 0:
                            indices_comunes = set(detalle['Valores'].keys())
                        else:
                            indices_comunes &= set(detalle['Valores'].keys())
                    
                    if 'Potencial General' in indices_comunes:
                        # Mostrar información básica de las jugadoras
                        columns = st.columns(len(detalles_jugadoras))
                        for i, detalle in enumerate(detalles_jugadoras):
                            with columns[i]:
                                st.markdown(f"### {detalle['Nombre']}")
                                st.markdown(f"**Equipo:** {detalle['Squad']}")
                                st.markdown(f"**Posición:** {detalle['Posición']}")
                                st.markdown(f"**Edad:** {detalle['Edad']} años")
                                st.markdown(f"**Potencial General:** {detalle['Valores']['Potencial General']:.1f}/100")
                        
                        # Crear gráfico de barras para comparar todos los índices
                        st.subheader("Comparativa de Índices")
                        
                        # Preparar datos para el gráfico
                        indices = list(indices_comunes)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        bar_width = 0.8 / len(detalles_jugadoras)
                        positions = np.arange(len(indices))
                        
                        for i, detalle in enumerate(detalles_jugadoras):
                            values = [detalle['Valores'].get(indice, 0) for indice in indices]
                            offset = (i - len(detalles_jugadoras) / 2 + 0.5) * bar_width
                            bars = ax.bar(positions + offset, values, bar_width, 
                                         label=detalle['Nombre'],
                                         alpha=0.8)
                            
                            # Añadir valores sobre las barras
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)
                        
                        ax.set_title('Comparativa de Índices de Talento')
                        ax.set_xticks(positions)
                        ax.set_xticklabels(indices, rotation=45, ha='right')
                        ax.set_ylim(0, 105)  # Para dejar espacio para las etiquetas
                        ax.set_ylabel('Puntuación (0-100)')
                        ax.legend(loc='upper right')
                        ax.grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Si son de la misma posición, mostrar detalles de métricas
                        posiciones_unicas = set(detalle['Posición'] for detalle in detalles_jugadoras)
                        if len(posiciones_unicas) == 1:
                            posicion = list(posiciones_unicas)[0]
                            st.subheader(f"Métricas Clave para {posicion}")
                            
                            # Obtener métricas clave para esta posición
                            if 'Potencial General' in talent_indices[posicion]:
                                metricas_clave = talent_indices[posicion]['Potencial General']['metricas']
                                
                                # Obtener valores para cada jugadora
                                datos_metricas = {}
                                for metrica in metricas_clave:
                                    datos_metricas[metrica] = []
                                    
                                jugadoras_nombres = []
                                for detalle in detalles_jugadoras:
                                    jugadora_nombre = detalle['Nombre']
                                    jugadoras_nombres.append(jugadora_nombre)
                                    
                                    # Obtener datos de la jugadora
                                    row = df_combined[df_combined['Player'] == jugadora_nombre]
                                    if not row.empty:
                                        for metrica in metricas_clave:
                                            if metrica in row.columns:
                                                datos_metricas[metrica].append(row[metrica].iloc[0])
                                            else:
                                                datos_metricas[metrica].append(0)
                                
                                # Crear gráfico para cada métrica
                                for metrica in metricas_clave:
                                    if all(pd.notna(valor) for valor in datos_metricas[metrica]):
                                        fig, ax = plt.subplots(figsize=(10, 4))
                                        bars = ax.bar(jugadoras_nombres, datos_metricas[metrica], 
                                                     alpha=0.7, color='skyblue')
                                        
                                        # Añadir valores
                                        for bar in bars:
                                            height = bar.get_height()
                                            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                                                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                                        
                                        ax.set_title(f'{metric_display_names.get(metrica, metrica)}')
                                        ax.grid(axis='y', alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                        else:
                            st.info("Para ver comparativa detallada de métricas, selecciona jugadoras de la misma posición.")
                    else:
                        st.warning("Las jugadoras seleccionadas no comparten índices comunes para una comparación directa.")
                else:
                    st.warning("No se encontraron datos detallados para las jugadoras seleccionadas.")
            else:
                st.info("Selecciona al menos una jugadora para ver su análisis detallado.")
        else:
            st.warning("No hay jugadoras disponibles para comparar con los filtros actuales.")
    
    with tab4:
        st.header("Distribución por Edad")
        
        # Crear análisis de distribución de talento por edad
        if jugadoras_ranking:
            # Agrupar por edad
            edades = {}
            for jugadora in jugadoras_ranking:
                edad = jugadora['Edad']
                if edad not in edades:
                    edades[edad] = []
                edades[edad].append(jugadora)
            
            # Calcular promedio de potencial por edad
            promedios_edad = {}
            for edad, jugadoras in edades.items():
                promedios_edad[edad] = sum(j['Valor'] for j in jugadoras) / len(jugadoras)
            
            # Crear gráfico de línea para mostrar evolución del potencial por edad
            edades_ordenadas = sorted(promedios_edad.keys())
            valores_promedio = [promedios_edad[edad] for edad in edades_ordenadas]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(edades_ordenadas, valores_promedio, 'o-', linewidth=2, markersize=8)
            
            # Añadir etiquetas con el número de jugadoras
            for i, edad in enumerate(edades_ordenadas):
                ax.annotate(f"{len(edades[edad])} jugadoras", 
                           (edad, valores_promedio[i]),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center')
            
            ax.set_title('Promedio de Potencial por Edad')
            ax.set_xlabel('Edad')
            ax.set_ylabel('Puntuación Promedio (0-100)')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(edades_ordenadas)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Mostrar distribución por posición y edad
            st.subheader("Distribución por Posición y Edad")
            
            # Agrupar por posición y edad
            posicion_edad = {}
            for jugadora in jugadoras_ranking:
                posicion = jugadora['Posición']
                edad = jugadora['Edad']
                
                if posicion not in posicion_edad:
                    posicion_edad[posicion] = {}
                
                if edad not in posicion_edad[posicion]:
                    posicion_edad[posicion][edad] = []
                
                posicion_edad[posicion][edad].append(jugadora)
            
            # Crear gráfico de burbujas
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = {'GK': 'red', 'DF': 'blue', 'MF': 'green', 'FW': 'orange'}
            
            for posicion, datos_edad in posicion_edad.items():
                x = []  # Edades
                y = []  # Promedios de potencial
                sizes = []  # Tamaños de las burbujas (número de jugadoras)
                
                for edad, jugadoras in datos_edad.items():
                    x.append(edad)
                    promedio = sum(j['Valor'] for j in jugadoras) / len(jugadoras)
                    y.append(promedio)
                    sizes.append(len(jugadoras) * 50)  # Escalar para mejor visualización
                
                ax.scatter(x, y, s=sizes, alpha=0.6, label=posicion, color=colors.get(posicion, 'gray'))
                
                # Añadir etiquetas
                for i in range(len(x)):
                    ax.annotate(f"{len(datos_edad[x[i]])}", (x[i], y[i]), ha='center', va='center', fontweight='bold')
            
            ax.set_title('Distribución de Talento por Posición y Edad')
            ax.set_xlabel('Edad')
            ax.set_ylabel('Puntuación Promedio (0-100)')
            ax.grid(True, alpha=0.3)
            ax.legend(title="Posición")
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla de distribución
            st.subheader("Resumen de Distribución")
            
            # Crear datos para la tabla
            resumen_datos = []
            
            for posicion in sorted(posicion_edad.keys()):
                total_jugadoras = sum(len(jugadoras) for jugadoras in posicion_edad[posicion].values())
                promedio_general = sum(sum(j['Valor'] for j in jugadoras) for jugadoras in posicion_edad[posicion].values()) / total_jugadoras
                mejor_edad = max(posicion_edad[posicion].keys(), key=lambda edad: sum(j['Valor'] for j in posicion_edad[posicion][edad]) / len(posicion_edad[posicion][edad]))
                
                resumen_datos.append({
                    'Posición': posicion,
                    'Total Jugadoras': total_jugadoras,
                    'Promedio Potencial': f"{promedio_general:.1f}",
                    'Edad con Mayor Potencial': mejor_edad,
                    'Mejor Puntuación': f"{max(j['Valor'] for edad in posicion_edad[posicion] for j in posicion_edad[posicion][edad]):.1f}"
                })
            
            # Mostrar tabla
            st.table(pd.DataFrame(resumen_datos))
        else:
            st.warning("No hay suficientes datos para crear visualizaciones de distribución por edad.")
            
    with tab3:
        st.header("Comparativa con Jugadoras del Atlético de Madrid")
        
        if jugadoras_ranking:
            # Sección para seleccionar una jugadora joven del ranking
            st.subheader("Selecciona una joven promesa para comparar")
            
            # Preparar opciones para el selectbox (top 20 del ranking)
            opciones_joven = [f"{j['Player']} ({j['Posición']}, {j['Squad']})" for j in jugadoras_ranking_sorted[:20]]
            seleccion_joven = st.selectbox("Joven promesa:", opciones_joven)
            
            if seleccion_joven:
                # Extraer nombre de la jugadora seleccionada
                nombre_joven = seleccion_joven.split(" (")[0]
                
                # Buscar datos completos de la jugadora seleccionada
                datos_joven = None
                posicion_joven = None
                for j in jugadoras_ranking:
                    if j['Player'] == nombre_joven:
                        datos_joven = j
                        posicion_joven = j['Posición']
                        break
                
                if datos_joven and posicion_joven:
                    # Filtrar jugadoras del Atlético de Madrid con la misma posición
                    jugadoras_atm = df_combined[(df_combined['Squad'] == 'Atlético de Madrid') & 
                                              (df_combined['Posición Principal'] == posicion_joven)]
                    
                    if not jugadoras_atm.empty:
                        st.subheader(f"Jugadoras del Atlético de Madrid ({posicion_joven})")
                        
                        # Seleccionar jugadora del Atlético
                        opciones_atm = sorted(jugadoras_atm['Player'].unique().tolist())
                        seleccion_atm = st.selectbox("Selecciona una jugadora del Atlético de Madrid:", opciones_atm)
                        
                        if seleccion_atm:
                            # Crear visualización de comparación
                            st.subheader(f"Comparativa: {nombre_joven} vs {seleccion_atm}")
                            
                            # Obtener los datos de la jugadora del Atlético
                            datos_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm].iloc[0]
                            
                            # Mostrar información básica en columnas
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"### {nombre_joven}")
                                st.markdown(f"**Equipo:** {datos_joven['Squad']}")
                                st.markdown(f"**Edad:** {datos_joven['Edad']} años")
                                st.markdown(f"**Potencial:** {datos_joven['Valor']:.1f}/100")
                                st.markdown(f"**Partidos:** {datos_joven['Partidos']}")
                                st.markdown(f"**Minutos:** {datos_joven['Minutos']}")
                                
                                # Añadir foto si está disponible (para ATM o jugadoras conocidas)
                                if datos_joven['Squad'] == 'Atlético de Madrid' and df_atm_photos is not None:
                                    try:
                                        foto_joven = df_atm_photos[df_atm_photos['Player'] == nombre_joven]
                                        if not foto_joven.empty and 'url_photo' in foto_joven.columns:
                                            st.image(foto_joven['url_photo'].iloc[0], width=150)
                                    except:
                                        pass
                                
                            with col2:
                                st.markdown(f"### {seleccion_atm}")
                                st.markdown("**Equipo:** Atlético de Madrid")
                                
                                # Información de la jugadora del ATM
                                if 'Born' in datos_atm and pd.notna(datos_atm['Born']):
                                    st.markdown(f"**Edad:** {2025 - datos_atm['Born']} años")
                                
                                if 'MP' in datos_atm and pd.notna(datos_atm['MP']):
                                    st.markdown(f"**Partidos:** {datos_atm['MP']}")
                                    
                                if 'Min' in datos_atm and pd.notna(datos_atm['Min']):
                                    st.markdown(f"**Minutos:** {datos_atm['Min']}")
                                    
                                # Añadir foto si está disponible
                                if df_atm_photos is not None:
                                    try:
                                        foto_atm = df_atm_photos[df_atm_photos['Player'] == seleccion_atm]
                                        if not foto_atm.empty and 'url_photo' in foto_atm.columns:
                                            st.image(foto_atm['url_photo'].iloc[0], width=150)
                                    except:
                                        pass
                            
                            # Obtener métricas relevantes para la posición
                            if posicion_joven in talent_indices and 'Potencial General' in talent_indices[posicion_joven]:
                                metricas_clave = talent_indices[posicion_joven]['Potencial General']['metricas']
                                
                                # Crear gráfico de comparación para cada métrica
                                st.markdown("### Comparativa de Métricas Clave")
                                
                                # Obtener datos de joven promesa
                                datos_joven_completos = df_combined[df_combined['Player'] == nombre_joven]
                                
                                if not datos_joven_completos.empty and not jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm].empty:
                                    # Organizar los gráficos en filas de 3 columnas para ahorrar espacio
                                    metricas_para_mostrar = [m for m in metricas_clave if m in datos_joven_completos.columns and m in jugadoras_atm.columns]
                                    num_cols = 3
                                    num_filas = (len(metricas_para_mostrar) + num_cols - 1) // num_cols
                                    
                                    for fila in range(num_filas):
                                        cols = st.columns(num_cols)
                                        for i in range(num_cols):
                                            idx = fila * num_cols + i
                                            if idx < len(metricas_para_mostrar):
                                                metrica = metricas_para_mostrar[idx]
                                                with cols[i]:
                                                    valor_joven = datos_joven_completos[metrica].iloc[0]
                                                    valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                                    
                                                    if pd.notna(valor_joven) and pd.notna(valor_atm):
                                                        # Crear gráfico de barras lado a lado (versión más pequeña)
                                                        fig, ax = plt.subplots(figsize=(5, 3))
                                                        
                                                        # Datos para el gráfico
                                                        nombres = [nombre_joven[:10]+"..." if len(nombre_joven)>10 else nombre_joven, 
                                                                 seleccion_atm[:10]+"..." if len(seleccion_atm)>10 else seleccion_atm]
                                                        valores = [valor_joven, valor_atm]
                                                        colores = ['#ff9f1c', '#e71d36']  # Joven vs ATM
                                                        
                                                        # Crear barras
                                                        bars = ax.bar(nombres, valores, color=colores, width=0.6)
                                                        
                                                        # Añadir etiquetas
                                                        for bar in bars:
                                                            height = bar.get_height()
                                                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(valores),
                                                                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
                                                        
                                                        # Configurar gráfico
                                                        ax.set_title(f'{metric_display_names.get(metrica, metrica)}', fontsize=10)
                                                        ax.tick_params(axis='x', labelsize=8)
                                                        ax.tick_params(axis='y', labelsize=8)
                                                        ax.grid(axis='y', alpha=0.3)
                                                        
                                                        # Ajustar límites para valores positivos y negativos
                                                        if min(valores) < 0:
                                                            ax.set_ylim(min(valores) * 1.1, max(valores) * 1.1)
                                                        else:
                                                            ax.set_ylim(0, max(valores) * 1.1)
                                                        
                                                        plt.tight_layout()
                                                        st.pyplot(fig)
                                
                                # Crear gráfico radar con los índices de talento
                                st.markdown("### Comparativa de Índices de Talento")
                                
                                # Verificar si tenemos índices calculados para ambas jugadoras
                                indices_joven = {}
                                
                                # Obtener índices para la joven promesa
                                if posicion_joven in resultados_indices:
                                    for nombre_indice in resultados_indices[posicion_joven]:
                                        if nombre_joven in resultados_indices[posicion_joven][nombre_indice]:
                                            indices_joven[nombre_indice] = resultados_indices[posicion_joven][nombre_indice][nombre_joven]['valor']
                                
                                # Si tenemos índices para la joven, calculamos también para la jugadora ATM
                                if indices_joven:
                                    # Calcular índices para la jugadora del Atlético
                                    indices_atm = {}
                                    
                                    # Para cada índice, calculamos el valor para la jugadora del ATM
                                    for nombre_indice, definicion in talent_indices[posicion_joven].items():
                                        metricas = definicion['metricas']
                                        pesos = definicion['pesos']
                                        
                                        # Verificar si se cumplen los umbrales mínimos
                                        datos_atm_row = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm].iloc[0]
                                        cumple_umbrales = True
                                        
                                        for metrica, valor in definicion['min_threshold'].items():
                                            if metrica in datos_atm_row and datos_atm_row[metrica] < valor:
                                                cumple_umbrales = False
                                                break
                                        
                                        if not cumple_umbrales:
                                            continue
                                        
                                        # Calcular el índice
                                        valor_indice = 0
                                        valores_por_metrica = {}
                                        
                                        # Obtener máximos y mínimos para normalización
                                        max_vals = {}
                                        min_vals = {}
                                        
                                        for metrica in metricas:
                                            if metrica in df_combined.columns:
                                                # Filtrar por posición para obtener rango comparable
                                                valores_metrica = df_combined[df_combined['Posición Principal'] == posicion_joven][metrica].dropna()
                                                if not valores_metrica.empty:
                                                    max_vals[metrica] = valores_metrica.max()
                                                    min_vals[metrica] = valores_metrica.min()
                                        
                                        # Normalizar y calcular
                                        for i, metrica in enumerate(metricas):
                                            if (metrica in datos_atm_row and 
                                                not pd.isna(datos_atm_row[metrica]) and 
                                                metrica in max_vals and 
                                                metrica in min_vals and 
                                                max_vals[metrica] > min_vals[metrica]):
                                                
                                                # Normalizar entre 0 y 1
                                                valor_norm = (datos_atm_row[metrica] - min_vals[metrica]) / (max_vals[metrica] - min_vals[metrica])
                                                valores_por_metrica[metrica] = valor_norm
                                                valor_indice += valor_norm * pesos[i]
                                            elif metrica in pesos:
                                                # Si no hay rango válido, usar 0.5
                                                valor_norm = 0.5
                                                valores_por_metrica[metrica] = valor_norm
                                                valor_indice += valor_norm * pesos[i]
                                        
                                        # Escalar a 0-100
                                        indices_atm[nombre_indice] = valor_indice * 100
                                    
                                    # Crear gráfico radar con los índices
                                    if indices_atm:
                                        # Obtener los índices comunes
                                        indices_comunes = set(indices_joven.keys()) & set(indices_atm.keys())
                                        
                                        if indices_comunes:
                                            # Convertir a lista ordenada
                                            indices_comunes = sorted(list(indices_comunes))
                                            
                                            # Crear figura para el radar
                                            fig = plt.figure(figsize=(8, 8))
                                            ax = fig.add_subplot(111, polar=True)
                                            
                                            # Extraer los valores
                                            valores_joven = [indices_joven[indice]/100 for indice in indices_comunes]  # Normalizar a 0-1
                                            valores_atm = [indices_atm[indice]/100 for indice in indices_comunes]
                                            
                                            # Configurar ángulos
                                            angulos = np.linspace(0, 2*np.pi, len(indices_comunes), endpoint=False).tolist()
                                            
                                            # Cerrar el círculo
                                            valores_joven = np.concatenate((valores_joven, [valores_joven[0]]))
                                            valores_atm = np.concatenate((valores_atm, [valores_atm[0]]))
                                            angulos += [angulos[0]]
                                            
                                            # Dibujar líneas
                                            ax.plot(angulos, valores_joven, 'o-', linewidth=2, color='#ff9f1c', label=nombre_joven)
                                            ax.fill(angulos, valores_joven, alpha=0.1, color='#ff9f1c')
                                            
                                            ax.plot(angulos, valores_atm, 'o-', linewidth=2, color='#e71d36', label=seleccion_atm)
                                            ax.fill(angulos, valores_atm, alpha=0.1, color='#e71d36')
                                            
                                            # Añadir etiquetas
                                            plt.xticks(angulos[:-1], indices_comunes, size=9)
                                            
                                            # Configurar escala
                                            ax.set_rlabel_position(0)
                                            plt.yticks([0.2, 0.4, 0.6, 0.8], ["20", "40", "60", "80"], color="grey", size=8)
                                            plt.ylim(0, 1)
                                            
                                            # Leyenda y título
                                            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                                            plt.title(f'Comparativa de Índices de Talento', size=14)
                                            
                                            st.pyplot(fig)
                                            
                                            # Tabla comparativa de índices
                                            data_tabla = {'Índice': indices_comunes,
                                                         nombre_joven: [indices_joven[i] for i in indices_comunes],
                                                         seleccion_atm: [indices_atm[i] for i in indices_comunes],
                                                         'Diferencia': [indices_joven[i] - indices_atm[i] for i in indices_comunes]}
                                            
                                            df_indices = pd.DataFrame(data_tabla)
                                            st.dataframe(df_indices.style.format({nombre_joven: '{:.1f}', 
                                                                                seleccion_atm: '{:.1f}', 
                                                                                'Diferencia': '{:.1f}'}))
                                        else:
                                            st.warning("No hay índices comunes calculados para ambas jugadoras.")
                                    else:
                                        st.warning("No se pudieron calcular índices para la jugadora del Atlético de Madrid.")
                                else:
                                    st.warning("No se encontraron índices calculados para la joven promesa seleccionada.")
                                
                                # Crear gráfico radar con todas las métricas disponibles
                                st.markdown("### Perfil Completo (Radar)")
                                
                                # Filtrar métricas numéricas disponibles para ambas jugadoras
                                metricas_radar = []
                                for metrica in position_metrics[posicion_joven]:
                                    if (metrica in datos_joven_completos.columns and 
                                        metrica in jugadoras_atm.columns and
                                        metrica not in ['Player', 'Squad', 'Born'] and
                                        pd.api.types.is_numeric_dtype(df_combined[metrica])):
                                        
                                        # Verificar que ambos valores están disponibles
                                        valor_joven = datos_joven_completos[metrica].iloc[0]
                                        valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                        
                                        if pd.notna(valor_joven) and pd.notna(valor_atm):
                                            metricas_radar.append(metrica)
                                
                                # Limitar a 12 métricas para mejor visualización
                                if len(metricas_radar) > 12:
                                    # Priorizar métricas de Potencial General
                                    metricas_potencial = talent_indices[posicion_joven]['Potencial General']['metricas']
                                    metricas_prioritarias = [m for m in metricas_radar if m in metricas_potencial]
                                    metricas_restantes = [m for m in metricas_radar if m not in metricas_potencial]
                                    
                                    # Completar hasta 12 con otras métricas
                                    metricas_radar = metricas_prioritarias + metricas_restantes[:12-len(metricas_prioritarias)]
                                
                                if len(metricas_radar) >= 3:  # Mínimo necesario para un radar
                                    # Crear figura para el radar
                                    fig = plt.figure(figsize=(10, 10))
                                    ax = fig.add_subplot(111, polar=True)
                                    
                                    # Extraer los valores
                                    valores_joven = []
                                    valores_atm = []
                                    
                                    # Obtener valores máximos para normalización
                                    max_valores = {}
                                    for metrica in metricas_radar:
                                        # Obtener máximo considerando solo jugadoras de esa posición
                                        pos_df = df_combined[df_combined['Posición Principal'] == posicion_joven]
                                        max_valores[metrica] = pos_df[metrica].max()
                                        
                                        # Si el máximo es 0 o NaN, usar 1 para evitar división por cero
                                        if pd.isna(max_valores[metrica]) or max_valores[metrica] == 0:
                                            max_valores[metrica] = 1
                                    
                                    # Normalizar valores
                                    for metrica in metricas_radar:
                                        valor_joven = datos_joven_completos[metrica].iloc[0]
                                        valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                        
                                        # Normalizar entre 0 y 1
                                        valores_joven.append(valor_joven / max_valores[metrica])
                                        valores_atm.append(valor_atm / max_valores[metrica])
                                    
                                    # Configurar ángulos
                                    angulos = np.linspace(0, 2*np.pi, len(metricas_radar), endpoint=False).tolist()
                                    
                                    # Cerrar el círculo
                                    valores_joven = np.concatenate((valores_joven, [valores_joven[0]]))
                                    valores_atm = np.concatenate((valores_atm, [valores_atm[0]]))
                                    angulos += [angulos[0]]
                                    
                                    # Dibujar líneas
                                    ax.plot(angulos, valores_joven, 'o-', linewidth=2, color='#ff9f1c', label=nombre_joven)
                                    ax.fill(angulos, valores_joven, alpha=0.1, color='#ff9f1c')
                                    
                                    ax.plot(angulos, valores_atm, 'o-', linewidth=2, color='#e71d36', label=seleccion_atm)
                                    ax.fill(angulos, valores_atm, alpha=0.1, color='#e71d36')
                                    
                                    # Añadir etiquetas
                                    plt.xticks(angulos[:-1], [metric_display_names.get(m, m) for m in metricas_radar], size=9)
                                    
                                    # Configurar escala
                                    ax.set_rlabel_position(0)
                                    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
                                    plt.ylim(0, 1)
                                    
                                    # Leyenda y título
                                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                                    plt.title(f'Comparativa de perfil completo: {nombre_joven} vs {seleccion_atm}', size=14)
                                    
                                    st.pyplot(fig)
                                    
                                    # Añadir explicación de normalización
                                    st.info("""
                                    **Nota sobre la normalización:**
                                    En el gráfico radar, todas las métricas se normalizan respecto al valor máximo entre todas las jugadoras 
                                    de la misma posición, lo que permite comparar distintas métricas en una misma escala.
                                    """)
                                else:
                                    st.warning("No hay suficientes métricas comparables para generar el gráfico radar.")
                                
                                # Sección para análisis de fortalezas y debilidades
                                st.markdown("### Análisis Comparativo")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"#### Fortalezas de {nombre_joven} vs {seleccion_atm}")
                                    
                                    # Identificar métricas donde la joven supera a la jugadora ATM
                                    metricas_superiores = []
                                    for metrica in metricas_radar:
                                        valor_joven = datos_joven_completos[metrica].iloc[0]
                                        valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                        
                                        if pd.notna(valor_joven) and pd.notna(valor_atm) and valor_joven > valor_atm:
                                            # Calcular la diferencia porcentual
                                            if valor_atm != 0:
                                                diff_pct = ((valor_joven - valor_atm) / valor_atm) * 100
                                                metricas_superiores.append((metrica, valor_joven, valor_atm, diff_pct))
                                    
                                    # Ordenar por diferencia porcentual
                                    metricas_superiores.sort(key=lambda x: x[3], reverse=True)
                                    
                                    # Mostrar las top 5 fortalezas
                                    for metrica, val_joven, val_atm, diff_pct in metricas_superiores[:5]:
                                        nombre_metrica = metric_display_names.get(metrica, metrica)
                                        st.markdown(f"**{nombre_metrica}**: {val_joven:.2f} vs {val_atm:.2f} (+{diff_pct:.1f}%)")
                                
                                with col2:
                                    st.markdown(f"#### Áreas de mejora de {nombre_joven} vs {seleccion_atm}")
                                    
                                    # Identificar métricas donde la joven es inferior a la jugadora ATM
                                    metricas_inferiores = []
                                    for metrica in metricas_radar:
                                        valor_joven = datos_joven_completos[metrica].iloc[0]
                                        valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                        
                                        if pd.notna(valor_joven) and pd.notna(valor_atm) and valor_joven < valor_atm:
                                            # Calcular la diferencia porcentual
                                            if valor_joven != 0:
                                                diff_pct = ((valor_atm - valor_joven) / valor_joven) * 100
                                                metricas_inferiores.append((metrica, valor_joven, valor_atm, diff_pct))
                                    
                                    # Ordenar por diferencia porcentual
                                    metricas_inferiores.sort(key=lambda x: x[3], reverse=True)
                                    
                                    # Mostrar las top 5 áreas de mejora
                                    for metrica, val_joven, val_atm, diff_pct in metricas_inferiores[:5]:
                                        nombre_metrica = metric_display_names.get(metrica, metrica)
                                        st.markdown(f"**{nombre_metrica}**: {val_joven:.2f} vs {val_atm:.2f} (-{diff_pct:.1f}%)")
                                
                                # Mostrar tabla completa con todas las métricas
                                with st.expander("Ver comparativa completa de métricas"):
                                    # Preparar datos para la tabla
                                    tabla_data = []
                                    
                                    for metrica in sorted(metricas_radar):
                                        valor_joven = datos_joven_completos[metrica].iloc[0]
                                        valor_atm = jugadoras_atm[jugadoras_atm['Player'] == seleccion_atm][metrica].iloc[0]
                                        
                                        if pd.notna(valor_joven) and pd.notna(valor_atm):
                                            # Calcular diferencia
                                            diff_abs = valor_joven - valor_atm
                                            
                                            # Determinar quién tiene ventaja
                                            if diff_abs > 0:
                                                ventaja = nombre_joven
                                            elif diff_abs < 0:
                                                ventaja = seleccion_atm
                                            else:
                                                ventaja = "Empate"
                                            
                                            # Calcular diferencia porcentual
                                            if valor_atm != 0:
                                                diff_pct = (diff_abs / valor_atm) * 100
                                            else:
                                                diff_pct = 0
                                            
                                            tabla_data.append({
                                                'Métrica': metric_display_names.get(metrica, metrica),
                                                nombre_joven: valor_joven,
                                                seleccion_atm: valor_atm,
                                                'Diferencia': diff_abs,
                                                'Diferencia (%)': f"{diff_pct:.1f}%",
                                                'Ventaja': ventaja
                                            })
                                    
                                    # Crear dataframe y mostrar
                                    df_tabla = pd.DataFrame(tabla_data)
                                    st.dataframe(df_tabla)
                            
                            else:
                                st.warning(f"No se encontraron métricas definidas para la posición {posicion_joven}.")
                        else:
                            st.info("Selecciona una jugadora del Atlético de Madrid para comparar.")
                    else:
                        st.warning(f"No hay jugadoras del Atlético de Madrid en la posición {posicion_joven}.")
                else:
                    st.error("No se encontraron datos completos para la jugadora seleccionada.")
            else:
                st.info("Selecciona una joven promesa para empezar la comparación.")
        else:
            st.warning("No hay jugadoras disponibles para comparar con los filtros actuales. Ajusta los filtros para encontrar jóvenes promesas.")
        
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
