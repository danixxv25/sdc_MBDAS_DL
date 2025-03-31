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
    tab1, tab2, tab3 = st.tabs(["Ranking de Talentos", "Comparativa Detallada", "Visualización por Edad"])
    
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
                fig, ax = plt.subplots(figsize=(10, 4))
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
    
    with tab3:
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
