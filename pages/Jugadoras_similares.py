import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import encontrar_jugadoras_similares

@st.cache_data
def cargar_datos():
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
    return df_players_info, df_teams_info, df_combined, df_atm_photos
    
def listar_jugadoras(df, posicion=None):
    """
    Lista las jugadoras disponibles en el dataset, opcionalmente filtradas por posición.
    
    Args:
        df (DataFrame): DataFrame con los datos de las jugadoras
        posicion (str, optional): Filtrar por posición (GK, DF, MF, FW)
        
    Returns:
        list: Lista de nombres de jugadoras
    """
    if posicion:
        jugadoras = df[df['Posición Principal'] == posicion]['Player'].unique().tolist()
    else:
        jugadoras = df['Player'].unique().tolist()
    
    return sorted(jugadoras)

def analizar_jugadora(nombre_jugadora, df, n_similares=10):
    """
    Realiza el análisis completo para una jugadora y muestra los resultados.
    
    Args:
        nombre_jugadora (str): Nombre de la jugadora a analizar
        df (DataFrame): DataFrame con los datos de todas las jugadoras
        n_similares (int): Número de jugadoras similares a buscar
        
    Returns:
        None
    """
    if nombre_jugadora not in df['Player'].values:
        print(f"Error: La jugadora '{nombre_jugadora}' no se encuentra en el dataset.")
        print("Asegúrate de escribir el nombre exactamente como aparece en los datos.")
        return
    
    try:
        # Llamar a la función para encontrar jugadoras similares
        print(f"\nAnalizando jugadora: {nombre_jugadora}")
        similar_players_df, fig = encontrar_jugadoras_similares(nombre_jugadora, df, n_similares)
        
        # Mostrar tabla de resultados
        print("\nRESULTADOS: Top 10 jugadoras más similares")
        print("="*80)
        
        # Formatear la tabla para mejor visualización
        result_table = similar_players_df[['Player', 'Squad', 'Born', 'Min', 'Distance']].copy()
        result_table['Distance'] = result_table['Distance'].round(4)
        result_table.columns = ['Jugadora', 'Equipo', 'Año Nacimiento', 'Minutos Jugados', 'Distancia']
        
        print(result_table.to_string(index=False))
        
        # Guardar la visualización
        output_filename = f"{nombre_jugadora.replace(' ', '_')}_similares.png"
        fig.savefig(output_filename)
        print(f"\nVisualización guardada como: {output_filename}")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")

def main():
    """
    Función principal que ejecuta la aplicación interactiva.
    """
    print("="*80)
    print("ANÁLISIS DE JUGADORAS SIMILARES EN FÚTBOL FEMENINO")
    print("="*80)
    
    # Cargar datos
    print("\nCargando datos...")
    df_players_info, df_teams_info, df_combined, df_atm_photos = cargar_datos()
    
    if df_combined is None:
        print("No se pudieron cargar los datos. Por favor verifica las rutas de los archivos.")
        return
    
    while True:
        print("\nOpciones:")
        print("1. Listar todas las jugadoras")
        print("2. Listar jugadoras por posición")
        print("3. Analizar una jugadora específica")
        print("4. Salir")
        
        opcion = input("\nSelecciona una opción (1-4): ")
        
        if opcion == '1':
            jugadoras = listar_jugadoras(df_combined)
            print(f"\nTotal de jugadoras: {len(jugadoras)}")
            print("Primeras 20 jugadoras:")
            for i, jugadora in enumerate(jugadoras[:20]):
                print(f"  {i+1}. {jugadora}")
            print("...")
            
        elif opcion == '2':
            print("\nPosiciones disponibles:")
            print("  GK - Porteras")
            print("  DF - Defensas")
            print("  MF - Centrocampistas")
            print("  FW - Delanteras")
            
            posicion = input("\nSelecciona una posición (GK/DF/MF/FW): ").upper()
            if posicion in ['GK', 'DF', 'MF', 'FW']:
                jugadoras = listar_jugadoras(df_combined, posicion)
                print(f"\nJugadoras en posición {posicion}: {len(jugadoras)}")
                for i, jugadora in enumerate(jugadoras[:20]):
                    print(f"  {i+1}. {jugadora}")
                if len(jugadoras) > 20:
                    print("...")
            else:
                print("Posición no válida.")
                
        elif opcion == '3':
            nombre_jugadora = input("\nIntroduce el nombre de la jugadora a analizar: ")
            n_similares = 10
            analizar_jugadora(nombre_jugadora, df_combined, n_similares)
            
        elif opcion == '4':
            print("\n¡Hasta pronto!")
            break
            
        else:
            print("Opción no válida. Por favor, selecciona una opción del 1 al 4.")

if __name__ == "__main__":
    main()