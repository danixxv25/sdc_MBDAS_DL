import pandas as pd

import random
import re
import os
import time
import pygit2

from termcolor import colored
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
os.environ['WDM_LOG'] = '0'
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_PRINT_FIRST_LINE'] = 'False'

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from tqdm import tqdm
import concurrent.futures

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

# Lista de user agents para rotación
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

#------------------------------------------------------------------------------------------------

urls = {
    "Liga F" : "https://fbref.com/en/comps/230/stats/Liga-F-Stats",
    "Women's Super League":"https://fbref.com/en/comps/189/Womens-Super-League-Stats",
    "Frauen-Bundesliga":"https://fbref.com/en/comps/183/Frauen-Bundesliga-Stats",
    "Première Ligue":"https://fbref.com/en/comps/193/Premiere-Ligue-Stats",
    "Serie A":"https://fbref.com/en/comps/208/Serie-A-Stats"
    }
urls_df = pd.DataFrame(list(urls.items()), columns=['Competición', 'URL'])
compID = []
for url in urls_df['URL']:
    id = url.split('/comps/')[1].split('/')[0]
    compID.append(id)

urls_df['comp_ID'] = compID
competitions_id = urls_df['comp_ID'].to_list()
competition_names = urls_df['Competición'].to_list()
competitions_urls = urls_df['URL'].to_list()

#------------------------------------------------------------------------------------------------

def setup_selenium():
    """Configura y devuelve un driver de Selenium con opciones para evitar detección"""
    # Suprimir los mensajes del WebDriver Manager
    chrome_service = Service(ChromeDriverManager().install())
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    # Suprimir los logs del navegador
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
    chrome_options.add_experimental_option('prefs', {'logging': {'browser': 'OFF'}})
    
    # Deshabilitar cualquier tipo de log
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--silent")
    
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    return driver

#------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------

def random_sleep_time():
    return random.uniform(5, 7.5)

#------------------------------------------------------------------------------------------------

def get_time(cond):
    if cond == "start":
        p = "El proceso comenzó a las "
    elif cond == "end":
        p = "El proceso finalizó a las "
    print("")
    print(colored(p + str(datetime.now().strftime("%H:%M:%S")), "green", "on_white", attrs=["bold",'reverse', 'blink']))

#------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------

def extract_team_logos(comps=competitions_id):
    get_time("start")
    output_folder = 'Proyecto_DL/data/teams_info'
    for c in comps:
        if c == '230':
            url = "https://ligaf.es/competicion/primera_division_femenina/282/2025"
                
            print("-------------------------------------------------------------")
            print(f"Extracción de los Logos y Nombres Oficiales de los Clubes de la competición {c}.")
            time.sleep(2)
            print("-------------------------------------------------------------")

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            time.sleep(4)

            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            club_names_list = []
            club_shields_list = []

            for card in soup.find_all('div', class_='team-info pb5'):
                club_name = card.find('p').text.strip()
                shield_link = card.find('img')['src']

                if club_name not in club_names_list:
                    club_names_list.append(club_name)
                    club_shields_list.append(shield_link)

            equipo_data = pd.DataFrame({
                "Squad": club_names_list,
                "Shield URL": club_shields_list
            })            
    
            df_name = f"df_{c}_teams_info"
            filename = f"{df_name}.csv"
            filepath = os.path.join(output_folder, filename)
            equipo_data.to_csv(filepath, index=False)

            print(f"DataFrame {df_name} creado y exportado con éxito en {filepath}.")
        
        elif c == '183':
            url = "https://www.dfb.de/frauen/ligen-frauen/google-pixel-frauen-bundesliga"
                
            print("-------------------------------------------------------------")
            print(f"Extracción de los Logos y Nombres Oficiales de los Clubes de la competición {c}.")
            time.sleep(2)
            print("-------------------------------------------------------------")

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            time.sleep(4)

            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            p = soup.find_all('div', class_='c-Table-emblem')
            
            club_names_list = []
            club_shields_list = []
            
            for x in p:
                club_name = x.find('img')['title']
                shield_link = x.find('img')['src']
                if club_name not in club_names_list:
                    club_names_list.append(club_name)
                    club_shields_list.append(shield_link)

            equipo_data = pd.DataFrame({
                "Squad": club_names_list,
                "Shield URL": club_shields_list
            })
    
            df_name = f"df_{c}_teams_info"
            filename = f"{df_name}.csv"
            filepath = os.path.join(output_folder, filename)
            equipo_data.to_csv(filepath, index=False)

            print(f"DataFrame {df_name} creado y exportado con éxito en {filepath}.")


        elif c == '189':
            url = "https://womensleagues.thefa.com/bwsl-clubs/"
    
            print("-------------------------------------------------------------")
            print(f"Extracción de los Logos y Nombres Oficiales de los Clubes de la competición {c}.")
            time.sleep(2)

            print("-------------------------------------------------------------")

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            p = soup.find_all('article', class_='global-clubCard')
            
            club_names_list = []
            club_shields_list = []
            
            for x in p:
                club_name = x.find('h3', class_="uppercase").text.strip()
                img_tag = x.find('img')
                shield_link = img_tag.get('data-src') or img_tag.get('data-srcset').split(' ')[0]
                if club_name not in club_names_list:
                    club_names_list.append(club_name)
                    club_shields_list.append(shield_link)

            equipo_data = pd.DataFrame({
                "Squad": club_names_list,
                "Shield URL": club_shields_list
            })

            df_name = f"df_{c}_teams_info.csv"
            filename = f"{df_name}"
            filepath = os.path.join(output_folder, filename)
            equipo_data.to_csv(filepath, index=False)

            print(f"DataFrame {df_name} creado y exportado con éxito en {filepath}.")


        elif c == '193':
            url = "https://www.fff.fr/competition/engagement/424635-arkema-premiere-ligue/phase/1/classement.html?gp_no=1"
                
            print("-------------------------------------------------------------")
            print(f"Extracción de los Logos y Nombres Oficiales de los Clubes de la competición {c}.")
            time.sleep(2)

            print("-------------------------------------------------------------")

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            table = soup.find('table', class_='ranking-group margin_b30')

            club_names_list = []
            club_shields_list = []

            for row in table.find_all("tr")[1:]:
                club_name = row.find('td', class_="uppercase text_left data-team").text.strip()
                shield_link = row.find('img')['src']

                if club_name not in club_names_list:
                    club_names_list.append(club_name)
                    club_shields_list.append(shield_link)

            name_mapping = {
                "OLYMPIQUE LYONNAIS": "Olympique Lyonnais",
                "PARIS SAINT-GERMAIN": "Paris Saint-Germain",
                "PARIS FC": "Paris FC",
                "DIJON FCO": "Dijon FCO",
                "FC FLEURY 91": "FC Fleury 91",
                "MONTPELLIER HSC": "Montpellier HSC",
                "FC NANTES": "FC Nantes",
                "HAVRE AC": "Havre AC",
                "AS SAINT ETIENNE": "AS Saint Etienne",
                "STADE DE REIMS": "Stade de Reims",
                "RC STRASBOURG ALSACE": "RC Strasbourg Alsace",
                "EA GUINGAMP": "EA Guingamp"
            }

            equipo_data = pd.DataFrame({
                "Squad": club_names_list,
                "Shield URL": club_shields_list
            })

            equipo_data['Squad'] = equipo_data['Squad'].replace(name_mapping)

            df_name = f"df_{c}_teams_info.csv"
            filename = f"{df_name}"
            filepath = os.path.join(output_folder, filename)
            equipo_data.to_csv(filepath, index=False)

            print(f"DataFrame {df_name} creado y exportado con éxito en {filepath}.")


        elif c == '208':
            url = "https://www.figc.it/it/femminile/club/club-serie-a/"
                
            print("-------------------------------------------------------------")
            print(f"Extracción de los Logos y Nombres Oficiales de los Clubes de la competición {c}.")
            time.sleep(2)

            print("-------------------------------------------------------------")

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            club_names_list = []
            club_shields_list = []

            for card in soup.find_all('div', class_='col-sm-4'):
                club_name = card.find('h5', class_='card-title').text.strip()
                shield_link = card.find('img')['src']
                shield_link = f"https://www.figc.it" + shield_link

                if club_name not in club_names_list:
                    club_names_list.append(club_name)
                    club_shields_list.append(shield_link)

            equipo_data = pd.DataFrame({
                            "Squad": club_names_list,
                            "Shield URL": club_shields_list
                        })

            df_name = f"df_{c}_teams_info.csv"
            filename = f"{df_name}"
            filepath = os.path.join(output_folder, filename)
            equipo_data.to_csv(filepath, index=False)

            print(f"DataFrame {df_name} creado y exportado con éxito en {filepath}.")
    get_time("end")

#------------------------------------------------------------------------------------------------

def fbref_player_ids1(comps=competitions_id):   
    get_time("start")

    urls = ["https://fbref.com/en/comps/{}/stats/", "https://fbref.com/en/comps/{}/keepers/"]
    output_folder1 = 'Proyecto_DL/data/players_info/'
    output_folder_global = 'Proyecto_DL/data/players_info/big/'
    
    # Crear directorios si no existen
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder_global, exist_ok=True)
    
    for c in comps:
        print("-------------------------------------------------------------")
        print(f"Iniciando raspado los Player_ID, Player_URL, headshots para la competición {c}.")

        i = 0

        for u in urls:
            i += 1
            time.sleep(5)

            players = []
            player_ids = []
            player_urls = []
            player_imgs = []
            player_weights = []
            player_heights = []
            player_birth_dates = []
            player_clubs = []

            url = u.format(c)

            if i == 1:
                table_id = "stats_standard"
                players_df_name = f"df_{c}_players_info"
            elif i == 2:
                table_id = "stats_keeper"
                players_df_name = f"df_{c}_keepers_info"

            filename = f"{players_df_name}.csv"
            filepath = os.path.join(output_folder1, filename)

            # Cargar archivo CSV existente si existe
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath)
                existing_ids = set(existing_data["Player_ID"])
                print(f"Archivo existente cargado: {filename}")
            else:
                existing_data = pd.DataFrame(columns=["Player", "Player_ID", "Height", "Weight", 
                                                     "Birth_Date", "Club", 
                                                     "Player_URL", "Photo"])
                existing_ids = set()
                print(f"No se encontró archivo existente. Creando nuevo: {filename}")

            # Abrir el navegador y obtener el HTML
            driver = setup_selenium()
            driver.get(url)
            html_page_source = driver.page_source

            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            # Contar número total de jugadoras en la tabla
            player_count = 0
            table = soup.find("table", id=table_id)

            for row in table.find_all("tr"):
                player_cell = row.find("td", {"data-stat": "player"})
                if player_cell:
                    player_count += 1
            
            print(f"Identificadas {player_count} jugadoras en la tabla. Filtrando duplicados...")

            # Inicializar contadores
            new_players = 0
            skipped_players = 0
            
            # Crear una única barra de progreso para la extracción de IDs
            id_progress = tqdm(total=player_count, desc="Extrayendo IDs", unit="jugadora")
            
            # Procesar cada fila de la tabla para extraer IDs           
            for row in table.find_all("tr"):
                player_cell = row.find("td", {"data-stat": "player"})
                if player_cell:
                    # Actualizar la barra de progreso
                    id_progress.update(1)
                    
                    # Extraer nombre del jugador(a)
                    player_name = player_cell.text.strip()
                    # Extraer player_id del atributo 'data-append-csv'
                    player_id = player_cell.get("data-append-csv")

                    if player_id and player_id not in player_ids:
                        players.append(player_name)
                        player_ids.append(player_id)
                        # Construir la URL del perfil del jugador(a)
                        player_urls.append(f"https://fbref.com/en/players/{player_id}/")
                        
                        # Actualizar lista de IDs existentes para evitar duplicados
                        existing_ids.add(player_id)
                        new_players += 1
                    else:
                        skipped_players += 1
            
            # Cerrar la barra de progreso de IDs al finalizar
            id_progress.close()
            
            print(f"Nuevas jugadoras añadidas: {new_players}")
            print(f"Jugadoras omitidas (duplicadas): {skipped_players}")

            # Extraer datos de los perfiles de los jugadores(as)
            print(f"Total de jugadoras a procesar: {len(player_urls)}")
            print("Iniciando extracción de datos de perfiles...")
            
            # Crear una única barra de progreso
            progress_bar = tqdm(total=len(player_urls), desc="Procesando perfiles", unit="jugadora")
            
            for url_idx, x in enumerate(player_urls):
                time.sleep(6)  # Esperar para no sobrecargar el servidor

                # Actualizar la barra de progreso con información sobre el perfil actual
                progress_bar.set_description(f"Procesando: {players[url_idx]}")
                progress_bar.update(1)

                driver = setup_selenium()  # Abrir Selenium nuevamente
                driver.get(x)
                html_page_source = driver.page_source

                soup = BeautifulSoup(html_page_source, "html.parser")
                driver.quit()

                # Buscar la sección donde está la información
                table_info = soup.find("div", id="info")
                
                # Inicializar valores por defecto
                image_url = "media/logos/silueta.jpg"
                height = None
                weight = None
                birth_date = None
                club = None
                
                if table_info:
                    # Extraer la URL de la imagen
                    image_tag = table_info.find('img')
                    if image_tag and 'src' in image_tag.attrs:
                        image_url = image_tag['src']
                    
                    # Extraer la altura y peso
                    height_weight_tag = table_info.select_one('p:contains("cm")')
                    if height_weight_tag:
                        # Extraer altura (164cm)
                        height_span = height_weight_tag.select_one('span:nth-of-type(1)')
                        if height_span:
                            height = height_span.text.strip()
                        
                        # Extraer peso (60kg)
                        weight_span = height_weight_tag.select_one('span:nth-of-type(2)')
                        if weight_span:
                            weight = weight_span.text.strip()
                    
                    # Extraer fecha de nacimiento
                    birth_tag = table_info.select_one('#necro-birth')
                    if birth_tag:
                        birth_date = birth_tag.text.strip()
                    
                    # Extraer club actual
                    club_tag = table_info.select_one('p:contains("Club:") a')
                    if club_tag:
                        club = club_tag.text.strip()
                
                # Agregar datos a las listas
                player_imgs.append(image_url)
                player_heights.append(height)
                player_weights.append(weight)
                player_birth_dates.append(birth_date)
                player_clubs.append(club)
            
            # Cerrar la barra de progreso al finalizar
            progress_bar.close()
            
            # Crear DataFrame con los datos recopilados
            new_data = pd.DataFrame({
                "Player": players,
                "Player_ID": player_ids,
                "Height": player_heights,
                "Weight": player_weights,
                "Birth_Date": player_birth_dates,
                "Club": player_clubs,
                "Player_URL": player_urls,
                "Photo": player_imgs
            })
            
            # Combinar datos nuevos con existentes
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            # Eliminar duplicados por Player_ID
            combined_data = combined_data.drop_duplicates(subset=["Player_ID"], keep="last")
            
            # Guardar DataFrame combinado
            combined_data.to_csv(filepath, index=False)
            print(f"Datos guardados en {filepath}")
            print(f"Completado: {len(player_urls)} perfiles procesados para {players_df_name}")
            
            # Mostrar un resumen
            print("\nRESUMEN:")
            print(f"- Total de jugadoras encontradas: {player_count}")
            print(f"- Nuevas jugadoras procesadas: {new_players}")
            print(f"- Jugadoras omitidas (duplicadas): {skipped_players}")
            print(f"- Total en la base de datos: {len(combined_data)}")
            print("-------------------------------------------------------------")
    
    get_time("end")
    
    # Concatenar todos los DataFrames en un archivo global
    print("\n-------------------------------------------------------------")
    print("Concatenando todos los DataFrames en un archivo global...")
    
    # Listas para almacenar DataFrames
    dfs_players = []
    dfs_keepers = []
    
    # Leer todos los archivos CSV generados
    for c in comps:
        # Intentar leer el archivo de jugadoras
        players_file = os.path.join(output_folder1, f"df_{c}_players_info.csv")
        try:
            df_players = pd.read_csv(players_file, sep=",", encoding="utf-8")
            dfs_players.append(df_players)
            print(f"Leído: {players_file}")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {players_file}")
        except Exception as e:
            print(f"Error al leer {players_file}: {e}")
        
        # Intentar leer el archivo de porteras
        keepers_file = os.path.join(output_folder1, f"df_{c}_keepers_info.csv")
        try:
            df_keepers = pd.read_csv(keepers_file, sep=",", encoding="utf-8")
            dfs_keepers.append(df_keepers)
            print(f"Leído: {keepers_file}")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {keepers_file}")
        except Exception as e:
            print(f"Error al leer {keepers_file}: {e}")
    
    # Verificar que hayamos encontrado al menos algunos DataFrames
    if dfs_players or dfs_keepers:
        # Concatenar DataFrames de jugadoras
        if dfs_players:
            big_df = pd.concat(dfs_players, axis=0)
            print(f"Concatenados {len(dfs_players)} DataFrames de jugadoras")
        else:
            big_df = pd.DataFrame()
            print("No se encontraron DataFrames de jugadoras para concatenar")
        
        # Añadir DataFrames de porteras
        if dfs_keepers:
            if not big_df.empty:
                big_df = pd.concat([big_df] + dfs_keepers, axis=0)
            else:
                big_df = pd.concat(dfs_keepers, axis=0)
            print(f"Añadidos {len(dfs_keepers)} DataFrames de porteras")
        
        # Eliminar duplicados y resetear índice
        if not big_df.empty:
            # Número de filas antes de eliminar duplicados
            rows_before = big_df.shape[0]
            
            # Resetear índice y eliminar duplicados
            big_df.reset_index(drop=True, inplace=True)
            big_df.drop_duplicates(subset='Player_ID', keep='first', inplace=True)
            
            # Número de filas después de eliminar duplicados
            rows_after = big_df.shape[0]
            duplicates_removed = rows_before - rows_after
            
            # Guardar el DataFrame combinado
            global_file = os.path.join(output_folder_global, "df_players_info_global.csv")
            big_df.to_csv(global_file, index=False)
            
            print(f"\nArchivo global creado: {global_file}")
            print(f"Total de jugadoras en el archivo global: {rows_after}")
            print(f"Duplicados eliminados: {duplicates_removed}")
        else:
            print("No se pudieron encontrar DataFrames para concatenar")
    else:
        print("No se encontraron archivos CSV para combinar")
    
    print("Proceso completo.")

#------------------------------------------------------------------------------------------------

def concat_teams_info_dfs(comps=competitions_id):
    output_folder = "Proyecto_DL/data/teams_info/big/"
    dfs_teams = []
    for i in comps:
        ruta = f"Proyecto_DL/data/teams_info/df_{i}_teams_info.csv"
        try:
            df = pd.read_csv(ruta, sep=",", encoding="utf-8")
            dfs_teams.append(df)
        except FileNotFoundError:
            print(f"File not found: {ruta}")
        except Exception as e:
            print(f"Error reading {ruta}: {e}")
    
    big_df = pd.concat(dfs_teams, axis=0)
    big_df.reset_index(drop=True, inplace=True)
    
    filename = f"df_teams_info_global.csv"
    filepath = os.path.join(output_folder, filename)
    big_df.to_csv(filepath, index=False)
    print(f"{filename} has been created and exported")

#------------------------------------------------------------------------------------------------

def fbref_extract_all_stats(comps=competitions_id, attributes = ["standard", "shooting", "passing", "passing_types", "gca", "defense", "possession", "playing_time", "misc", "keeper", "keeper_adv"]):
    get_time("start")
    url_base = "https://fbref.com/en/comps/{}/{}/"
    output_folder = '/Users/daniutus/Documents/MBDAS/M11_PFM/Proyecto_DL/data/raw_data/'
    attributes = ["standard", "shooting", "passing", "passing_types", "gca", "defense", "possession", "playing_time", "misc", "keeper", "keeper_adv"]
    #attributes = ["playing_time"]
    sleep_time = random_sleep_time()
    #fbref_player_ids(comps, sleep_time)
    
    for c in comps:
        for attr in attributes: 
               
            print("-------------------------------------------------------------")
            print(f"Iniciando raspado de las {attr} stats para la competición {c}. Durmiendo por {sleep_time:.2f} segundos")
            time.sleep(sleep_time)
            print("-------------------------------------------------------------")

            if attr =="standard":
                url = url_base.format(c, "stats")
            elif attr =="playing_time":
                url = url_base.format(c, "playingtime")
            elif attr =="keeper":
                url = url_base.format(c, "keepers")
            elif attr =="keeper_adv":
                url = url_base.format(c, "keepersadv")
            else:
                url = url_base.format(c, attr)
                
            id=f"stats_{attr}"

            driver = setup_selenium() #Abrir el navegador (instanciar un driver) - Función de usuario /functions.py

            driver.get(url)
            html_page_source = driver.page_source
            
            soup = BeautifulSoup(html_page_source, "html.parser")
            driver.quit()

            print(url)
            print(id)

            p = soup.find("table", id=id)

            if p is not None:
                
                temp_table = pd.read_html(str(p))[0]
                temp_table.fillna("N/A", inplace=True)

                df_name = f"df_{c}_{attr}_stats"
                globals()[df_name] = temp_table

                #Exportar el Dataframe a un archivo CSV
                filename = f"{df_name}.csv"
                filepath = os.path.join(output_folder, filename)
                temp_table.to_csv(filepath, index=False)

                print(f"DataFrame {df_name} creado y exportado con éxito a {filepath}.")
                    
            else:
                print(f"No se encontró tabla para {c} en {attr} Stats.")
            
            time.sleep(2)
    get_time("end")
 
#------------------------------------------------------------------------------------------------

def clean_structure(ruta):
    df = pd.read_csv(ruta, sep=",", encoding="utf-8", header=1)

    df = df[df['Player'] != 'Player']
    df = df.reset_index()
    df.drop(columns=['Rk', 'Matches', 'index', 'Age'], inplace=True)

    cols_no_num = ['Player', 'Nation', 'Pos', 'Squad', 'Age']
    cols_num = []
    cols_num = [col for col in df.columns if col not in cols_no_num]

    for col in cols_num:
        df[col] = pd.to_numeric(df[col])

    # Asegurarse de que los valores en la columna 'Nation' sean cadenas antes de aplicar split
    df['Nation'] = df['Nation'].fillna('Unknown')  # Reemplazar NaN con un valor predeterminado
    df['Nation'] = df['Nation'].astype(str).apply(lambda x: x.split(' ')[1] if ' ' in x else x)
    df.columns = [col.replace('.1', '/90') for col in df.columns]


    # Rellenar NaN
    total_nan = df.isna().sum().sum()
    if not total_nan ==0:
        df = df.fillna(0)

    return df

#------------------------------------------------------------------------------------------------

def clean_passing(df):
    df.columns = [col.replace("/90", "_short") for col in df.columns]
    df.columns = [col.replace(".2", "_med") for col in df.columns]
    df.columns = [col.replace(".3", "_long") for col in df.columns]
    df.columns = [col.replace("1/3", "pass_1/3") for col in df.columns]


    return df

#------------------------------------------------------------------------------------------------

def clean_passingtypes(df):
    df.columns = [col.replace("FK", "FK_pass") for col in df.columns]
    df.columns = [col.replace("Sw", "Switch") for col in df.columns]
    df.columns = [col.replace("CK", "CK_pass") for col in df.columns]
    df.columns = [col.replace("TB", "BallThrough") for col in df.columns]
    
    return df

#------------------------------------------------------------------------------------------------

def clean_possession(df):
    df.columns = [col.replace("1/3", "carries_1/3") for col in df.columns]
    df.columns = [col.replace("TotDist", "carries_TotDist") for col in df.columns]
    df.columns = [col.replace("PrgDist", "carries_PrgDist") for col in df.columns]
    df.columns = [col.replace("Def Pen", "touch_Def Pen") for col in df.columns]
    df.columns = [col.replace("Def 3rd", "touch_Def 3rd") for col in df.columns]
    df.columns = [col.replace("Mid 3rd", "touch_Mid 3rd") for col in df.columns]
    df.columns = [col.replace("Att 3rd", "touch_Att 3rd") for col in df.columns]
    df.columns = [col.replace("Att Pen", "touch_Att Pen") for col in df.columns]
    df.columns = [col.replace("Succ%", "TO_Succ%") for col in df.columns]

    return df

#------------------------------------------------------------------------------------------------

def clean_defense(df): 
    df.columns = [col.replace("Def 3rd", "tkl_Def 3rd") for col in df.columns]
    df.columns = [col.replace("Mid 3rd", "tkl_Mid 3rd") for col in df.columns]
    df.columns = [col.replace("Att 3rd", "tkl_Att 3rd") for col in df.columns]

    return df

#------------------------------------------------------------------------------------------------

def clean_gk_std(df):
    df.columns = [col.replace("/90", "_PK") for col in df.columns]

    return df

#------------------------------------------------------------------------------------------------

def clean_gk_adv(df):
    rename_dict = {
        'Cmp': 'Pass_Cmp_+40y',
        'Cmp%': 'Pass_Cmp_+40y%',
        'Att': 'Pass_Att_+40y',
        'Att (GK)': 'GK_Att',
        'Launch%': 'Pass_Launch%',
        'Launch%/90': 'GK_Launch%',
        'AvgLen': 'Pass_AvgLen',
        'AvgLen/90': 'GK_AvgLen',
        'Att/90': 'GK_Att',
        "PSxG+/-": "PSxG-GA"
    }

    # Renombrar columnas usando el diccionario
    df.columns = [rename_dict[col] if col in rename_dict else col for col in df.columns]

    return df

#------------------------------------------------------------------------------------------------

def data_clearing_silver(comps=competitions_id):
  
    attributes = ["standard", "shooting", "gca", "passing", "passing_types", "possession", "playing_time","defense","misc","keeper", "keeper_adv"]
    output_folder = 'Proyecto_DL/data/data_silver'

    cols_keep_df1 = ["Player","Nation","Pos","Squad","Born","MP","Min","90s","Gls","Ast","G+A","G-PK","xAG","Gls/90","Ast/90","G+A/90","G-PK/90","xG/90","xAG/90","xG+xAG","npxG/90"]
    cols_keep_df2 = ["Player","Sh","SoT", "SoT%","SoT/90","G/Sh","Dist","FK","PK","PKatt","xG","npxG","npxG/Sh","G-xG","np:G-xG"]
    cols_keep_df3 = ["Player","SCA","SCA90","GCA","GCA90"]
    cols_keep_df4 = ["Player","Cmp","Att","Cmp%","TotDist","PrgDist","Cmp_short","Att_short","Cmp%_short","Cmp_med","Att_med","Cmp%_med","Cmp_long","Att_long","Cmp%_long","xA","A-xAG","KP","pass_1/3","PPA","CrsPA","PrgP"]
    cols_keep_df5 = ["Player","FK_pass","BallThrough","Switch","Crs","CK_pass","Off","Blocks"]
    cols_keep_df6 = ["Player","Touches","touch_Def Pen","touch_Def 3rd","touch_Mid 3rd","touch_Att 3rd","touch_Att Pen","TO_Succ%","Carries","carries_TotDist","carries_PrgDist","PrgC","carries_1/3","CPA","Rec","PrgR"]
    cols_keep_df7 = ["Player","Mn/MP","Min%","Starts","Mn/Start","Compl","Subs","Mn/Sub","unSub","PPM","onG","onGA","+/-","+/-90","On-Off/90"]
    cols_keep_df8 = ["Player","Tkl","TklW","tkl_Def 3rd","tkl_Mid 3rd","tkl_Att 3rd","Tkl/90","Tkl%","Blocks","Int","Tkl+Int","Clr"]
    cols_keep_df9 = ["Player","CrdY","CrdR","2CrdY","Fls","Fld","Off","PKwon","OG","Recov","Won","Lost","Won%"]
    cols_keep_df10 = ["Player","Pos","Squad","Born","MP","Starts","Min","90s","GA","GA90","SoTA","Saves","Save%","W","D","L","CS","CS%","PKatt","PKA","Save%_PK"]
    cols_keep_df11 = ["Player",'FK','CK','OG','PSxG','PSxG/SoT','PSxG-GA','Pass_Cmp_+40y','Pass_Att_+40y','Pass_Cmp_+40y%','GK_Att','Pass_Launch%','Pass_AvgLen','Opp','Stp%','#OPA','#OPA/90','AvgDist']

    for comp in comps:
        i = 0
        for attr in attributes:
            i = i+1

            ruta = f"Proyecto_DL/data/raw_data/df_{comp}_{attr}_stats.csv"
            ruta_save = f"Proyecto_DL/data/data_bronze/df_{comp}_{attr}_bronze.csv"
            ruta_pl_info =f"Proyecto_DL/data/players_info/df_{comp}_players_info.csv"
            ruta_gk_info =f"Proyecto_DL/data/players_info/df_{comp}_keepers_info.csv"

            temp_table = clean_structure(ruta)

            if attr == "standard":
                temp_table = temp_table[cols_keep_df1]
                temp_table['Born'] = temp_table['Born'].astype(int)
                temp_table.to_csv(ruta_save)
                df1_std_filt = temp_table.set_index('Player')

            elif attr == "shooting":
                temp_table = temp_table[cols_keep_df2]
                temp_table.to_csv(ruta_save)
                df2_shooting_filt = temp_table.set_index('Player')

            elif attr == "gca":
                temp_table = temp_table[cols_keep_df3]
                temp_table.to_csv(ruta_save)
                df3_gca_filt = temp_table.set_index('Player')

            elif attr == "passing":
                temp_table = clean_passing(temp_table)
                temp_table = temp_table[cols_keep_df4]
                temp_table.to_csv(ruta_save)
                df4_passing_filt = temp_table.set_index('Player')

            elif attr == "passing_types":
                temp_table = clean_passingtypes(temp_table)
                temp_table = temp_table[cols_keep_df5]
                temp_table.to_csv(ruta_save)
                df5_passing_types_filt = temp_table.set_index('Player')

            elif attr == "possession":
                temp_table = clean_possession(temp_table)
                temp_table = temp_table[cols_keep_df6]
                temp_table.to_csv(ruta_save)
                df6_possession_filt = temp_table.set_index('Player')

            elif attr == "playing_time":
                temp_table = temp_table[temp_table['MP'] > 0]
                temp_table = temp_table[cols_keep_df7]
                temp_table.to_csv(ruta_save)
                df7_playingtime_filt = temp_table.set_index('Player')

            elif attr == "defense":
                temp_table = clean_defense(temp_table)
                temp_table = temp_table[cols_keep_df8]
                temp_table.to_csv(ruta_save)
                df8_defense_filt = temp_table.set_index('Player')

            elif attr == "misc":
                temp_table = temp_table[cols_keep_df9]
                temp_table.to_csv(ruta_save)
                df9_misc_filt = temp_table.set_index('Player')

            elif attr == "keeper":
                temp_table = clean_gk_std(temp_table)
                temp_table = temp_table[cols_keep_df10]
                temp_table['Born'] = temp_table['Born'].astype(int)
                temp_table.to_csv(ruta_save)
                df10_keeper_filt = temp_table.set_index('Player')

            elif attr == "keeper_adv":
                temp_table = clean_gk_adv(temp_table)
                temp_table = temp_table[cols_keep_df11]
                temp_table.to_csv(ruta_save)
                df11_keeperadv_filt = temp_table.set_index('Player')

            #df_name = f"df{i}_{attr}_filt"
            #globals()[df_name] = temp_table

        df_final = f"df_final_{comp}"
        gk_final = f"gk_final_{comp}"
        
        players = pd.concat([df1_std_filt, df2_shooting_filt, df3_gca_filt, 
                             df4_passing_filt, df5_passing_types_filt, df6_possession_filt, df7_playingtime_filt, 
                             df8_defense_filt, df9_misc_filt], axis=1)
        
        gks = pd.concat([df10_keeper_filt, df11_keeperadv_filt], axis=1)

        if comp == '230':
            players.insert(loc=players.columns.get_loc('Born'), column='League', value= "Liga F - ESP")
            gks.insert(loc=gks.columns.get_loc('Born'), column='League', value= "Liga F - ESP")
        if comp == '189':
            players.insert(loc=players.columns.get_loc('Born'), column='League', value= "Women's Super League - ENG")
            gks.insert(loc=gks.columns.get_loc('Born'), column='League', value= "Women's Super League - ENG")
        if comp == '183':
            players.insert(loc=players.columns.get_loc('Born'), column='League', value= "Frauen-Bundesliga - GER")
            gks.insert(loc=gks.columns.get_loc('Born'), column='League', value= "Frauen-Bundesliga - GER")
        if comp == '193':
            players.insert(loc=players.columns.get_loc('Born'), column='League', value= "Première Ligue - FRA")
            gks.insert(loc=gks.columns.get_loc('Born'), column='League', value= "Première Ligue - FRA")
        if comp == '208':
            players.insert(loc=players.columns.get_loc('Born'), column='League', value= "Serie A - ITA")
            gks.insert(loc=gks.columns.get_loc('Born'), column='League', value= "Serie A - ITA")

        players = players.reset_index()
        gks = gks.reset_index()

        #Exportar los Dataframe a archivos CSV
        filename= f"df_{comp}_players_silver.csv"
        filepath = os.path.join(output_folder, filename)
        players.to_csv(filepath, index=False)
        
        filename = f"df_{comp}_keepers_silver.csv"
        filepath = os.path.join(output_folder, filename)
        gks.to_csv(filepath, index=False)

        #players.to_csv("Proyecto_DL/data/data_silver/df_players_silver.csv")
        #gks.to_csv("Proyecto_DL/data/data_silver/df_keepers_silver.csv")

        globals()[df_final] = players
        globals()[gk_final] = gks

#------------------------------------------------------------------------------------------------

def missing_data():# Datos de las jugadoras
    data = {
        "Player": [
            "Nerea Carmona", 
            "Alba Cerrato", 
            "Aina Durán", 
            "Ylenia Estrella", 
            "Esther Gómez", 
            "Carla Julià", 
            "Paula Sánchez", 
            "Sara Tamarit", 
            "Julia Torres", 
            "Olga Ahtinen", 
            "Maddy Duffy", 
            "Sammy Kaczmar", 
            "Lauren Thomas", 
            "Kassandra Potsi", 
            "Kaylie Ronan", 
            "Cora Zical", 
            "Nina Falgayrac", 
            "Victoria Della", 
            "Langella Gabriella", 
            "Olamide Sandra Adugbe",
            "Rosignoli Syria"
        ],
        "Born": [
            2007,  # Nerea Carmona [1]
            2007,  # Alba Cerrato [2][5]
            2003,  # Aina Durán [8]
            2004,  # Ylenia Estrella (sin datos disponibles)
            2003,  # Esther Gómez [12][14]
            2006,  # Carla Julià [16]
            2000,  # Paula Sánchez [19]
            2005,  # Sara Tamarit [23][25]
            2009,  # Julia Torres (sin datos disponibles)
            1997,  # Olga Ahtinen [28][29]
            2007,  # Maddy Duffy (sin datos disponibles)
            2007,  # Sammy Kaczmar (sin datos disponibles)
            2000,  # Lauren Thomas (sin datos disponibles)
            2008,  # Kassandra Potsi (sin datos disponibles)
            2002,  # Kaylie Ronan (sin datos disponibles)
            2004,  # Cora Zical (sin datos disponibles)
            2007,  # Nina Falgayrac (sin datos disponibles)
            2004,  # Victoria Della (sin datos disponibles)
            2007,  # Langella Gabriella (sin datos disponibles)
            2003,   # Olamide Sandra Adugbe (sin datos disponibles)
            2006
        ],
        "Nation": [
            "ESP",   # Nerea Carmona [1]
            "ESP",   # Alba Cerrato [2][5]
            "ESP",   # Aina Durán [8]
            "ESP",       # Ylenia Estrella (sin datos disponibles)
            "ESP",   # Esther Gómez [12][14]
            "ESP",   # Carla Julià [16]
            "ESP",   # Paula Sánchez [19]
            "ESP",   # Sara Tamarit [23][25]
            "ESP",       # Julia Torres (sin datos disponibles)
            "FIN",# Olga Ahtinen [28][29]
            "ENG",       # Maddy Duffy (sin datos disponibles)
            "ENG",       # Sammy Kaczmar (sin datos disponibles)
            "ENG",       # Lauren Thomas (sin datos disponibles)
            "GER",       # Kassandra Potsi (sin datos disponibles)
            "USA",       # Kaylie Ronan (sin datos disponibles)
            "GER",       # Cora Zical (sin datos disponibles)
            "FRA",       # Nina Falgayrac (sin datos disponibles)
            "ITA",       # Victoria Della (sin datos disponibles)
            "ITA",       # Langella Gabriella (sin datos disponibles)
            "NGA",       # Olamide Sandra Adugbe (sin datos disponibles)
            "ITA"
        ]
    }

    # Crear el dataframe
    df = pd.DataFrame(data)
    df.to_csv("Proyecto_DL/data/players_info/missing_data.csv")

#------------------------------------------------------------------------------------------------

def data_clearing_gold(comps=competitions_id):

    positions = ['players', 'keepers']
    attributes = ["standard", "shooting", "gca", "passing", "passing_types", "possession", "playing_time","defense","misc","keeper", "keeper_adv"]
    dfs_players = []
    dfs_gks = []
    output_folder = "Proyecto_DL/data/data_gold/"

    squads_mapping = {"Leverkusen" : "Bayer 04 Leverkusen",
                          "Köln" : "1. FC Köln",
                          "Freiburg" : "SC Freiburg",
                          "Eint Frankfurt" : "Eintracht Frankfurt",
                          "Hoffenheim" : "TSG Hoffenheim",
                          "RB Leipzig" : "RB Leipzig",
                          "Werder Bremen" : "SV Werder Bremen",
                          "Essen" : "SGS Essen",
                          "Wolfsburg" : "VfL Wolfsburg",
                          "Turbine Potsdam" : "Turbine Potsdam",
                          "Carl-Zeiss Jena" : "FC Carl Zeiss Jena",
                          "Bayern Munich" : "Bayern München",

                          "West Ham" : "West Ham United",
                          "Brighton" : "Brighton & Hove Albion",
                          "Tottenham" : "Tottenham Hotspur",
                          "Leicester City" : "Leicester City",
                          "Manchester City": "Manchester City",
                          "Crystal Palace" : "Crystal Palace",
                          "Manchester Utd" : "Manchester United",
                          "Aston Villa" : "Aston Villa",
                          "Chelsea" : "Chelsea",
                          "Liverpool" : "Liverpool",
                          "Everton" : "Everton",
                          "Arsenal" : "Arsenal",

                          "Guingamp" : "EA Guingamp",
                          "Stade de Reims" : "Stade de Reims",
                          "Le Havre" : "Havre AC",
                          "Paris S-G" : "Paris Saint-Germain",
                          "Saint-Étienne" : "AS Saint Etienne",
                          "Strasbourg" : "RC Strasbourg Alsace",
                          "Lyon" : "Olympique Lyonnais",
                          "Montpellier" : "Montpellier HSC",
                          "Fleury" : "FC Fleury 91",
                          "Paris FC" : "Paris FC",
                          "Nantes" : "FC Nantes",
                          "Dijon" : "Dijon FCO",

                          "Sassuolo" : "Sassuolo",
                          "AS Roma" : "Roma",
                          "Internazionale" : "Inter",
                          "Napoli" : "Napoli Femminile",
                          "Sampdoria" : "Sampdoria",
                          "FC Como Women" : "Como Women",
                          "Milan" : "Milan",
                          "Fiorentina" : "Fiorentina",
                          "Lazio" : "Lazio",
                          "Juventus" : "Juventus",

                          "Real Madrid" : "Real Madrid CF",
                          "Athletic Club" : "Athletic Club",
                          "Real Betis" : "R. Betis Féminas",
                          "Sevilla" : "Sevilla FC",
                          "Atlético Madrid" : "Atlético de Madrid",
                          "Valencia" : "VCF Femenino",
                          "Levante" : "Levante UD",
                          "Eibar" : "SD Eibar",
                          "Dep La Coruña" : "Deportivo Abanca",
                          "Granada" : "Granada CF",
                          "Madrid CFF" : "Madrid CFF",
                          "Espanyol" : "RCD Espanyol",
                          "Real Sociedad" : "Real Sociedad",
                          "UDG Tenerife" : "Costa Adeje Tenerife",
                          "Levante Planas" : "FC Levante Badalona",
                          "Barcelona" : "FC Barcelona"}  

    missing_data()

    missing_birth_years_file = 'Proyecto_DL/data/players_info/missing_data.csv'
    if os.path.exists(missing_birth_years_file):
        missing_birth_years_df = pd.read_csv(missing_birth_years_file)
    else:
        print(f"Archivo {missing_birth_years_file} no encontrado. Asegúrate de crearlo antes de ejecutar el script.")
    
    for i in comps:
        for pos in positions:
            ruta = f"Proyecto_DL/data/data_silver/df_{i}_{pos}_silver.csv"
            teams_info = pd.read_csv(f"Proyecto_DL/data/teams_info/df_{i}_teams_info.csv", sep=',', encoding='utf-8')

            try:
                df = pd.read_csv(ruta, sep=",", encoding="utf-8")
                if pos == 'players':
                    dfs_players.append(df)
                elif pos == 'keepers':
                    dfs_gks.append(df)
            except FileNotFoundError:
                print(f"File not found: {ruta}")
            except Exception as e:
                print(f"Error reading {ruta}: {e}")
    
    if dfs_players:
        big_df_players = pd.concat(dfs_players, axis=0)
        big_df_players.reset_index(drop=True, inplace=True)

        big_df_players.drop_duplicates(subset='Player', keep='first', inplace=True)

        missing_indices = big_df_players.index[big_df_players['Born'] == 0].tolist()
        if missing_indices:  # Si hay registros con Born == 0
            print("\nReemplazando años de nacimiento faltantes...")
            
            for idx in missing_indices:
                player_name = big_df_players.loc[idx, 'Player']
                
                # Buscar el año de nacimiento en el archivo CSV
                matching_row = missing_birth_years_df[missing_birth_years_df['Player'] == player_name]
                
                if not matching_row.empty:
                    # Reemplazar el valor en el DataFrame principal
                    big_df_players.loc[idx, 'Born'] = matching_row['Born'].values[0]
                    big_df_players.loc[idx, 'Nation'] = matching_row['Nation'].values[0]
                else:
                    print(f"No se encontraron todos los datos para '{player_name}' en {missing_birth_years_file}.")
       
        big_df_players['Squad'] = big_df_players['Squad'].map(squads_mapping)

        big_df_players[['Posición Principal', 'Posición Secundaria']] = big_df_players['Pos'].str.split(',', expand=True)
        big_df_players['Posición Secundaria'] = big_df_players['Posición Secundaria'].fillna('None')

        # Eliminar la columna 'Pos'
        big_df_players = big_df_players.drop(columns=['Pos'])

        # Reordenar las columnas para que las nuevas columnas estén en la posición original de 'Pos'
        columns = list(big_df_players.columns)
        columns.remove('Posición Principal')
        columns.remove('Posición Secundaria')

        # Insertar las nuevas columnas en la posición original de 'Pos'
        columns.insert(3, 'Posición Principal')  # Posición original de 'Pos'
        columns.insert(4, 'Posición Secundaria')  # Justo después de 'Posición Principal'

        big_df_players = big_df_players[columns]

        filename = f"df_players_gold_1.csv"
        filepath = os.path.join(output_folder, filename)
        big_df_players.to_csv(filepath, index=False)
        print("Players dataframe has been created and exported")
    else:
        big_df_players = None
        print("No players dataframes found.")
    
    if dfs_gks:
        big_df_gks = pd.concat(dfs_gks, axis=0)
        big_df_gks.reset_index(drop=True, inplace=True)

        big_df_gks['Squad'] = big_df_gks['Squad'].map(squads_mapping)

        filename = f"df_keepers_gold_1.csv"
        filepath = os.path.join(output_folder, filename)
        big_df_gks.to_csv(filepath, index=False)
        print("Goalkeepers dataframe has been created and exported")
    else:
        big_df_gks = None
        print("No keepers dataframes found.")

#------------------------------------------------------------------------------------------------

# Función para manejar problemas de reintentos en caso de errores de conexión
def retry_request(url, max_retries=4, delay=5):
    """Intenta realizar una solicitud con reintentos en caso de fallos"""
    for attempt in range(max_retries):
        try:
            driver = setup_selenium()
            driver.get(url)
            html = driver.page_source
            driver.quit()
            return html
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Intento {attempt+1} fallido para {url}. Reintentando en {delay} segundos...")
                time.sleep(delay)
                delay *= 2  # Espera exponencial entre intentos
            else:
                print(f"Todos los intentos fallaron para {url}")
                raise

#------------------------------------------------------------------------------------------------

def fbref_player_ids(comps=competitions_id):   
    get_time("start")

    urls = ["https://fbref.com/en/comps/{}/stats/", "https://fbref.com/en/comps/{}/keepers/"]
    output_folder1 = 'Proyecto_DL/data/players_info/'
    output_folder_global = 'Proyecto_DL/data/players_info/big/'
    
    # Crear directorios si no existen
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder_global, exist_ok=True)
       
    for c in comps:
        print("-------------------------------------------------------------")
        print(f"Iniciando raspado los Player_ID, Player_URL, headshots para la competición {c}.")

        i = 0

        for u in urls:
            i = i + 1
            time.sleep(random_sleep_time())  # Usando tu función de tiempo aleatorio

            players = []
            player_ids = []
            player_urls = []
            player_imgs = []
            player_weights = []
            player_heights = []
            player_birth_dates = []
            player_clubs = []

            url = u.format(c)

            if i == 1:
                table_id = "stats_standard"
                players_df_name = f"df_{c}_players_info"
            elif i == 2:
                table_id = "stats_keeper"
                players_df_name = f"df_{c}_keepers_info"

            filename = f"{players_df_name}.csv"
            filepath = os.path.join(output_folder1, filename)

            # Cargar archivo CSV existente si existe
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath)
                existing_ids = set(existing_data["Player_ID"])
                print(f"Archivo existente cargado: {filename}")
            else:
                existing_data = pd.DataFrame(columns=["Player", "Player_ID", "Height", "Weight", 
                                                     "Birth_Date", "Club", 
                                                     "Player_URL", "Photo"])
                existing_ids = set()
                print(f"No se encontró archivo existente. Creando nuevo: {filename}")

            # Abrir el navegador y obtener el HTML con reintentos
            html_page_source = retry_request(url, max_retries=4, delay=5)
            soup = BeautifulSoup(html_page_source, "html.parser")

            player_count = 0
            table = soup.find("table", id=table_id)

            for row in table.find_all("tr"):
                player_cell = row.find("td", {"data-stat": "player"})
                if player_cell:
                    player_count += 1
            
            print(f"Identificadas {player_count} jugadoras en la tabla. Filtrando duplicados...")

            # Inicializar contadores
            new_players = 0
            skipped_players = 0
            
            # Crear una única barra de progreso para la extracción de IDs
            id_progress = tqdm(total=player_count, desc="Extrayendo IDs", unit="jugadora")
            
            # Procesar cada fila de la tabla para extraer IDs           
            for row in table.find_all("tr"):
                player_cell = row.find("td", {"data-stat": "player"})
                if player_cell is not None:
                    # Actualizar la barra de progreso
                    id_progress.update(1)
                    
                    # Extraer nombre del jugador(a)
                    player_name = player_cell.text.strip()
                    # Extraer player_id del atributo 'data-append-csv'
                    player_id = player_cell.get("data-append-csv")

                    if player_id and player_id not in existing_ids and player_id not in player_ids:
                        players.append(player_name)
                        player_ids.append(player_id)
                        # Construir la URL del perfil del jugador(a)
                        player_urls.append(f"https://fbref.com/en/players/{player_id}/")
                        
                        # Actualizar lista de IDs existentes para evitar duplicados
                        new_players += 1
                    else:
                        skipped_players += 1
            
            # Cerrar la barra de progreso de IDs al finalizar
            id_progress.close()
            
            print(f"Nuevas jugadoras añadidas: {new_players}")
            print(f"Jugadoras omitidas (duplicadas): {skipped_players}")

            # Extraer datos de los perfiles de los jugadores(as)
            print(f"Total de jugadoras a procesar: {len(player_urls)}")
            print("Iniciando extracción de datos de perfiles...")
            
            # Procesar perfiles de jugadoras secuencialmente en lugar de en paralelo
            profile_progress = tqdm(total=len(player_urls), desc="Procesando perfiles", unit="jugadora")
            
            # Inicializar listas para resultados
            for idx, player_url in enumerate(player_urls):
                player_name = players[idx]
                profile_progress.set_description(f"Procesando: {player_name}")
                
                try:
                    # Añadir tiempo de espera entre solicitudes para evitar errores 429
                    time.sleep(random_sleep_time() * 1.5)  # Aumentamos el tiempo de espera
                    
                    # Usar la función de reintentos
                    html_page_source = retry_request(player_url, max_retries=4, delay=5)
                    soup = BeautifulSoup(html_page_source, "html.parser")

                    # Buscar la sección donde está la información
                    table_info = soup.find("div", id="info")
                    
                    # Inicializar valores por defecto
                    image_url = "media/logos/silueta.jpg"
                    height = None
                    weight = None
                    birth_date = None
                    club = None
                    
                    if table_info:
                        # Extraer la URL de la imagen
                        image_tag = table_info.find('img')
                        if image_tag and 'src' in image_tag.attrs:
                            image_url = image_tag['src']
                        
                        # Extraer la altura y peso
                        height_weight_tag = table_info.select_one('p:contains("cm")')
                        if height_weight_tag:
                            # Extraer altura (164cm)
                            height_span = height_weight_tag.select_one('span:nth-of-type(1)')
                            if height_span:
                                height = height_span.text.strip()
                            
                            # Extraer peso (60kg)
                            weight_span = height_weight_tag.select_one('span:nth-of-type(2)')
                            if weight_span:
                                weight = weight_span.text.strip()
                        
                        # Extraer fecha de nacimiento
                        birth_tag = table_info.select_one('#necro-birth')
                        if birth_tag:
                            birth_date = birth_tag.text.strip()
                        
                        # Extraer club actual
                        club_tag = table_info.select_one('p:contains("Club:") a')
                        if club_tag:
                            club = club_tag.text.strip()
                    
                    # Guardar resultados
                    player_imgs.append(image_url)
                    player_heights.append(height)
                    player_weights.append(weight)
                    player_birth_dates.append(birth_date)
                    player_clubs.append(club)
                    
                except Exception as e:
                    # En caso de error, usar valores por defecto
                    player_imgs.append("media/logos/silueta.jpg")
                    player_heights.append(None)
                    player_weights.append(None)
                    player_birth_dates.append(None)
                    player_clubs.append(None)
                
                # Actualizar la barra de progreso
                profile_progress.update(1)
            
            # Cerrar la barra de progreso
            profile_progress.close()
            
            # Crear DataFrame con los datos recopilados
            new_data = pd.DataFrame({
                "Player": players,
                "Player_ID": player_ids,
                "Height": player_heights,
                "Weight": player_weights,
                "Birth_Date": player_birth_dates,
                "Club": player_clubs,
                "Player_URL": player_urls,
                "Photo": player_imgs
            })
            
            # Combinar datos nuevos con existentes
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            # Eliminar duplicados por Player_ID
            combined_data = combined_data.drop_duplicates(subset=["Player_ID"], keep="last")
            
            # Guardar DataFrame combinado
            combined_data.to_csv(filepath, index=False)
            print(f"Datos guardados en {filepath}")
            print(f"Completado: {len(player_urls)} perfiles procesados para {players_df_name}")
            
            # Mostrar un resumen
            print("\nRESUMEN:")
            print(f"- Total de jugadoras encontradas: {player_count}")
            print(f"- Nuevas jugadoras procesadas: {new_players}")
            print(f"- Jugadoras omitidas (duplicadas): {skipped_players}")
            print(f"- Total en la base de datos: {len(combined_data)}")
            print("-------------------------------------------------------------")
            
    
    get_time("end")
    
    # El resto del código para concatenar DataFrames se mantiene igual
    print("\n-------------------------------------------------------------")
    print("Concatenando todos los DataFrames en un archivo global...")
    
    # Listas para almacenar DataFrames
    dfs_players = []
    dfs_keepers = []
    
    # Leer todos los archivos CSV generados
    for c in comps:
        # Intentar leer el archivo de jugadoras
        players_file = os.path.join(output_folder1, f"df_{c}_players_info.csv")
        try:
            df_players = pd.read_csv(players_file, sep=",", encoding="utf-8")
            dfs_players.append(df_players)
            print(f"Leído: {players_file}")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {players_file}")
        except Exception as e:
            print(f"Error al leer {players_file}: {e}")
        
        # Intentar leer el archivo de porteras
        keepers_file = os.path.join(output_folder1, f"df_{c}_keepers_info.csv")
        try:
            df_keepers = pd.read_csv(keepers_file, sep=",", encoding="utf-8")
            dfs_keepers.append(df_keepers)
            print(f"Leído: {keepers_file}")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {keepers_file}")
        except Exception as e:
            print(f"Error al leer {keepers_file}: {e}")
    
    # Verificar que hayamos encontrado al menos algunos DataFrames
    if dfs_players or dfs_keepers:
        # Concatenar DataFrames de jugadoras
        if dfs_players:
            big_df = pd.concat(dfs_players, axis=0)
            print(f"Concatenados {len(dfs_players)} DataFrames de jugadoras")
        else:
            big_df = pd.DataFrame()
            print("No se encontraron DataFrames de jugadoras para concatenar")
        
        # Añadir DataFrames de porteras
        if dfs_keepers:
            if not big_df.empty:
                big_df = pd.concat([big_df] + dfs_keepers, axis=0)
            else:
                big_df = pd.concat(dfs_keepers, axis=0)
            print(f"Añadidos {len(dfs_keepers)} DataFrames de porteras")
        
        # Eliminar duplicados y resetear índice
        if not big_df.empty:
            # Número de filas antes de eliminar duplicados
            rows_before = big_df.shape[0]
            
            # Resetear índice y eliminar duplicados
            big_df.reset_index(drop=True, inplace=True)
            big_df.drop_duplicates(subset='Player_ID', keep='first', inplace=True)
            
            # Número de filas después de eliminar duplicados
            rows_after = big_df.shape[0]
            duplicates_removed = rows_before - rows_after
            
            # Guardar el DataFrame combinado
            global_file = os.path.join(output_folder_global, "df_players_info_global.csv")
            big_df.to_csv(global_file, index=False)
            
            print(f"\nArchivo global creado: {global_file}")
            print(f"Total de jugadoras en el archivo global: {rows_after}")
            print(f"Duplicados eliminados: {duplicates_removed}")
            

        else:
            print("No se pudieron encontrar DataFrames para concatenar")
    else:
        print("No se encontraron archivos CSV para combinar")
    
    
    print("Proceso completo.")

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
