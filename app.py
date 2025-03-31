import streamlit as st
from utils import display_logo

st.set_page_config(page_title="Análisis de Jugadoras", layout="wide")

main_page = st.Page("pages/Main_page.py", title="Inicio", icon="🏠")
page_1= st.Page("pages/Equipo_propio.py", title="Equipo Propio", icon="🇵🇪")
page_2= st.Page("pages/Buscar_jugadoras.py", title="Buscar Jugadoras", icon="🔎")
page_3 = st.Page("pages/Comparar_jugadoras completo.py", title="Comparar Jugadoras", icon="📊")
page_4 = st.Page("pages/Reemplazar_jugadora final.py", title="Reemplazar Jugadoras", icon="🔄")
page_5 = st.Page("pages/Talentos_emergentes.py", title="Talentos emergentes", icon="🌟")

pg = st.navigation([main_page, page_1, page_2, page_3, page_4])
pg.run()





