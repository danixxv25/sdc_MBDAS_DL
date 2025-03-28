import streamlit as st
from utils import display_logo

# Configuración de la página
display_logo(360)

st.title("Club Atlético de Madrid")
st.subheader("Aplicación de Benchmarking y Análisis de Jugadoras")
st.write("Seleccione una herramienta de la barra lateral para navegar.")

col1, col2, col3, col4, col5  = st.columns([1,1,1,1,1])
with col1:
    st.image("media/logos/leagues/BWSL_MASTER_HORIZONTAL_BADGE_RGB.png", width=100)
with col2:
    st.image("media/logos/leagues/FIGC_Serie_A_femminile_(2020).svg.png", width=75)
with col3:
    st.image("media/logos/leagues/Google_Pixel_Frauen-Bundesliga_Wordmark.svg.png", width=100)
with col4:
    st.image("media/logos/leagues/liga-f-seeklogo.png", width=100)
with col5:
    st.image("media/logos/leagues/Première_Ligue.png", width=75)

    