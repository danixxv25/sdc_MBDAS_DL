import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from utils import display_logo
from user_activity_log import get_user_activities, display_activity_dashboard, ActivityType

def run():
    st.title("Monitoreo de Actividad de Usuarios")
    
    if not st.session_state.get("authenticated", False) or st.session_state.get("role") != "admin":
        st.error("No tienes permiso para acceder a esta página")
        return
    
    display_logo()
    
    # Registrar visita a esta página
    from user_activity_log import log_activity
    log_activity(
        username=st.session_state.username,
        action=ActivityType.PAGE_VISIT,
        details="Visitó la página de monitoreo de actividad"
    )
    
    tab1, tab2 = st.tabs(["Dashboard", "Búsqueda Avanzada"])
    
    with tab1:
        display_activity_dashboard()
    
    with tab2:
        st.subheader("Búsqueda Avanzada de Actividades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Obtener lista de usuarios únicos
            all_activities = get_user_activities(limit=10000)
            unique_users = list(set(activity["username"] for activity in all_activities))
            unique_users.insert(0, "Todos")
            
            selected_user = st.selectbox("Usuario", unique_users)
        
        with col2:
            # Obtener tipos de actividades únicas
            unique_actions = list(set(activity["action"] for activity in all_activities))
            unique_actions.insert(0, "Todas")
            
            selected_action = st.selectbox("Tipo de Actividad", unique_actions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Fecha de inicio", 
                                      value=datetime.datetime.now() - datetime.timedelta(days=7))
        
        with col2:
            end_date = st.date_input("Fecha de fin", 
                                    value=datetime.datetime.now())
        
        with col3:
            limit = st.number_input("Número máximo de registros", min_value=10, max_value=1000, value=100)
        
        search_button = st.button("Buscar")
        
        if search_button:
            username = selected_user if selected_user != "Todos" else None
            action = selected_action if selected_action != "Todas" else None
            
            # Obtener actividades filtradas
            filtered_activities = get_user_activities(
                username=username,
                action=action,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                limit=limit
            )
            
            if filtered_activities:
                # Convertir a DataFrame para visualización
                df = pd.DataFrame(filtered_activities)
                
                # Mostrar resultados
                st.subheader(f"Resultados ({len(filtered_activities)} registros)")
                st.dataframe(df)
                
                # Opción para descargar como CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Descargar Resultados como CSV",
                    data=csv,
                    file_name="actividades_filtradas.csv",
                    mime="text/csv"
                )
                
                # Visualizaciones
                if len(filtered_activities) > 1:
                    st.subheader("Visualizaciones")
                    
                    # Contar actividades por tipo
                    activity_counts = df["action"].value_counts().reset_index()
                    activity_counts.columns = ["Tipo de Actividad", "Cantidad"]
                    
                    fig1 = px.pie(activity_counts, values="Cantidad", names="Tipo de Actividad", 
                                 title="Distribución de Tipos de Actividad")
                    st.plotly_chart(fig1)
                    
                    # Actividad a lo largo del tiempo
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["fecha"] = df["timestamp"].dt.date
                    
                    activity_over_time = df.groupby("fecha").size().reset_index(name="cantidad")
                    
                    fig2 = px.line(activity_over_time, x="fecha", y="cantidad", 
                                  title="Actividad a lo largo del tiempo")
                    st.plotly_chart(fig2)
                    
                    # Si hay más de un usuario
                    if username is None and len(df["username"].unique()) > 1:
                        user_activity = df["username"].value_counts().reset_index()
                        user_activity.columns = ["Usuario", "Actividades"]
                        
                        fig3 = px.bar(user_activity, x="Usuario", y="Actividades", 
                                     title="Actividad por Usuario")
                        st.plotly_chart(fig3)
            else:
                st.info("No se encontraron registros con los filtros seleccionados")

if __name__ == "__main__":
    run()
