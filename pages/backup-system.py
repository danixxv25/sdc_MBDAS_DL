import streamlit as st
import pandas as pd
import datetime
import json
import os
import zipfile
import io
import base64
from user_activity_log import get_user_activities, init_activity_log
from user_database import get_users
from security_alerts import get_security_alerts, init_alerts_log

def export_all_data():
    """
    Exporta todos los datos del sistema: usuarios, actividad y alertas
    
    Retorna:
    - BytesIO con archivo ZIP que contiene todos los datos en formato JSON y CSV
    """
    # Crear un buffer en memoria para el archivo ZIP
    zip_buffer = io.BytesIO()
    
    # Crear archivo ZIP
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Exportar usuarios
        users = get_users()
        # Eliminar contraseñas por seguridad
        for user in users.values():
            if "password" in user:
                user["password"] = "********"
        
        users_json = json.dumps(users, indent=4)
        zip_file.writestr("users.json", users_json)
        
        # Convertir a DataFrame para CSV
        users_list = []
        for username, data in users.items():
            user_data = {"username": username}
            user_data.update(data)
            users_list.append(user_data)
        
        users_df = pd.DataFrame(users_list)
        zip_file.writestr("users.csv", users_df.to_csv(index=False))
        
        # Exportar actividad
        activities = init_activity_log()
        activities_json = json.dumps(activities, indent=4)
        zip_file.writestr("user_activity.json", activities_json)
        
        # Convertir a DataFrame para CSV
        activities_df = pd.DataFrame(activities)
        zip_file.writestr("user_activity.csv", activities_df.to_csv(index=False))
        
        # Exportar alertas
        alerts = init_alerts_log()
        alerts_json = json.dumps(alerts, indent=4)
        zip_file.writestr("security_alerts.json", alerts_json)
        
        # Convertir a DataFrame para CSV
        alerts_df = pd.DataFrame(alerts)
        zip_file.writestr("security_alerts.csv", alerts_df.to_csv(index=False))
        
        # Crear un archivo README
        readme = f"""
        # Respaldo del Sistema de Análisis de Jugadoras
        
        Fecha de exportación: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Exportado por: {st.session_state.get("username", "sistema")}
        
        ## Contenido:
        
        - users.json / users.csv: Lista de usuarios del sistema (sin contraseñas)
        - user_activity.json / user_activity.csv: Registro de actividad de usuarios
        - security_alerts.json / security_alerts.csv: Registro de alertas de seguridad
        
        ## Estadísticas:
        
        - Total de usuarios: {len(users)}
        - Total de registros de actividad: {len(activities)}
        - Total de alertas de seguridad: {len(alerts)}
        """
        
        zip_file.writestr("README.md", readme)
    
    # Volver al inicio del buffer
    zip_buffer.seek(0)
    
    # Registrar actividad
    from user_activity_log import log_activity, ActivityType
    log_activity(
        username=st.session_state.get("username", "sistema"),
        action=ActivityType.DATA_EXPORT,
        details="Exportó una copia de seguridad completa del sistema"
    )
    
    return zip_buffer

def generate_backup_link():
    """Genera un enlace para descargar una copia de seguridad completa"""
    try:
        zip_buffer = export_all_data()
        
        # Generar nombre de archivo con fecha
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"respaldo_sistema_{now}.zip"
        
        # Convertir a Base64 para el enlace de descarga
        b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        
        # Crear enlace de descarga
        href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Descargar respaldo completo</a>'
        
        return True, href
    except Exception as e:
        return False, str(e)

def purge_old_records(days_to_keep=180):
    """
    Elimina registros de actividad y alertas más antiguos que el número de días especificado
    
    Parámetros:
    - days_to_keep: Número de días a mantener (los registros más antiguos se eliminarán)
    
    Retorna:
    - Tupla (éxito, mensaje, registros_eliminados)
    """
    try:
        # Calcular fecha límite
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
        
        # Obtener registros de actividad
        activities = init_activity_log()
        
        # Filtrar por fecha
        new_activities = [a for a in activities if a["timestamp"][:10] >= cutoff_date]
        records_removed = len(activities) - len(new_activities)
        
        # Guardar nuevos registros
        with open("data/user_activity_log.json", "w") as f:
            json.dump(new_activities, f, indent=4)
        
        # Obtener alertas
        alerts = init_alerts_log()
        
        # Filtrar por fecha
        new_alerts = [a for a in alerts if a["timestamp"][:10] >= cutoff_date]
        alerts_removed = len(alerts) - len(new_alerts)
        
        # Guardar nuevas alertas
        with open("data/security_alerts_log.json", "w") as f:
            json.dump(new_alerts, f, indent=4)
        
        # Registrar actividad
        from user_activity_log import log_activity, ActivityType
        log_activity(
            username=st.session_state.get("username", "sistema"),
            action="data_purge",
            details=f"Eliminó {records_removed} registros de actividad y {alerts_removed} alertas antiguas (> {days_to_keep} días)"
        )
        
        return True, f"Se eliminaron {records_removed} registros de actividad y {alerts_removed} alertas antiguas", records_removed + alerts_removed
    except Exception as e:
        return False, f"Error al purgar registros: {str(e)}", 0

def display_backup_dashboard():
    """Muestra un dashboard para gestionar copias de seguridad y eliminación de datos antiguos"""
    st.subheader("Gestión de Datos del Sistema")
    
    tab1, tab2 = st.tabs(["Copias de Seguridad", "Mantenimiento de Datos"])
    
    with tab1:
        st.write("Crea y descarga una copia de seguridad completa del sistema, incluyendo usuarios, registros de actividad y alertas.")
        
        # Información sobre los datos
        users = get_users()
        activities = init_activity_log()
        alerts = init_alerts_log()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Usuarios", len(users))
        
        with col2:
            st.metric("Registros de Actividad", len(activities))
        
        with col3:
            st.metric("Alertas de Seguridad", len(alerts))
        
        # Fechas extremas
        if activities:
            oldest_activity = min(activities, key=lambda x: x["timestamp"])["timestamp"]
            newest_activity = max(activities, key=lambda x: x["timestamp"])["timestamp"]
            
            st.write(f"Rango de fechas de registros: {oldest_activity[:10]} a {newest_activity[:10]}")
        
        # Botón para generar respaldo
        if st.button("Generar Copia de Seguridad"):
            success, result = generate_backup_link()
            
            if success:
                st.markdown(result, unsafe_allow_html=True)
                st.success("Copia de seguridad generada correctamente")
            else:
                st.error(f"Error al generar copia de seguridad: {result}")
    
    with tab2:
        st.write("Elimina registros antiguos para mantener el sistema optimizado.")
        
        days_to_keep = st.slider("Días a mantener", min_value=30, max_value=365, value=180, step=30)
        
        # Calcular fecha límite
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
        
        st.write(f"Se eliminarán todos los registros anteriores a: **{cutoff_date}**")
        
        # Calcular cuántos registros se eliminarían
        activities = init_activity_log()
        alerts = init_alerts_log()
        
        activities_to_remove = [a for a in activities if a["timestamp"][:10] < cutoff_date]
        alerts_to_remove = [a for a in alerts if a["timestamp"][:10] < cutoff_date]
        
        st.write(f"Se eliminarían aproximadamente:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Registros de Actividad", len(activities_to_remove))
        
        with col2:
            st.metric("Alertas de Seguridad", len(alerts_to_remove))
        
        # Advertencia
        st.warning("⚠️ Se recomienda crear una copia de seguridad antes de eliminar registros antiguos.")
        
        # Confirmación
        confirm = st.checkbox("Confirmo que quiero eliminar estos registros antiguos")
        
        if confirm:
            if st.button("Eliminar Registros Antiguos"):
                success, message, count = purge_old_records(days_to_keep)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)

def run():
    st.title("Gestión de Datos y Respaldos")
    
    if not st.session_state.get("authenticated", False) or st.session_state.get("role") != "admin":
        st.error("No tienes permiso para acceder a esta página")
        return
    
    # Registrar visita a esta página
    from user_activity_log import log_activity, ActivityType
    log_activity(
        username=st.session_state.username,
        action=ActivityType.PAGE_VISIT,
        details="Visitó la página de gestión de datos y respaldos"
    )
    
    display_backup_dashboard()

if __name__ == "__main__":
    run()
