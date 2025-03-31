import streamlit as st
import pandas as pd
import datetime
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from user_activity_log import get_user_activities, ActivityType

# Configuración de alertas
ALERTS_CONFIG = "data/security_alerts_config.json"
ALERTS_LOG = "data/security_alerts_log.json"

def init_alerts_config():
    """Inicializa la configuración de alertas si no existe"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists(ALERTS_CONFIG):
        # Configuración predeterminada de alertas
        default_config = {
            "failed_login_threshold": 3,  # Número de intentos fallidos para generar alerta
            "failed_login_window": 10,    # Ventana de tiempo (minutos) para contar intentos
            "unusual_time_start": "23:00", # Hora de inicio fuera de horario normal
            "unusual_time_end": "06:00",   # Hora de fin fuera de horario normal
            "email_notifications": False,  # Enviar notificaciones por email
            "email_config": {
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": []
            },
            "monitor_page_access": True,   # Monitorear accesos a páginas restringidas
            "critical_pages": ["Administrar Usuarios", "Monitoreo de Actividad"]
        }
        
        # Guardar configuración
        with open(ALERTS_CONFIG, "w") as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    else:
        # Cargar configuración existente
        with open(ALERTS_CONFIG, "r") as f:
            return json.load(f)

def get_alerts_config():
    """Obtiene la configuración de alertas"""
    return init_alerts_config()

def save_alerts_config(config):
    """Guarda la configuración de alertas"""
    with open(ALERTS_CONFIG, "w") as f:
        json.dump(config, f, indent=4)
    return True

def init_alerts_log():
    """Inicializa el registro de alertas si no existe"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists(ALERTS_LOG):
        # Crear registro de alertas vacío
        alerts = []
        
        # Guardar en archivo JSON
        with open(ALERTS_LOG, "w") as f:
            json.dump(alerts, f, indent=4)
        
        return alerts
    else:
        # Leer alertas existentes
        with open(ALERTS_LOG, "r") as f:
            return json.load(f)

def log_security_alert(alert_type, details, severity="medium", related_user=None):
    """
    Registra una alerta de seguridad
    
    Parámetros:
    - alert_type: Tipo de alerta
    - details: Detalles de la alerta
    - severity: Nivel de severidad (low, medium, high, critical)
    - related_user: Usuario relacionado con la alerta
    """
    # Inicializar el registro si no existe
    alerts = init_alerts_log()
    
    # Obtener la fecha y hora actual
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Crear el registro de alerta
    alert = {
        "timestamp": timestamp,
        "type": alert_type,
        "details": details,
        "severity": severity,
        "related_user": related_user,
        "status": "new",
        "reviewed_by": None,
        "review_timestamp": None,
        "review_notes": None
    }
    
    # Añadir al registro
    alerts.append(alert)
    
    # Guardar el registro actualizado
    with open(ALERTS_LOG, "w") as f:
        json.dump(alerts, f, indent=4)
    
    # Enviar notificación por email si está configurado
    config = get_alerts_config()
    if config["email_notifications"] and severity in ["high", "critical"]:
        try:
            send_email_alert(alert)
        except Exception as e:
            print(f"Error al enviar notificación por email: {str(e)}")
    
    return True

def send_email_alert(alert):
    """Envía una notificación por email sobre una alerta de seguridad"""
    config = get_alerts_config()
    email_config = config["email_config"]
    
    if not email_config["smtp_server"] or not email_config["to_emails"]:
        return False
    
    # Crear mensaje
    msg = MIMEMultipart()
    msg["From"] = email_config["from_email"]
    msg["To"] = ", ".join(email_config["to_emails"])
    msg["Subject"] = f"[ALERTA DE SEGURIDAD] {alert['severity'].upper()}: {alert['type']}"
    
    # Cuerpo del mensaje
    body = f"""
    <html>
    <body>
        <h2>Alerta de Seguridad - {alert['severity'].upper()}</h2>
        <p><strong>Tipo:</strong> {alert['type']}</p>
        <p><strong>Fecha y Hora:</strong> {alert['timestamp']}</p>
        <p><strong>Usuario Relacionado:</strong> {alert['related_user'] or 'N/A'}</p>
        <p><strong>Detalles:</strong> {alert['details']}</p>
        <p>Por favor, revise esta alerta en el sistema de monitoreo.</p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, "html"))
    
    # Enviar mensaje
    try:
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["username"], email_config["password"])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error al enviar email: {str(e)}")
        return False

def check_failed_logins(username=None):
    """
    Verifica intentos fallidos de inicio de sesión para un usuario o todos
    y genera alertas si es necesario
    """
    config = get_alerts_config()
    threshold = config["failed_login_threshold"]
    window_minutes = config["failed_login_window"]
    
    # Calcular ventana de tiempo
    now = datetime.datetime.now()
    window_start = (now - datetime.timedelta(minutes=window_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Obtener intentos fallidos recientes
    activities = get_user_activities(
        username=username,
        action=ActivityType.FAILED_LOGIN,
        start_date=window_start[:10],  # Solo la fecha
        limit=100
    )
    
    # Filtrar por la hora exacta
    recent_failures = [a for a in activities if a["timestamp"] >= window_start]
    
    # Agrupar por usuario
    failures_by_user = {}
    for activity in recent_failures:
        user = activity["username"]
        if user not in failures_by_user:
            failures_by_user[user] = []
        failures_by_user[user].append(activity)
    
    # Verificar cada usuario
    for user, failures in failures_by_user.items():
        if len(failures) >= threshold:
            # Generar alerta
            ip_addresses = set(failure.get("ip_address", "desconocida") for failure in failures)
            
            log_security_alert(
                alert_type="multiple_failed_logins",
                details=f"{len(failures)} intentos fallidos en los últimos {window_minutes} minutos desde IP(s): {', '.join(ip_addresses)}",
                severity="high" if len(failures) >= threshold * 2 else "medium",
                related_user=user
            )
            
            return True
    
    return False

def check_unusual_time_access():
    """Verifica accesos fuera del horario normal"""
    config = get_alerts_config()
    unusual_start = config["unusual_time_start"]
    unusual_end = config["unusual_time_end"]
    
    # Obtener hora actual
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    
    # Verificar si estamos en horario inusual
    is_unusual_time = False
    if unusual_start > unusual_end:  # Cruza la medianoche
        is_unusual_time = current_time >= unusual_start or current_time <= unusual_end
    else:
        is_unusual_time = current_time >= unusual_start and current_time <= unusual_end
    
    if is_unusual_time and st.session_state.get("authenticated", False):
        username = st.session_state.get("username")
        
        # Evitar alertas duplicadas verificando si ya generamos una alerta para este usuario hoy
        today = now.strftime("%Y-%m-%d")
        alerts = init_alerts_log()
        today_alerts = [a for a in alerts if a["timestamp"].startswith(today) and 
                        a["type"] == "unusual_time_access" and 
                        a["related_user"] == username]
        
        if not today_alerts:
            log_security_alert(
                alert_type="unusual_time_access",
                details=f"Acceso al sistema fuera del horario normal ({current_time})",
                severity="low",
                related_user=username
            )
            return True
    
    return False

def get_security_alerts(status=None, severity=None, start_date=None, end_date=None, limit=100):
    """
    Obtiene alertas de seguridad con filtros opcionales
    
    Parámetros:
    - status: Filtrar por estado (new, reviewed, false_positive)
    - severity: Filtrar por severidad (low, medium, high, critical)
    - start_date: Fecha de inicio (formato: "YYYY-MM-DD")
    - end_date: Fecha de fin (formato: "YYYY-MM-DD")
    - limit: Número máximo de alertas a devolver
    """
    alerts = init_alerts_log()
    
    # Convertir fechas de string a datetime si se proporcionan
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_datetime = datetime.datetime(1900, 1, 1)
    
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        # Añadir un día para incluir todo el día final
        end_datetime = end_datetime + datetime.timedelta(days=1)
    else:
        end_datetime = datetime.datetime.now() + datetime.timedelta(days=1)
    
    # Filtrar alertas
    filtered_alerts = []
    for alert in alerts:
        # Convertir timestamp a datetime
        alert_datetime = datetime.datetime.strptime(alert["timestamp"], "%Y-%m-%d %H:%M:%S")
        
        # Verificar criterios de filtro
        if status and alert["status"] != status:
            continue
        
        if severity and alert["severity"] != severity:
            continue
        
        if alert_datetime < start_datetime or alert_datetime > end_datetime:
            continue
        
        filtered_alerts.append(alert)
    
    # Ordenar por timestamp (más reciente primero)
    filtered_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Limitar número de resultados
    return filtered_alerts[:limit]

def update_alert_status(alert_timestamp, new_status, review_notes=None):
    """
    Actualiza el estado de una alerta de seguridad
    
    Parámetros:
    - alert_timestamp: Timestamp de la alerta a actualizar
    - new_status: Nuevo estado (reviewed, false_positive)
    - review_notes: Notas de revisión
    """
    alerts = init_alerts_log()
    username = st.session_state.get("username", "sistema")
    
    # Buscar la alerta por timestamp
    for alert in alerts:
        if alert["timestamp"] == alert_timestamp:
            alert["status"] = new_status
            alert["reviewed_by"] = username
            alert["review_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert["review_notes"] = review_notes
            
            # Guardar cambios
            with open(ALERTS_LOG, "w") as f:
                json.dump(alerts, f, indent=4)
            
            return True
    
    return False

def display_security_alerts_dashboard():
    """Muestra un dashboard con alertas de seguridad"""
    st.subheader("Dashboard de Alertas de Seguridad")
    
    # Verificar alertas activas
    alerts = get_security_alerts(status="new")
    
    # Mostrar resumen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alertas Nuevas", len([a for a in alerts if a["status"] == "new"]))
    
    with col2:
        st.metric("Alertas Críticas", len([a for a in alerts if a["severity"] == "critical" and a["status"] == "new"]))
    
    with col3:
        st.metric("Alertas Altas", len([a for a in alerts if a["severity"] == "high" and a["status"] == "new"]))
    
    with col4:
        total_alerts = len(init_alerts_log())
        st.metric("Total Histórico", total_alerts)
    
    # Mostrar alertas activas
    if alerts:
        st.subheader("Alertas Activas")
        
        for i, alert in enumerate(alerts):
            severity_color = {
                "low": "blue",
                "medium": "orange",
                "high": "red",
                "critical": "darkred"
            }.get(alert["severity"], "gray")
            
            # Crear un expander para cada alerta
            with st.expander(f"{alert['timestamp']} - {alert['type'].upper()} [{alert['severity'].upper()}]"):
                st.markdown(f"<p style='color:{severity_color};'>Severidad: {alert['severity'].upper()}</p>", unsafe_allow_html=True)
                st.write(f"**Detalles:** {alert['details']}")
                st.write(f"**Usuario relacionado:** {alert['related_user'] or 'N/A'}")
                
                # Opciones para actualizar el estado
                col1, col2 = st.columns(2)
                
                with col1:
                    notes = st.text_area(f"Notas de revisión #{i}", key=f"notes_{i}")
                
                with col2:
                    mark_reviewed = st.button("Marcar como Revisada", key=f"reviewed_{i}")
                    mark_false = st.button("Falso Positivo", key=f"false_{i}")
                
                if mark_reviewed:
                    update_alert_status(alert["timestamp"], "reviewed", notes)
                    st.success("Alerta marcada como revisada")
                    st.rerun()
                
                if mark_false:
                    update_alert_status(alert["timestamp"], "false_positive", notes)
                    st.success("Alerta marcada como falso positivo")
                    st.rerun()
    else:
        st.info("No hay alertas activas en este momento")
    
    # Configuración de alertas
    with st.expander("Configuración de Alertas de Seguridad"):
        config = get_alerts_config()
        
        st.subheader("Umbrales de Alerta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            failed_threshold = st.number_input("Umbral de intentos fallidos", 
                                              min_value=1, max_value=10, 
                                              value=config["failed_login_threshold"])
        
        with col2:
            failed_window = st.number_input("Ventana de tiempo (minutos)", 
                                           min_value=1, max_value=60, 
                                           value=config["failed_login_window"])
        
        st.subheader("Horario Normal de Acceso")
        
        col1, col2 = st.columns(2)
        
        with col1:
            unusual_start = st.time_input("Inicio de horario inusual", 
                                         datetime.time(int(config["unusual_time_start"].split(":")[0]), 
                                                     int(config["unusual_time_start"].split(":")[1])))
        
        with col2:
            unusual_end = st.time_input("Fin de horario inusual", 
                                       datetime.time(int(config["unusual_time_end"].split(":")[0]), 
                                                   int(config["unusual_time_end"].split(":")[1])))
        
        st.subheader("Notificaciones por Email")
        
        email_enabled = st.checkbox("Habilitar notificaciones por email", 
                                  value=config["email_notifications"])
        
        if email_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                smtp_server = st.text_input("Servidor SMTP", 
                                          value=config["email_config"]["smtp_server"])
                smtp_port = st.number_input("Puerto SMTP", 
                                          min_value=1, max_value=65535, 
                                          value=config["email_config"]["smtp_port"])
                from_email = st.text_input("Email remitente", 
                                         value=config["email_config"]["from_email"])
            
            with col2:
                username = st.text_input("Usuario SMTP", 
                                       value=config["email_config"]["username"])
                password = st.text_input("Contraseña SMTP", 
                                       value=config["email_config"]["password"], 
                                       type="password")
                to_emails = st.text_area("Destinatarios (uno por línea)", 
                                       value="\n".join(config["email_config"]["to_emails"]))
                to_emails_list = [email.strip() for email in to_emails.split("\n") if email.strip()]
        
        save_button = st.button("Guardar Configuración")
        
        if save_button:
            # Actualizar configuración
            config["failed_login_threshold"] = failed_threshold
            config["failed_login_window"] = failed_window
            config["unusual_time_start"] = unusual_start.strftime("%H:%M")
            config["unusual_time_end"] = unusual_end.strftime("%H:%M")
            config["email_notifications"] = email_enabled
            
            if email_enabled:
                config["email_config"]["smtp_server"] = smtp_server
                config["email_config"]["smtp_port"] = smtp_port
                config["email_config"]["username"] = username
                config["email_config"]["password"] = password
                config["email_config"]["from_email"] = from_email
                config["email_config"]["to_emails"] = to_emails_list
            
            save_alerts_config(config)
            st.success("Configuración guardada correctamente")

def run_security_checks():
    """Ejecuta verificaciones de seguridad automáticas"""
    # Verificar intentos fallidos de inicio de sesión
    check_failed_logins()
    
    # Verificar accesos fuera de horario normal
    check_unusual_time_access()
