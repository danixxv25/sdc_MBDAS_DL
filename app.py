import streamlit as st
from utils import display_logo
from user_database import verify_credentials, get_user_permissions, init_user_database
from user_activity_log import log_activity, ActivityType
from security_alerts import run_security_checks

# Asegurarse de que la base de datos de usuarios exista
init_user_database()

# Configuración de la página
st.set_page_config(page_title="Análisis de Jugadoras", layout="wide")

# Inicializar estado de sesión para autenticación
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.user_data = None

# Función para mostrar página de login
def show_login():
    display_logo()
    st.title("Bienvenido al Sistema de Análisis de Jugadoras")
    
    st.markdown("---")
    st.subheader("Iniciar Sesión")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        login_button = st.button("Ingresar")
        
        if login_button:
            if not username or not password:
                st.error("Por favor ingresa usuario y contraseña")
            else:
                success, user_data = verify_credentials(username, password)
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = user_data["role"]
                    st.session_state.user_data = user_data
                    
                    # Registrar actividad de inicio de sesión exitoso
                    ip = st.session_state.get("client_ip", "desconocida")
                    log_activity(
                        username=username,
                        action=ActivityType.LOGIN,
                        details=f"Inicio de sesión exitoso desde IP: {ip}",
                        ip_address=ip
                    )
                    
                    st.success("Inicio de sesión exitoso. Redirigiendo...")
                    st.rerun()
                else:
                    # Registrar intento fallido de inicio de sesión
                    ip = st.session_state.get("client_ip", "desconocida")
                    log_activity(
                        username=username, 
                        action=ActivityType.FAILED_LOGIN,
                        details=f"Intento fallido de inicio de sesión desde IP: {ip}",
                        ip_address=ip
                    )
                    st.error("Usuario o contraseña incorrectos")

# Si no está autenticado, mostrar pantalla de login
if not st.session_state.authenticated:
    show_login()
else:
    # Si está autenticado, mostrar la aplicación con las páginas permitidas
    # Obtener los permisos según el rol
    permissions = get_user_permissions(st.session_state.role)
    allowed_pages = permissions["pages"]
    
    # Definir todas las páginas
    main_page = st.Page("pages/Main_page.py", title="Inicio", icon="🏠")
    page_1 = st.Page("pages/Equipo_propio.py", title="Equipo Propio", icon="🇵🇪")
    page_2 = st.Page("pages/Buscar_jugadoras.py", title="Buscar Jugadoras", icon="🔎")
    page_3 = st.Page("pages/Comparar_jugadoras completo.py", title="Comparar Jugadoras", icon="📊")
    page_4 = st.Page("pages/Reemplazar_jugadora final.py", title="Reemplazar Jugadoras", icon="🔄")
    page_5 = st.Page("pages/Talentos_emergentes.py", title="Talentos emergentes", icon="🌟")
    admin_page = st.Page("pages/admin-page.py", title="Administrar Usuarios", icon="👤")
    activity_page = st.Page("pages/activity-monitor-page.py", title="Monitoreo de Actividad", icon="📈")
    security_page = st.Page("pages/security-alerts.py", title="Alertas de Seguridad", icon="🔒")
    backup_page = st.Page("pages/Gestion_datos.py", title="Respaldos y Datos", icon="💾")
    
    # Filtrar las páginas según los permisos
    available_pages = []
    if "Inicio" in allowed_pages:
        available_pages.append(main_page)
    if "Equipo Propio" in allowed_pages:
        available_pages.append(page_1)
    if "Buscar Jugadoras" in allowed_pages:
        available_pages.append(page_2)
    if "Comparar Jugadoras" in allowed_pages:
        available_pages.append(page_3)
    if "Reemplazar Jugadoras" in allowed_pages:
        available_pages.append(page_4)
    if "Talentos emergentes" in allowed_pages:
        available_pages.append(page_5)
    if "Administrar Usuarios" in allowed_pages:
        available_pages.append(admin_page)
    if "Monitoreo de Actividad" in allowed_pages and st.session_state.role == "admin":
        available_pages.append(activity_page)
    if "Alertas de Seguridad" in allowed_pages and st.session_state.role == "admin":
        available_pages.append(security_page)
    if "Respaldos y Datos" in allowed_pages and st.session_state.role == "admin":
        available_pages.append(backup_page)
    
    # Barra lateral con información de usuario y opción de cerrar sesión
    with st.sidebar:
        st.write(f"**Usuario:** {st.session_state.username}")
        st.write(f"**Rol:** {st.session_state.role}")
        
        if "nombre_completo" in st.session_state.user_data:
            st.write(f"**Nombre:** {st.session_state.user_data['nombre_completo']}")
        
        if "equipo" in st.session_state.user_data:
            st.write(f"**Equipo:** {st.session_state.user_data['equipo']}")
        
        st.markdown("---")
        
        if st.button("Cerrar Sesión"):
            # Registrar actividad de cierre de sesión
            log_activity(
                username=st.session_state.username,
                action=ActivityType.LOGOUT,
                details="Cierre de sesión manual"
            )
            
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.user_data = None
            st.rerun()
    
    # Si está autenticado, ejecutar verificaciones de seguridad
    if st.session_state.authenticated:
        run_security_checks()
        
    # Capturar la página actual para registro de actividad
    current_page = st.query_params.get("page", None)
    
    # Si hay un cambio de página, registrar la actividad
    if "last_page" not in st.session_state:
        st.session_state.last_page = current_page
    
    if current_page and current_page != st.session_state.last_page:
        # Obtener el título de la página actual
        page_title = None
        for page in available_pages:
            if page.path.endswith(f"/{current_page}.py"):
                page_title = page.title
                break
        
        # Registrar la visita a la página
        if page_title:
            log_activity(
                username=st.session_state.username,
                action=ActivityType.PAGE_VISIT,
                details=f"Visitó la página: {page_title}"
            )
        
        # Actualizar la última página visitada
        st.session_state.last_page = current_page
    
    # Ejecutar la navegación con las páginas permitidas
    pg = st.navigation(available_pages)
    pg.run()
