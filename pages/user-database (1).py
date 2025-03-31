import pandas as pd
import hashlib
import streamlit as st
import os
import json
import sys

# Importamos el módulo de registro de actividad
try:
    from user_activity_log import log_activity, ActivityType
except ImportError:
    # Para manejar el caso de importación circular durante la inicialización
    log_activity = lambda *args, **kwargs: None
    class ActivityType:
        USER_CREATED = "user_created"
        PASSWORD_CHANGED = "password_changed"
        USER_MODIFIED = "user_modified"

# Archivo para almacenar usuarios (en producción usaría una base de datos real)
USERS_DB = "data/users.json"

def init_user_database():
    """Inicializa la base de datos de usuarios si no existe"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists(USERS_DB):
        # Crea usuarios predeterminados
        default_users = {
            "admin": {
                "password": hashlib.sha256("admin".encode()).hexdigest(),
                "role": "admin",
                "nombre_completo": "Administrador del Sistema",
                "equipo": "Staff Técnico"
            },
            "usuario": {
                "password": hashlib.sha256("1234".encode()).hexdigest(),
                "role": "usuario",
                "nombre_completo": "Usuario Estándar",
                "equipo": "Análisis de Datos"
            },
            "analista": {
                "password": hashlib.sha256("analista123".encode()).hexdigest(),
                "role": "analista",
                "nombre_completo": "Analista Deportivo",
                "equipo": "Scouting"
            }
        }
        
        # Guarda en archivo JSON
        with open(USERS_DB, "w") as f:
            json.dump(default_users, f, indent=4)
        
        return default_users
    else:
        # Lee usuarios existentes
        with open(USERS_DB, "r") as f:
            return json.load(f)

def get_users():
    """Obtiene la lista de usuarios"""
    if not os.path.exists(USERS_DB):
        return init_user_database()
    
    with open(USERS_DB, "r") as f:
        return json.load(f)

def add_user(username, password, role="usuario", nombre_completo="", equipo=""):
    """Añade un nuevo usuario"""
    users = get_users()
    
    if username in users:
        return False, "El nombre de usuario ya existe"
    
    users[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "role": role,
        "nombre_completo": nombre_completo,
        "equipo": equipo
    }
    
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=4)
    
    # Registrar la actividad de creación de usuario
    creator = st.session_state.get("username", "sistema")
    log_activity(
        username=creator,
        action=ActivityType.USER_CREATED,
        details=f"Creó el usuario '{username}' con rol '{role}'"
    )
    
    return True, "Usuario creado correctamente"

def verify_credentials(username, password):
    """Verifica las credenciales de un usuario"""
    users = get_users()
    
    if username not in users:
        return False, None
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if hashed_password == users[username]["password"]:
        return True, users[username]
    
    return False, None

def change_password(username, current_password, new_password, is_admin=False):
    """
    Cambia la contraseña de un usuario
    
    Parámetros:
    - username: Nombre de usuario
    - current_password: Contraseña actual
    - new_password: Nueva contraseña
    - is_admin: Si True, permite el cambio sin verificar la contraseña actual (solo para admins)
    """
    users = get_users()
    
    if username not in users:
        return False, "Usuario no encontrado"
    
    # Verificar contraseña actual (a menos que sea admin)
    if not is_admin:
        hashed_current = hashlib.sha256(current_password.encode()).hexdigest()
        
        if hashed_current != users[username]["password"]:
            return False, "Contraseña actual incorrecta"
    
    # Actualizar contraseña
    users[username]["password"] = hashlib.sha256(new_password.encode()).hexdigest()
    
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=4)
    
    # Registrar actividad
    modifier = st.session_state.get("username", "sistema")
    action_details = f"Cambió la contraseña del usuario '{username}'"
    
    if is_admin and modifier != username:
        action_details += f" (realizado por admin '{modifier}')"
    
    log_activity(
        username=modifier,
        action=ActivityType.PASSWORD_CHANGED,
        details=action_details
    )
    
    return True, "Contraseña actualizada correctamente"

def get_user_permissions(role):
    """Obtiene los permisos según el rol del usuario"""
    permissions = {
        "admin": {
            "pages": ["Inicio", "Equipo Propio", "Buscar Jugadoras", "Comparar Jugadoras", 
                      "Reemplazar Jugadoras", "Talentos emergentes", "Administrar Usuarios",
                      "Monitoreo de Actividad", "Alertas de Seguridad", "Respaldos y Datos"],
            "can_edit": True,
            "can_add_users": True,
            "can_export_data": True
        },
        "analista": {
            "pages": ["Inicio", "Equipo Propio", "Buscar Jugadoras", "Comparar Jugadoras", 
                      "Reemplazar Jugadoras", "Talentos emergentes"],
            "can_edit": True,
            "can_add_users": False,
            "can_export_data": True
        },
        "usuario": {
            "pages": ["Inicio", "Equipo Propio", "Buscar Jugadoras", "Comparar Jugadoras"],
            "can_edit": False,
            "can_add_users": False,
            "can_export_data": False
        }
    }
    
    return permissions.get(role, permissions["usuario"])
