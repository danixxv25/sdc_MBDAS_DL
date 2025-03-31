import streamlit as st
import pandas as pd
from utils import display_logo
from user_database import get_users, add_user, change_password

def run():
    st.title("Administración de Usuarios")
    
    if not st.session_state.get("authenticated", False) or st.session_state.get("role") != "admin":
        st.error("No tienes permiso para acceder a esta página")
        return
    
    display_logo()
    
    tab1, tab2, tab3 = st.tabs(["Lista de Usuarios", "Añadir Usuario", "Cambiar Contraseñas"])
    
    with tab1:
        st.subheader("Usuarios del Sistema")
        
        users = get_users()
        user_data = []
        
        for username, details in users.items():
            user_data.append({
                "Usuario": username,
                "Rol": details["role"],
                "Nombre Completo": details.get("nombre_completo", ""),
                "Equipo": details.get("equipo", "")
            })
        
        if user_data:
            df = pd.DataFrame(user_data)
            st.dataframe(df)
        else:
            st.info("No hay usuarios registrados")
    
    with tab2:
        st.subheader("Añadir Nuevo Usuario")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Nombre de Usuario", key="new_username")
            new_password = st.text_input("Contraseña", type="password", key="new_password")
            confirm_password = st.text_input("Confirmar Contraseña", type="password", key="confirm_password")
        
        with col2:
            nombre_completo = st.text_input("Nombre Completo", key="nombre_completo")
            equipo = st.text_input("Equipo", key="equipo")
            role = st.selectbox("Rol", ["usuario", "analista", "admin"], key="role")
        
        if st.button("Crear Usuario"):
            if not new_username or not new_password:
                st.error("El nombre de usuario y la contraseña son obligatorios")
            elif new_password != confirm_password:
                st.error("Las contraseñas no coinciden")
            else:
                success, message = add_user(new_username, new_password, role, nombre_completo, equipo)
                if success:
                    st.success(message)
                    # Limpia los campos
                    st.session_state.new_username = ""
                    st.session_state.new_password = ""
                    st.session_state.confirm_password = ""
                    st.session_state.nombre_completo = ""
                    st.session_state.equipo = ""
                    st.session_state.role = "usuario"
                else:
                    st.error(message)
    
    with tab3:
        st.subheader("Cambiar Contraseña")
        
        users = get_users()
        usernames = list(users.keys())
        
        selected_user = st.selectbox("Selecciona Usuario", usernames)
        
        if selected_user:
            new_password = st.text_input("Nueva Contraseña", type="password", key="admin_new_pass")
            confirm_new_password = st.text_input("Confirmar Nueva Contraseña", type="password", key="admin_confirm_pass")
            
            if st.button("Actualizar Contraseña"):
                if not new_password:
                    st.error("La contraseña no puede estar vacía")
                elif new_password != confirm_new_password:
                    st.error("Las contraseñas no coinciden")
                else:
                    # En este caso, el admin puede cambiar la contraseña sin conocer la actual
                    users[selected_user]["password"] = new_password
                    change_password(selected_user, "", new_password, is_admin=True)
                    st.success(f"Contraseña de {selected_user} actualizada correctamente")

if __name__ == "__main__":
    run()
