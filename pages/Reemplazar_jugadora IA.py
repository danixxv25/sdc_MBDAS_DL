import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import time

# Modificar la sección donde se crean las pestañas
# Cambiar esto:
# tab1, tab2, tab3 = st.tabs(["Información Básica", "Clustering y Radar", "Comparativa de Métricas"])

# Por esto:
tab1, tab2, tab3, tab4 = st.tabs(["Información Básica", "Clustering y Radar", "Comparativa de Métricas", "Análisis IA"])

# Y añadir el código para la nueva pestaña después de las otras pestañas:

# Pestaña 4: Análisis IA con DAFO y recomendaciones
with tab4:
    st.header(f"Análisis IA para {jugadora_seleccionada}")
    st.subheader("Informe generado por IA")
    
    # Contenedor para el análisis DAFO
    dafo_container = st.container()
    
    with dafo_container:
        st.write("### Análisis DAFO")
        
        # Función para generar el análisis DAFO basado en los datos y métricas
        def generar_dafo(jugadora, posicion, metricas_jugadora, metricas_similares, metricas_promedio):
            """
            Genera un análisis DAFO para la jugadora basado en sus métricas y comparándola con jugadoras similares.
            
            Args:
                jugadora: Nombre de la jugadora
                posicion: Posición de la jugadora
                metricas_jugadora: Dict con las métricas de la jugadora
                metricas_similares: Dict con las métricas de jugadoras similares
                metricas_promedio: Dict con los promedios por posición
            
            Returns:
                Dict con el análisis DAFO (debilidades, amenazas, fortalezas, oportunidades)
            """
            fortalezas = []
            debilidades = []
            oportunidades = []
            amenazas = []
            
            # Creamos un diccionario para mapear las métricas clave por posición
            metricas_importantes = {
                'GK': {
                    'defensivas': ['Save%', 'CS%', 'PSxG-GA', 'Stp%'], 
                    'tecnicas': ['Pass_Cmp_+40y%', '#OPA/90'],
                    'fisicas': ['AvgDist']
                },
                'DF': {
                    'defensivas': ['Tkl%', 'Blocks', 'Int', 'Tkl+Int', 'Recov'], 
                    'tecnicas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'TotDist'],
                    'fisicas': ['touch_Def 3rd', 'touch_Mid 3rd']
                },
                'MF': {
                    'defensivas': ['Tkl%', 'Int', 'Recov'], 
                    'ofensivas': ['G+A', 'Ast', 'SCA90', 'GCA90', 'KP'],
                    'tecnicas': ['Cmp%_short', 'Cmp%_med', 'Cmp%_long', 'PPA', 'pass_1/3'],
                    'fisicas': ['touch_Mid 3rd', 'touch_Att 3rd', 'PrgR']
                },
                'FW': {
                    'defensivas': ['Recov'], 
                    'ofensivas': ['Gls', 'G+A', 'SoT/90', 'G/Sh', 'xG', 'G-xG'],
                    'tecnicas': ['TO_Succ%', 'KP', 'SCA90', 'GCA90'],
                    'fisicas': ['touch_Att 3rd', 'touch_Att Pen', 'PrgR']
                }
            }
            
            # Verificamos si la posición existe en nuestro mapeo
            if posicion in metricas_importantes:
                categorias = metricas_importantes[posicion]
                
                # Analizamos fortalezas y debilidades
                for categoria, metricas in categorias.items():
                    for metrica in metricas:
                        if metrica in metricas_jugadora and metrica in metricas_promedio:
                            # Obtenemos el valor de la jugadora y el promedio del equipo
                            valor_jugadora = metricas_jugadora.get(metrica)
                            promedio_posicion = metricas_promedio.get(metrica)
                            
                            if pd.notna(valor_jugadora) and pd.notna(promedio_posicion) and promedio_posicion > 0:
                                # Calculamos el porcentaje de diferencia
                                diff_porcentaje = ((valor_jugadora - promedio_posicion) / promedio_posicion) * 100
                                metrica_nombre = metric_display_names.get(metrica, metrica)
                                
                                # Determinamos si es una fortaleza o debilidad
                                if diff_porcentaje >= 15:  # 15% mejor que el promedio
                                    fortalezas.append(f"**{metrica_nombre}**: Destaca con un {valor_jugadora:.2f} (un {abs(diff_porcentaje):.1f}% superior al promedio de su posición)")
                                elif diff_porcentaje <= -15:  # 15% peor que el promedio
                                    debilidades.append(f"**{metrica_nombre}**: Por debajo con un {valor_jugadora:.2f} (un {abs(diff_porcentaje):.1f}% inferior al promedio de su posición)")
                
                # Analizamos oportunidades y amenazas basadas en comparaciones y tendencias
                for categoria, metricas in categorias.items():
                    for metrica in metricas:
                        if metrica in metricas_jugadora and metrica in metricas_similares:
                            valor_jugadora = metricas_jugadora.get(metrica)
                            mejor_similar = max(jugadora_similar.get(metrica, 0) for jugadora_similar in metricas_similares)
                            
                            if pd.notna(valor_jugadora) and pd.notna(mejor_similar) and mejor_similar > 0:
                                diff_porcentaje = ((mejor_similar - valor_jugadora) / valor_jugadora) * 100
                                metrica_nombre = metric_display_names.get(metrica, metrica)
                                
                                if diff_porcentaje >= 20:  # 20% mejor que nuestra jugadora
                                    oportunidades.append(f"**{metrica_nombre}**: Potencial para mejorar un {abs(diff_porcentaje):.1f}% hasta {mejor_similar:.2f} (referencia de jugadoras similares)")
                                
                                # Identificar métricas donde está muy por encima de similares (posible riesgo de regresión)
                                if valor_jugadora > 0 and (valor_jugadora - mejor_similar) / mejor_similar > 0.3:
                                    amenazas.append(f"**{metrica_nombre}**: Rendimiento actual de {valor_jugadora:.2f} podría ser difícil de mantener (un {((valor_jugadora - mejor_similar) / mejor_similar * 100):.1f}% superior a jugadoras similares)")
            
            # Agregar análisis específicos por posición
            if posicion == 'GK':
                if 'GA90' in metricas_jugadora and 'Save%' in metricas_jugadora:
                    if metricas_jugadora['GA90'] > 1.2:
                        amenazas.append("Alto ratio de goles encajados podría indicar vulnerabilidad ante ciertos tipos de ataque")
                    if metricas_jugadora['Save%'] < 65:
                        oportunidades.append("Mejorar en porcentaje de paradas para aumentar la solidez defensiva")
            
            elif posicion == 'DF':
                if 'Tkl+Int' in metricas_jugadora and 'Blocks' in metricas_jugadora:
                    if metricas_jugadora['Tkl+Int'] < 3:
                        oportunidades.append("Aumentar acciones defensivas como tackles e intercepciones")
                    if metricas_jugadora['Blocks'] > 2:
                        fortalezas.append("Buena capacidad para bloquear tiros y pases peligrosos")
            
            elif posicion == 'MF':
                if 'pass_1/3' in metricas_jugadora and 'SCA90' in metricas_jugadora:
                    if metricas_jugadora['pass_1/3'] > 5:
                        fortalezas.append("Excelente capacidad para hacer progresar el balón al último tercio")
                    if metricas_jugadora['SCA90'] < 2:
                        oportunidades.append("Potencial para mejorar en la creación de oportunidades de tiro")
            
            elif posicion == 'FW':
                if 'G/Sh' in metricas_jugadora and 'G-xG' in metricas_jugadora:
                    if metricas_jugadora['G/Sh'] < 0.1:
                        oportunidades.append("Mejorar la eficiencia en la finalización de oportunidades")
                    if metricas_jugadora['G-xG'] > 0:
                        fortalezas.append("Sobrerrendimiento en goles respecto a lo esperado por las oportunidades")
            
            # Si no tenemos suficientes puntos, agregamos algunos genéricos
            if len(fortalezas) < 3:
                fortalezas.append("Jugadora con potencial para desarrollarse en su posición")
            if len(debilidades) < 2:
                debilidades.append("Datos insuficientes para identificar áreas de mejora específicas")
            if len(oportunidades) < 3:
                oportunidades.append("Analizar jugadoras de élite en la misma posición para adoptar mejores prácticas")
            if len(amenazas) < 2:
                amenazas.append("La competencia en la misma posición podría limitar las oportunidades de juego")
            
            return {
                "debilidades": debilidades, 
                "amenazas": amenazas, 
                "fortalezas": fortalezas, 
                "oportunidades": oportunidades
            }
        
        # Obtener datos de la jugadora seleccionada
        jugadora_info = df_combined[df_combined['Player'] == jugadora_seleccionada]
        
        # Obtener métricas de la jugadora
        metricas_jugadora = {}
        metricas_numericas = jugadora_info.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for metrica in metricas_numericas:
            if metrica in jugadora_info.columns:
                metricas_jugadora[metrica] = jugadora_info[metrica].iloc[0]
        
        # Obtener datos de jugadoras similares
        metricas_similares = []
        for nombre, _, _, _, _ in distancias_ordenadas[:5]:
            similar_info = df_combined[df_combined['Player'] == nombre]
            metricas_similar = {}
            for metrica in metricas_numericas:
                if metrica in similar_info.columns:
                    metricas_similar[metrica] = similar_info[metrica].iloc[0]
            metricas_similares.append(metricas_similar)
        
        # Obtener promedios por posición
        position = jugadora_info['Posición Principal'].iloc[0]
        jugadoras_misma_posicion = df_combined[df_combined['Posición Principal'] == position]
        
        metricas_promedio = {}
        for metrica in metricas_numericas:
            if metrica in jugadoras_misma_posicion.columns:
                metricas_promedio[metrica] = jugadoras_misma_posicion[metrica].mean()
        
        # Generar el DAFO
        dafo = generar_dafo(
            jugadora_seleccionada, 
            position, 
            metricas_jugadora, 
            metricas_similares, 
            metricas_promedio
        )
        
        # Mostrar DAFO en una presentación visual clara
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fortalezas")
            st.markdown('<div style="background-color: #d4edda; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
            for fortaleza in dafo["fortalezas"]:
                st.markdown(f"✅ {fortaleza}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Debilidades")
            st.markdown('<div style="background-color: #f8d7da; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
            for debilidad in dafo["debilidades"]:
                st.markdown(f"❌ {debilidad}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Oportunidades")
            st.markdown('<div style="background-color: #cce5ff; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
            for oportunidad in dafo["oportunidades"]:
                st.markdown(f"🚀 {oportunidad}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Amenazas")
            st.markdown('<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;">', unsafe_allow_html=True)
            for amenaza in dafo["amenazas"]:
                st.markdown(f"⚠️ {amenaza}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Separador para la siguiente sección
        st.divider()
    
    # Sección de métricas a mejorar
    st.write("### Plan de Mejora basado en Métricas")
    
    # Función para generar recomendaciones de mejora
    def generar_recomendaciones(jugadora, posicion, metricas_jugadora, metricas_similares, metricas_promedio):
        """
        Genera recomendaciones específicas para mejorar basadas en las métricas de la jugadora.
        """
        recomendaciones = []
        
        # Establecer umbrales de mejora por posición y métrica
        umbrales_mejora = {
            'GK': {
                'Save%': {'umbral': 70, 'recomendacion': 'Trabajar en posicionamiento y técnica de paradas'},
                'CS%': {'umbral': 30, 'recomendacion': 'Mejorar comunicación con la defensa para mantener portería a cero'},
                'PSxG-GA': {'umbral': 0, 'recomendacion': 'Enfocarse en situaciones de tiro más difíciles'},
                '#OPA/90': {'umbral': 1, 'recomendacion': 'Aumentar participación fuera del área para ayudar en la construcción'}
            },
            'DF': {
                'Tkl%': {'umbral': 65, 'recomendacion': 'Mejorar timing en las entradas para aumentar efectividad'},
                'Int': {'umbral': 1.5, 'recomendacion': 'Trabajar en anticipación para aumentar intercepciones'},
                'Cmp%_long': {'umbral': 60, 'recomendacion': 'Practicar pases largos para mejorar la progresión desde atrás'},
                'Blocks': {'umbral': 1.2, 'recomendacion': 'Mejorar posicionamiento para bloquear más tiros y pases'}
            },
            'MF': {
                'PPA': {'umbral': 3, 'recomendacion': 'Aumentar los pases al área rival para crear más peligro'},
                'SCA90': {'umbral': 2.5, 'recomendacion': 'Desarrollar más acciones que deriven en ocasiones de tiro'},
                'GCA90': {'umbral': 0.3, 'recomendacion': 'Incrementar las acciones que derivan en gol'},
                'pass_1/3': {'umbral': 4, 'recomendacion': 'Mejorar la progresión del balón al último tercio del campo'},
                'KP': {'umbral': 1, 'recomendacion': 'Aumentar la creación de pases clave para generar ocasiones'}
            },
            'FW': {
                'G/Sh': {'umbral': 0.12, 'recomendacion': 'Mejorar la definición y la toma de decisiones en el área'},
                'SoT/90': {'umbral': 1, 'recomendacion': 'Aumentar la precisión en los disparos a portería'},
                'G-xG': {'umbral': 0, 'recomendacion': 'Trabajar en la finalización para superar las expectativas de gol'},
                'touch_Att Pen': {'umbral': 4, 'recomendacion': 'Incrementar la presencia en el área rival para recibir más balones'},
                'TO_Succ%': {'umbral': 50, 'recomendacion': 'Mejorar la efectividad en los regates para crear ventajas'}
            }
        }
        
        # Si la posición existe en nuestros umbrales
        if posicion in umbrales_mejora:
            # Para cada métrica importante en esa posición
            for metrica, datos in umbrales_mejora[posicion].items():
                if metrica in metricas_jugadora:
                    valor_jugadora = metricas_jugadora[metrica]
                    umbral = datos['umbral']
                    
                    # Si está por debajo del umbral, necesita mejorar
                    if pd.notna(valor_jugadora) and valor_jugadora < umbral:
                        recomendacion = datos['recomendacion']
                        metrica_nombre = metric_display_names.get(metrica, metrica)
                        diferencia = umbral - valor_jugadora
                        
                        # Añadir recomendación con valores actuales y objetivo
                        recomendaciones.append({
                            'metrica': metrica,
                            'nombre_metrica': metrica_nombre,
                            'valor_actual': valor_jugadora,
                            'objetivo': umbral,
                            'diferencia': diferencia,
                            'recomendacion': recomendacion
                        })
        
        # Ordenar las recomendaciones por diferencia (las que necesitan más mejora primero)
        recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x['diferencia'], reverse=True)
        
        return recomendaciones_ordenadas
    
    # Generar recomendaciones
    recomendaciones = generar_recomendaciones(
        jugadora_seleccionada, 
        position, 
        metricas_jugadora, 
        metricas_similares, 
        metricas_promedio
    )
    
    # Mostrar las recomendaciones en una tabla visual
    if recomendaciones:
        # Mostrar solo las 5 principales áreas de mejora
        top_recomendaciones = recomendaciones[:5]
        
        for i, rec in enumerate(top_recomendaciones, 1):
            # Crear una tarjeta para cada recomendación
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="color: #007bff;">Prioridad {i}: Mejorar {rec['nombre_metrica']}</h4>
                <p><b>Valor actual:</b> {rec['valor_actual']:.2f} | <b>Objetivo:</b> {rec['objetivo']:.2f}</p>
                <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px;">
                    <p><b>Recomendación:</b> {rec['recomendacion']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No se han encontrado áreas específicas de mejora basadas en los umbrales establecidos.")
    
    # Sección final con resumen ejecutivo
    st.write("### Resumen Ejecutivo")
    
    # Función para generar un resumen ejecutivo
    def generar_resumen(jugadora, posicion, metricas_jugadora, dafo, recomendaciones):
        """
        Genera un resumen ejecutivo personalizado para la jugadora.
        """
        # Extraer información clave
        nombre = jugadora
        
        # Nivel aproximado basado en las métricas
        nivel = "alto"  # por defecto
        
        # Contar cuántas métricas están por debajo del promedio
        if posicion in umbrales_mejora:
            metricas_clave = umbrales_mejora[posicion].keys()
            metricas_bajas = sum(1 for m in metricas_clave if m in metricas_jugadora and 
                               metricas_jugadora[m] < umbrales_mejora[posicion][m]['umbral'])
            
            total_metricas = sum(1 for m in metricas_clave if m in metricas_jugadora)
            
            if total_metricas > 0:
                ratio = metricas_bajas / total_metricas
                if ratio > 0.7:
                    nivel = "bajo"
                elif ratio > 0.3:
                    nivel = "medio"
        
        # Identificar principales fortalezas
        fortalezas_principales = dafo["fortalezas"][:2] if len(dafo["fortalezas"]) >= 2 else dafo["fortalezas"]
        
        # Identificar principales áreas de mejora
        areas_mejora = [rec['nombre_metrica'] for rec in recomendaciones[:3]] if recomendaciones else ["No se identificaron áreas específicas"]
        
        # Construir resumen personalizado
        resumen = f"""
        {nombre} es una jugadora de nivel {nivel} en su posición de {posicion}. 
        
        **Perfil general:** """
        
        # Añadir descripción según posición
        if posicion == 'GK':
            resumen += "Portera "
            if 'Save%' in metricas_jugadora:
                if metricas_jugadora['Save%'] > 75:
                    resumen += "con excelente porcentaje de paradas "
                elif metricas_jugadora['Save%'] > 65:
                    resumen += "con buen porcentaje de paradas "
                else:
                    resumen += "que necesita mejorar su porcentaje de paradas "
            if '#OPA/90' in metricas_jugadora:
                if metricas_jugadora['#OPA/90'] > 1.5:
                    resumen += "y activa fuera del área."
                else:
                    resumen += "y con margen de mejora en su participación fuera del área."
                    
        elif posicion == 'DF':
            resumen += "Defensora "
            if 'Tkl+Int' in metricas_jugadora:
                if metricas_jugadora['Tkl+Int'] > 4:
                    resumen += "con gran capacidad defensiva "
                elif metricas_jugadora['Tkl+Int'] > 2.5:
                    resumen += "con buena capacidad defensiva "
                else:
                    resumen += "con capacidad defensiva a mejorar "
            if 'Cmp%_long' in metricas_jugadora:
                if metricas_jugadora['Cmp%_long'] > 65:
                    resumen += "y excelente en la distribución larga del balón."
                elif metricas_jugadora['Cmp%_long'] > 50:
                    resumen += "y buena en la distribución del balón."
                else:
                    resumen += "y con margen de mejora en la distribución del balón."
                    
        elif posicion == 'MF':
            resumen += "Centrocampista "
            if 'SCA90' in metricas_jugadora:
                if metricas_jugadora['SCA90'] > 3:
                    resumen += "con gran capacidad creativa "
                elif metricas_jugadora['SCA90'] > 2:
                    resumen += "con buena capacidad creativa "
                else:
                    resumen += "con capacidad creativa a desarrollar "
            if 'Tkl+Int' in metricas_jugadora:
                if metricas_jugadora['Tkl+Int'] > 3.5:
                    resumen += "y excelente en la recuperación defensiva."
                elif metricas_jugadora['Tkl+Int'] > 2:
                    resumen += "y sólida en el aspecto defensivo."
                else:
                    resumen += "y con margen de mejora en el aspecto defensivo."
                    
        elif posicion == 'FW':
            resumen += "Delantera "
            if 'G/Sh' in metricas_jugadora:
                if metricas_jugadora['G/Sh'] > 0.15:
                    resumen += "con gran efectividad goleadora "
                elif metricas_jugadora['G/Sh'] > 0.1:
                    resumen += "con buena capacidad goleadora "
                else:
                    resumen += "con capacidad goleadora a mejorar "
            if 'SCA90' in metricas_jugadora:
                if metricas_jugadora['SCA90'] > 3:
                    resumen += "y excelente en la creación de oportunidades."
                elif metricas_jugadora['SCA90'] > 2:
                    resumen += "y buena en la creación de juego."
                else:
                    resumen += "y con margen de mejora en la creación de juego."
        
        # Añadir sección de fortalezas principales
        resumen += """
        
        **Principales fortalezas:**
        """
        for f in fortalezas_principales:
            resumen += f"\n- {f}"
        
        # Añadir sección de áreas de mejora
        resumen += """
        
        **Principales áreas de mejora:**
        """
        for i, area in enumerate(areas_mejora, 1):
            if i <= len(recomendaciones):
                resumen += f"\n- {area}: {recomendaciones[i-1]['recomendacion']}"
            else:
                resumen += f"\n- {area}"
        
        # Añadir conclusión
        resumen += """
        
        **Conclusión:** """
        
        if nivel == "alto":
            resumen += "Jugadora con excelente rendimiento que puede ayudar al equipo inmediatamente. Se recomienda mantener su desarrollo enfocándose en las áreas de mejora identificadas para maximizar su potencial."
        elif nivel == "medio":
            resumen += "Jugadora con buen rendimiento que puede contribuir al equipo. Con trabajo específico en las áreas identificadas, podría elevar significativamente su nivel y aportar más valor al equipo."
        else:
            resumen += "Jugadora con potencial que necesita desarrollo específico. Centrándose en las áreas de mejora identificadas, podría incrementar sustancialmente su aporte al equipo a medio plazo."
        
        return resumen
    
    # Generar el resumen ejecutivo
    resumen = generar_resumen(
        jugadora_seleccionada, 
        position, 
        metricas_jugadora, 
        dafo, 
        recomendaciones
    )
    
    # Mostrar el resumen en un contenedor destacado
    st.markdown('<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff;">', unsafe_allow_html=True)
    st.markdown(resumen)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Opciones para exportar el informe
    st.divider()
    st.write("### Opciones de exportación")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Exportar como PDF"):
            st.info("Función de exportación a PDF no implementada en esta versión. Esta funcionalidad requeriría integración con librerías externas como ReportLab o WeasyPrint.")
    
    with col2:
        if st.button("📊 Exportar como presentación"):
            st.info("Función de exportación a presentación no implementada en esta versión. Esta funcionalidad requeriría integración con librerías para generar PowerPoint o similares.")
    
    # Información de interpretación
    st.info("""
    **Nota sobre el análisis IA:**
    - El análisis se basa únicamente en datos estadísticos disponibles
    - Las recomendaciones son generales y deben ser evaluadas por el cuerpo técnico
    - El análisis DAFO y las métricas a mejorar son herramientas orientativas para la toma de decisiones
    """)
