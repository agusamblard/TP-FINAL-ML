
#=============================
# IMPORTS
#=============================
from utils.diccionarios import MARCAS_VALIDAS, MODELOS_POR_MARCA, TERMINOS_TRACCION, TERMINOS_TRANSMISION, TIPOS_COMBUSTIBLE
import pandas as pd
from difflib import get_close_matches
from collections import defaultdict
import re
import numpy as np
import datetime

#=============================
# FUNCIONES DE LIMPIEZA PRE-SPLIT
#=============================

def quitar_tildes(texto):

    """
    Elimina tildes y caracteres especiales de un texto.
    """
    if pd.isna(texto):
        return ''
    texto = str(texto).strip().lower()

    # Reemplazo manual de tildes y otros caracteres especiales
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
    }
    texto = ''.join(reemplazos.get(c, c) for c in texto)

    return texto



def normalizar(texto, eliminar_espacios=True):
    if pd.isna(texto):
        return ''
    texto = str(texto).strip().lower()

    # Reemplazo de tildes y diéresis
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'ñ': 'n'
    }
    texto = ''.join(reemplazos.get(c, c) for c in texto)

    # Símbolos raros a eliminar
    simbolos_a_eliminar = ['-', '.', ',', '"', "'", '“', '”', '’', '`',
                           '(', ')', '[', ']', '{', '}', ':', ';', '!', '?', '#', '@', '°', 'º', 'ª', '/', '\\', '|']

    if eliminar_espacios:
        simbolos_a_eliminar.append(' ')

    for simbolo in simbolos_a_eliminar:
        texto = texto.replace(simbolo, '')

    # Eliminar cualquier carácter no alfanumérico restante
    if eliminar_espacios:
        texto = re.sub(r'[^a-z0-9]', '', texto)
    else:
        texto = re.sub(r'[^a-z0-9 ]', '', texto)

    return texto



def limpiar_marcas(df):
    df = df.copy()
    indices_a_eliminar = []

    for idx, row in df.iterrows():
        marca = str(row['Marca'])
        titulo = str(row.get('Título', ''))
        modelo = row.get('Modelo')

        marca_norm = normalizar(marca)
        titulo_norm = normalizar(titulo)

        # Paso 0: hardcodeo de "haval" → "gwm"
        if marca_norm == "haval":
            df.at[idx, 'Marca'] = "gwm"
            continue

        # Paso 1: si la marca ya es válida
        if marca_norm in MARCAS_VALIDAS:
            df.at[idx, 'Marca'] = marca_norm
            continue

        # Paso 2: buscar una marca válida en el título
        encontrada = False
        for m_valida in MARCAS_VALIDAS:
            if m_valida in titulo_norm:
                df.at[idx, 'Marca'] = m_valida
                encontrada = True
                break
        if encontrada:
            continue

        # Paso 3: buscar por similitud
        match = get_close_matches(marca_norm, MARCAS_VALIDAS, n=1, cutoff=0.7)
        if match:
            df.at[idx, 'Marca'] = match[0]
        else:
            indices_a_eliminar.append(idx)

    # Paso 4: las marcas que no se pudieron corregir se marcan como NaN
    df.loc[indices_a_eliminar, 'Marca'] = pd.NA

    return df



def limpiar_modelo(df):

    df = df.copy()
    df['modelo_prev'] = df['Modelo']  # Guardar versión original

    # Utilizar la variable global MODELOS_POR_MARCA
    marcas_por_modelo = defaultdict(list)
    for marca, modelos in MODELOS_POR_MARCA.items():
        marca_norm = normalizar(marca)
        for modelo in modelos:
            modelo_norm = normalizar(modelo)
            marcas_por_modelo[modelo_norm].append(marca_norm)

    modelos_validos = list(marcas_por_modelo.keys())

    nuevos_modelos = []

    for _, row in df.iterrows():
        modelo = row['Modelo']
        marca = row['Marca']
        titulo = row['Título'] if 'Título' in row and pd.notna(row['Título']) else ''

        if pd.isna(modelo) or pd.isna(marca):
            nuevos_modelos.append(np.nan)
            continue

        modelo_norm = normalizar(modelo)
        marca_norm = normalizar(marca)
        titulo_norm = normalizar(titulo)

        # CASO 1: modelo exacto y marca válida
        if modelo_norm in modelos_validos and marca_norm in marcas_por_modelo[modelo_norm]:
            nuevos_modelos.append(modelo_norm)
            continue

        # CASO 2: búsqueda del modelo en el título
        modelo_en_titulo = None
        for posible_modelo in modelos_validos:
            if marca_norm in marcas_por_modelo[posible_modelo] and posible_modelo in titulo_norm:
                modelo_en_titulo = posible_modelo
                break

        if modelo_en_titulo:
            nuevos_modelos.append(modelo_en_titulo)
            continue

        # CASO 3: búsqueda por similitud
        modelos_de_marca = [m for m in modelos_validos if marca_norm in marcas_por_modelo[m]]
        coincidencias = get_close_matches(modelo_norm, modelos_de_marca, n=1, cutoff=0.7)
        if coincidencias:
            nuevos_modelos.append(coincidencias[0])
            continue

        # CASO 4: no se puede inferir el modelo
        nuevos_modelos.append(np.nan)

    df['Modelo'] = nuevos_modelos
    return df



def limpiar_carroceria(df):
    """
    Se fija la columna 'Tipo de carrocería'. Para todos los valores en esta columna
    que no sean 'SUV', revisar si el modelo está en el diccionario de MODELOS_POR_MARCA
    para la marca correspondiente. Si hay coincidencia, asignar 'SUV' a la columna 'Tipo de carrocería'.
    Si no hay coincidencia, eliminar la muestra ya que no es una SUV.
    Luego, eliminar la columna 'Tipo de carrocería'.
    """
    df = df.copy()

    indices_a_eliminar = []

    for idx, row in df.iterrows():
        tipo = str(row.get('Tipo de carrocería', '')).strip().lower()
        marca = normalizar(row['Marca'])
        modelo = normalizar(row['Modelo'])

        if tipo == 'suv':
            continue

        if modelo in [normalizar(m) for m in MODELOS_POR_MARCA.get(marca, [])]:
            df.at[idx, 'Tipo de carrocería'] = 'SUV'
        else:
            indices_a_eliminar.append(idx)

    df.drop(index=indices_a_eliminar, inplace=True)
    df.drop(columns=['Tipo de carrocería'], errors='ignore', inplace=True)

    return df


def extraer_hp(df):
    """
    Recibe un DataFrame y agrega una columna 'HP' con la potencia extraída 
    de cualquier columna (excepto 'Descripción') que contenga un valor como 123 cv o 123hp.
    """
    def detectar_hp(fila):
        texto = ' '.join(
            str(v).lower() for k, v in fila.items()
            if k != 'Descripción' and pd.notna(v)
        )
        match = re.search(r'\b(\d{2,3})\s*(cv|hp)\b', texto)
        if match:
            return int(match.group(1))
        return None

    df = df.copy()
    df['HP'] = df.apply(detectar_hp, axis=1)
    return df




def extraer_traccion(df):
    """
    Crea una nueva columna 'Tracción' basada solo en los campos 'Versión' y 'Título'.
    Si detecta un solo tipo ('4x4' o '4x2'), lo asigna. Si hay ambigüedad o falta de info, asigna NaN.
    """
    df = df.copy()

    # Términos por tipo
    # Usar TERMINOS_TRACCION
    todos_los_terminos = sum(TERMINOS_TRACCION.values(), [])
    mapa_traccion = {t: tipo for tipo, lista in TERMINOS_TRACCION.items() for t in lista}

    def detectar_traccion(fila):
        version = str(fila.get('Versión', '')).lower()
        titulo = str(fila.get('Título', '')).lower()
        texto = f"{version} {titulo}"

        # Buscar coincidencias exactas por palabra
        matches = re.findall(r'\b(' + '|'.join(map(re.escape, todos_los_terminos)) + r')\b', texto)
        tipos_detectados = {mapa_traccion[m] for m in matches if m in mapa_traccion}

        if len(tipos_detectados) == 1:
            return list(tipos_detectados)[0]
        else:
            return np.nan

    df['Tracción'] = df.apply(detectar_traccion, axis=1)
    return df


def limpiar_transmision(df):
    """
    Asigna la transmisión ('AUTOMÁTICA' o 'MANUAL') basada en las columnas 'Transmisión', 'Versión' y 'Título'.
    Si no hay información clara o hay conflicto, marca como NaN.
    """
    df = df.copy()
    df['transmision_prev'] = df['Transmisión']  # Guardar original

    # Construir terminos_transmision y mapa_normalizado desde TERMINOS_TRANSMISION
    terminos_transmision = sum(TERMINOS_TRANSMISION.values(), [])
    mapa_normalizado = {term: tipo for tipo, lista in TERMINOS_TRANSMISION.items() for term in lista}

    def detectar_transmision(fila):
        transmision_original = str(fila.get('Transmisión', '')).lower()
        version = str(fila.get('Versión', '')).lower()
        titulo = str(fila.get('Título', '')).lower()

        # Paso 1: Revisar directamente la columna 'Transmisión'
        matches_directos = re.findall(r'\b(' + '|'.join(map(re.escape, terminos_transmision)) + r')\b', transmision_original)
        tipos_directos = {mapa_normalizado[m] for m in matches_directos if m in mapa_normalizado}
        if len(tipos_directos) == 1:
            return list(tipos_directos)[0]

        # Paso 2: Buscar en 'Versión' y 'Título'
        texto = f"{version} {titulo}"
        matches_texto = re.findall(r'\b(' + '|'.join(map(re.escape, terminos_transmision)) + r')\b', texto)
        tipos_detectados = {mapa_normalizado[m] for m in matches_texto if m in mapa_normalizado}

        if len(tipos_detectados) == 1:
            return list(tipos_detectados)[0]

        # No se puede determinar un único tipo de transmisión
        return np.nan

    df['Transmisión'] = df.apply(detectar_transmision, axis=1)
    return df



def limpiar_version(df):
    """
    Limpia la columna 'Versión', elimina los términos irrelevantes y deja solo el nombre distintivo.
    """
    df = df.copy()
    df['Versión_prev'] = df['Versión']  # guardar versión original

    # Términos a remover (usar las variables globales)
    terminos_traccion = sum(TERMINOS_TRACCION.values(), [])
    terminos_transmision = sum(TERMINOS_TRANSMISION.values(), [])
    terminos_combustible = sum(TIPOS_COMBUSTIBLE.values(), [])

    regex_asientos = re.compile(r'\b[1-9]\s?(as|p|puertas|plazas|pasajeros|asientos|pas)\b', re.IGNORECASE)

    versiones_limpias = []

    for idx, row in df.iterrows():
        version = row['Versión']
        if pd.isna(version):
            versiones_limpias.append(version)
            continue

        version_limpia = normalizar(version, eliminar_espacios=False)

        marca = normalizar(row['Marca'], eliminar_espacios=False) if pd.notna(row['Marca']) else ''
        modelo = normalizar(row['Modelo'], eliminar_espacios=False) if pd.notna(row['Modelo']) else ''
        
        version_limpia = re.sub(rf'\b{re.escape(marca)}\b', '', version_limpia)
        version_limpia = re.sub(rf'\b{re.escape(modelo)}\b', '', version_limpia)

        version_limpia = re.sub(r'\b\d{1,2}(\.\d)?\s*[lt]?\b', '', version_limpia)  # 1.6, 2.0
        version_limpia = re.sub(r'\b\.\d\s*[lt]?\b', '', version_limpia)
        version_limpia = re.sub(r'\b\d{2,3}\s*(cv|hp)\b', '', version_limpia)

        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_traccion)) + r')\b', '', version_limpia)
        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_transmision)) + r')\b', '', version_limpia)
        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_combustible)) + r')\b', '', version_limpia)

        version_limpia = regex_asientos.sub('', version_limpia)
        version_limpia = re.sub(r'\s{2,}', ' ', version_limpia).strip()

        versiones_limpias.append(version_limpia)

    df['Versión'] = versiones_limpias

    return df


def limpiar_motor(df):
    """
    Procesa un DataFrame y limpia la columna 'Motor', extrayendo la cilindrada.
    Busca primero en 'Motor', luego en 'Título' y luego en 'Versión_prev'.
    Devuelve el DataFrame modificado.
    """
    df = df.copy()

    #guardar los valores de motor originales
    df['Motor_prev'] = df['Motor']
    motores_limpios = []

    for _, row in df.iterrows():
        motor = str(row['Motor']).lower().replace(',', '.') if pd.notna(row['Motor']) else ''
        match = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', motor)
        if match:
            motores_limpios.append(match.group(1))
            continue

        titulo = str(row['Título']).lower().replace(',', '.') if pd.notna(row['Título']) else ''
        match_titulo = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', titulo)
        if match_titulo:
            motores_limpios.append(match_titulo.group(1))
            continue

        version_prev = str(row['Versión_prev']).lower().replace(',', '.') if pd.notna(row['Versión_prev']) else ''
        match_version = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', version_prev)
        if match_version:
            motores_limpios.append(match_version.group(1))
        else:
            motores_limpios.append(None)

    df['Motor'] = motores_limpios


    return df




def limpiar_combustible(df):
    import re

    prioridad = list(TIPOS_COMBUSTIBLE.keys())

    regex_por_tipo = {
        tipo: re.compile(r'\b(' + '|'.join(map(re.escape, palabras)) + r')\b', re.IGNORECASE)
        for tipo, palabras in TIPOS_COMBUSTIBLE.items()
    }

    df = df.copy()
    df['Tipo original'] = df['Tipo de combustible']

    # --- Paso 0: Normalizar valores atípicos ---
    for idx, fila in df.iterrows():
        tipo_actual = str(fila['Tipo de combustible']).lower()
        tipo_actual_sin_tildes = quitar_tildes(tipo_actual)

        tipos_detectados = []
        for tipo, palabras in TIPOS_COMBUSTIBLE.items():
            for palabra in palabras:
                if palabra in tipo_actual_sin_tildes:
                    tipos_detectados.append(tipo)
                    break

        for tipo_prioritario in prioridad:
            if tipo_prioritario in tipos_detectados:
                df.at[idx, 'Tipo de combustible'] = tipo_prioritario
                break

    # --- Paso 1: Detectar tipo solo desde 'Versión_prev' y 'Motor_prev' ---
    def detectar_tipo(fila):
        texto = ''
        if 'Versión_prev' in fila and pd.notna(fila['Versión_prev']):
            texto += ' ' + quitar_tildes(str(fila['Versión_prev']).lower())
        if 'Motor_prev' in fila and pd.notna(fila['Motor_prev']):
            texto += ' ' + quitar_tildes(str(fila['Motor_prev']).lower())

        tipos_detectados = [tipo for tipo, regex in regex_por_tipo.items() if regex.search(texto)]
        for tipo_prioritario in prioridad:
            if tipo_prioritario in tipos_detectados:
                return tipo_prioritario

        return fila['Tipo de combustible']

    df['Tipo de combustible'] = df.apply(detectar_tipo, axis=1)

    # --- Paso 2: Corregir falsos positivos conservadores ---
    '''
    for tipo_objetivo in prioridad:
        regex_tipo = regex_por_tipo[tipo_objetivo]
        for idx, fila in df.iterrows():
            if fila['Tipo original'] == tipo_objetivo and fila['Tipo de combustible'] == tipo_objetivo:
                texto = ''
                if 'Versión_prev' in fila and pd.notna(fila['Versión_prev']):
                    texto += ' ' + quitar_tildes(str(fila['Versión_prev']).lower())
                if 'Motor_prev' in fila and pd.notna(fila['Motor_prev']):
                    texto += ' ' + quitar_tildes(str(fila['Motor_prev']).lower())

                if not regex_tipo.search(texto):
                    filtro = (
                        (df['Marca'] == fila['Marca']) &
                        (df['Modelo'] == fila['Modelo']) &
                        (df['Versión'] == fila['Versión']) &
                        (df['Año'] == fila['Año']) &
                        (df['Motor'] == fila['Motor']) &
                        (df['Tipo de combustible'] != tipo_objetivo) &
                        (df['Tipo de combustible'].notna())
                    )
                    tipo_mas_comun = df.loc[filtro, 'Tipo de combustible'].mode()
                    if not tipo_mas_comun.empty:
                        df.at[idx, 'Tipo de combustible'] = tipo_mas_comun.iloc[0]
'''
    return df


def limpiar_precio(df, valor_dolar=1250):
    """
    Convierte los precios en pesos a dólares en un DataFrame.
    Si la moneda es US$, no hace la conversión.
    Si la moneda es $, divide el precio por valor_dolar y redondea al entero más cercano.
    """
    df = df.copy()

    # Aplicar la conversión solo cuando la moneda es '$' y el precio no es nulo
    mask = (df['Moneda'] == '$') & df['Precio'].notna()
    df.loc[mask, 'Precio'] = df.loc[mask, 'Precio'].astype(float) / valor_dolar
    df.loc[mask, 'Precio'] = df.loc[mask, 'Precio'].round().astype(int)

    #si hay algun valor negativo, lo convertimos a NaN
    df.loc[df['Precio'] < 0, 'Precio'] = pd.NA

    return df

def limpiar_año(df):
    df = df.copy()
    año_actual = datetime.datetime.now().year

    # Convertir a numérico, forzar errores como NaN
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce')

    # Reemplazar años inválidos por NaN
    df.loc[(df['Año'] < 1980) | (df['Año'] > año_actual), 'Año'] = np.nan

    return df

def limpiar_km(df):
    """
    Limpia la columna 'Kilómetros' del DataFrame:
    - Detecta y corrige valores con separadores de miles y decimales.
    - Convierte todo a float.
    - Reemplaza valores no numéricos o negativos por NaN.
    """
    df = df.copy()

    def procesar_valor(val):
        if pd.isna(val):
            return np.nan

        texto = str(val).lower().replace('km', '').strip()
        texto = texto.replace(',', '.')  # primero: tratar comas como puntos

        # Caso: más de un punto → probablemente separadores de miles: eliminar todos
        if texto.count('.') > 1:
            texto = texto.replace('.', '')
        elif texto.count('.') == 1:
            punto_pos = texto.index('.')
            digitos_despues = len(texto) - punto_pos - 1
            # Si hay 3 dígitos después del punto y el resto son números → separador de miles
            if digitos_despues == 3 and texto.replace('.', '').isdigit():
                texto = texto.replace('.', '')

        # Eliminar cualquier otro caracter que no sea dígito o punto
        texto = re.sub(r'[^\d\.]', '', texto)

        try:
            valor = float(texto)
            if valor < 0:
                return np.nan
            return valor
        except:
            return np.nan

    df['Kilómetros'] = df['Kilómetros'].apply(procesar_valor)
    return df


def limpiar_puertas(df):
    df = df.copy()

    # Normalizar versión_prev
    versiones_norm = df['Versión_prev'].apply(lambda x: normalizar(x, eliminar_espacios=False) if pd.notna(x) else '')

    # Compilar patrones
    regex_3p = re.compile(r'\b(3p|3 puertas|3puertas|tres puertas|coupe)\b', re.IGNORECASE)
    regex_5p = re.compile(r'\b(5p|5 puertas|5puertas|cinco puertas)\b', re.IGNORECASE)

    nuevas_puertas = []

    for idx, row in df.iterrows():
        puertas = row['Puertas']
        version_texto = versiones_norm.iloc[idx]

        # Paso 1: invalidar valores no válidos (no 3 ni 5)
        if puertas not in [3, 5]:
            puertas = np.nan

        # Paso 2: detectar desde versión_prev
        detectado = None
        if regex_5p.search(version_texto):
            detectado = 5
        elif regex_3p.search(version_texto):
            detectado = 3

        # Paso 3: decidir valor final
        if pd.isna(puertas):
            nuevas_puertas.append(detectado if detectado else np.nan)
        else:
            if detectado and puertas != detectado:
                nuevas_puertas.append(detectado)
            else:
                nuevas_puertas.append(puertas)

    df['Puertas'] = nuevas_puertas
    return df


def limpiar_camara_de_retroceso(df):
    '''
    Revisar la columna de 'Con cámara de retroceso'.
    Si aparece algo que no es 'Sí' o 'No', reemplazarlo por NaN.
    Si es 'Sí', convertirlo a 1, y si es 'No', convertirlo a 0
    '''
    df = df.copy()
    df['Con cámara de retroceso'] = df['Con cámara de retroceso'].str.strip().str.lower()
    df['Con cámara de retroceso'] = df['Con cámara de retroceso'].map({'sí': 1, 'no': 0}).astype("Int64")

    # Reemplazar cualquier otro valor que no sea 1 o 0 por NaN
    df.loc[~df['Con cámara de retroceso'].isin([0, 1]), 'Con cámara de retroceso'] = np.nan

    return df

def limpiar_tipo_de_vendedor(df):
    """
    Revisa la columna 'Tipo de vendedor' y normaliza los valores.
    Si el valor no es 'concesionaria', 'tienda' o 'particular', elimina la muestra.
    """
    df = df.copy()
    df['Tipo de vendedor'] = df['Tipo de vendedor'].str.strip().str.lower()

    tipos_validos = ['concesionaria', 'tienda', 'particular']
    df = df[df['Tipo de vendedor'].isin(tipos_validos)]

    return df

#==============================
#FUNCIÓN DE LIMPIEZA PRE-SPLIT COMPACTA
#==============================


def limpiar_dataset(df):
    """
    Limpia un DataFrame de vehículos aplicando varias funciones de limpieza.
    Además, imprime la cantidad de muestras eliminadas por las funciones que eliminan filas.
    """
    # Eliminar columnas de tipo de combustible, descripción y titulo
    df = df.drop(columns=['Unnamed: 0', 'Descripción'], errors='ignore')

    original_count = len(df)

    # Lista de funciones de limpieza, con etiquetas
    limpieza_funcs = [
        ("limpiar_marcas", limpiar_marcas),
        ("limpiar_modelo", limpiar_modelo),
        ("limpiar_carroceria", limpiar_carroceria),
        ("extraer_hp", extraer_hp),
        ("limpiar_transmision", limpiar_transmision),
        ("extraer_traccion", extraer_traccion),
        ("limpiar_version", limpiar_version),
        ("limpiar_motor", limpiar_motor),
        ("limpiar_combustible", limpiar_combustible),
        ("limpiar_precio", limpiar_precio),
        ("limpiar_año", limpiar_año),
        ("limpiar_km", limpiar_km),
        ("limpiar_puertas", limpiar_puertas),
        ("limpiar_camara_de_retroceso", limpiar_camara_de_retroceso),
        ("limpiar_tipo_de_vendedor", limpiar_tipo_de_vendedor),
    ]

    for func_name, func in limpieza_funcs:
        count_before = len(df)
        df = func(df)
        count_after = len(df)
        if count_after < count_before:
            print(f"[{func_name}] Muestras eliminadas: {count_before - count_after}")

    # Eliminar columnas auxiliares
    df = df.drop(columns=['Título', 'modelo_prev', 'Tipo de carrocería', 'Moneda', 'transmision_prev', 'Motor_prev', 'Tipo original','Versión_prev'], errors='ignore')

    total_eliminadas = original_count - len(df)
    print(f"[TOTAL] Muestras eliminadas en total: {total_eliminadas}")

    return df