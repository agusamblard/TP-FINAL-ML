import numpy as np
import pandas as pd

import pandas as pd
from difflib import get_close_matches

import pandas as pd
from collections import Counter

# Lista de marcas válidas
marcas_validas = [
    'Ford', 'Jeep', 'Volkswagen', 'Chevrolet', 'Renault', 'Toyota',
    'Peugeot', 'Nissan', 'Citroën', 'BMW', 'Honda', 'Hyundai', 'Audi',
    'Fiat', 'Chery', 'Kia', 'Mercedes-Benz', 'Dodge', 'BAIC', 'Suzuki',
    'Porsche', 'Land Rover', 'Mitsubishi', 'Volvo', 'D.S.', 'SsangYong',
    'Alfa Romeo', 'JAC', 'Jetour', 'Haval', 'GWM', 'Isuzu', 'Lifan',
    'Lexus', 'Subaru', 'Daihatsu', 'Mini', 'KAIYI', 'Jaguar'
]


def corregir_marcas(df):
    df = df.copy()

    for idx, row in df.iterrows():
        marca = row['Marca']
        titulo = str(row['Título']).lower()
        modelo = row['Modelo']

        # Paso 1: si ya es válida, salteamos
        if marca in marcas_validas:
            continue

        # Paso 2: buscar una marca válida en el título
        encontrada = False
        for m in marcas_validas:
            if m.lower() in titulo:
                df.at[idx, 'Marca'] = m
                encontrada = True
                break
        if encontrada:
            continue

        # Paso 3: buscar la marca más frecuente para el mismo modelo
        if pd.notna(modelo):
            subset = df[df['Modelo'] == modelo]
            marcas_modelo = subset['Marca'].tolist()
            marcas_frecuentes = [m for m in marcas_modelo if m in marcas_validas]
            if marcas_frecuentes:
                marca_mas_frecuente = Counter(marcas_frecuentes).most_common(1)[0][0]
                df.at[idx, 'Marca'] = marca_mas_frecuente

    return df


from difflib import get_close_matches
import unidecode
import numpy as np
import pandas as pd

# Diccionario de modelos válidos → marcas (deberías completar las asignaciones reales)
marca_por_modelo = {
    'ecosport': 'ford', 'tracker': 'chevrolet', '2008': 'peugeot', 'duster': 'renault',
    'compass': 'jeep', 'kicks': 'nissan', 'taos': 'volkswagen', 'renegade': 'jeep',
    't-cross': 'volkswagen', 'corolla cross': 'toyota', 'c4 cactus': 'citroen',
    'nivus': 'volkswagen', 'tucson': 'hyundai', 'pulse': 'fiat', 'hr-v': 'honda',
    'hilux sw4': 'toyota', 'territory': 'ford', 'x1': 'bmw', 'cr-v': 'honda',
    'captur': 'renault', 'c3 aircross': 'citroen', 'sw4': 'toyota', 'tiguan': 'volkswagen',
    'q5': 'audi', 'grand cherokee': 'jeep', 'journey': 'dodge', 'kuga': 'ford',
    # ... (continuar completando según tu dataset)
}

modelos_validos = list(marca_por_modelo.keys())

def corregir_modelo(row):
    modelo = row['Modelo']
    marca = row['Marca']

    if pd.isna(modelo) or pd.isna(marca):
        return np.nan

    modelo_clean = unidecode.unidecode(str(modelo)).lower().strip()
    marca_clean = unidecode.unidecode(str(marca)).lower().strip()

    if modelo_clean in modelos_validos and marca_clean == marca_por_modelo.get(modelo_clean):
        return modelo.title()

    coincidencias = get_close_matches(modelo_clean, modelos_validos, n=1, cutoff=0.7)
    if coincidencias:
        modelo_sugerido = coincidencias[0]
        marca_sugerida = marca_por_modelo.get(modelo_sugerido)
        if marca_sugerida == marca_clean:
            return modelo_sugerido.title()

    return np.nan



def quitar_cilindrada(version):
    if pd.isna(version):
        return version
    # Reemplazar coma decimal por punto (casos como "1,5" → "1.5")
    version = version.replace(',', '.')
    # Eliminar patrones tipo 1.5, 2.0T, 1.6L, etc.
    version_limpia = re.sub(r'\b\d\.\d[A-Za-z]?\b', '', version)
    # Limpiar espacios dobles o sobrantes
    version_limpia = re.sub(r'\s{2,}', ' ', version_limpia).strip()
    return version_limpia
import re

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

terminos_traccion = {
    '4x4': ['4x4','awd', '4matic', 'quattro', '4m', '4wd', 'xdrive'],
    '4x2': ['2x4','4x2','fwd', 'rwd', '2wd']
}

# Generar set completo de términos
todos_los_terminos_traccion = sum(terminos_traccion.values(), [])

# Mapeo inverso para saber si un término es 4x4 o 4x2
mapa_traccion = {t: tipo for tipo, lista in terminos_traccion.items() for t in lista}

def extraer_traccion(df):
    """
    Recibe un DataFrame y agrega la columna 'Tracción' con los valores '4x4', '4x2', 'CLASH' o None.
    """
    def detectar_traccion(fila):
        texto = ' '.join(
            str(v).lower() for k, v in fila.items()
            if k != 'Descripción' and pd.notna(v)
        )
        matches = re.findall(r'\b(' + '|'.join(todos_los_terminos_traccion) + r')\b', texto)
        tipos_detectados = {mapa_traccion[m] for m in matches if m in mapa_traccion}

        if not tipos_detectados:
            return None
        elif len(tipos_detectados) == 1:
            return list(tipos_detectados)[0]
        else:
            return 'CLASH'

    df = df.copy()
    df['Tracción'] = df.apply(detectar_traccion, axis=1)
    return df


terminos_transmision = [
    'at', '6at', '8at', 'at6', 'atx', 'cvt', 'tiptronic', 'stronic',
    'dsg', 'automática', 'automatic', 'tronic', 'aut',
    'mt', '6mt', 'manual', 'automatico', 'automático'
]

mapa_normalizado = {
    'at': 'AUTOMÁTICA', '6at': 'AUTOMÁTICA', '8at': 'AUTOMÁTICA', 'at6': 'AUTOMÁTICA',
    'atx': 'AUTOMÁTICA','cvt': 'AUTOMÁTICA', 'tiptronic': 'AUTOMÁTICA', 'stronic': 'AUTOMÁTICA',
    'dsg': 'AUTOMÁTICA', 'automática': 'AUTOMÁTICA', 'automatic': 'AUTOMÁTICA',
    'tronic': 'AUTOMÁTICA', 'aut': 'AUTOMÁTICA', 'automatico': 'AUTOMÁTICA', 'automático': 'AUTOMÁTICA',
    'mt': 'MANUAL', '6mt': 'MANUAL', 'manual': 'MANUAL'
}

def extraer_transmision(df):
    """
    Recibe un DataFrame y agrega una columna 'Transmisión' con el tipo detectado
    ('AUTOMÁTICA', 'MANUAL' o 'CLASH' si hay conflicto).
    """
    def detectar_transmision(fila):
        texto = ' '.join(
            str(v).lower() for k, v in fila.items()
            if k != 'Descripción' and pd.notna(v)
        )
        matches = re.findall(r'\b(' + '|'.join(terminos_transmision) + r')\b', texto)
        tipos_detectados = {mapa_normalizado[m.lower()] for m in matches if m.lower() in mapa_normalizado}

        if not tipos_detectados:
            return None
        elif len(tipos_detectados) == 1:
            return list(tipos_detectados)[0]
        else:
            return 'CLASH'

    df = df.copy()
    df['Transmisión'] = df.apply(detectar_transmision, axis=1)
    return df



def quitar_tildes(texto):
    tildes = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
              'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u'}
    return ''.join(tildes.get(c, c) for c in texto)


def limpiar_version(df):
    """
    Recibe un DataFrame con columnas 'Versión', 'Marca' y 'Modelo'.
    Limpia el contenido de la columna 'Versión' directamente y devuelve el DataFrame modificado.
    """
    df = df.copy()

    terminos_traccion = ['4x4','awd', '4matic', 'quattro', '4m', '4wd', 'xdrive',
                         '2x4','4x2','fwd', 'rwd', '2wd']
    
    terminos_transmision = [
        'at', '6at', '8at', 'at6', 'atx', 'cvt', 'tiptronic', 'stronic',
        'dsg', 'automatica', 'automatic', 'tronic', 'aut',
        'mt', '6mt', 'manual', 'automatico', 'automático'
    ]

    versiones_limpias = []

    for _, row in df.iterrows():
        version = row['Versión']
        if pd.isna(version):
            versiones_limpias.append(version)
            continue

        version_limpia = quitar_tildes(str(version).lower().replace(',', '.'))

        # Eliminar marca y modelo
        marca = quitar_tildes(str(row['Marca']).lower()) if pd.notna(row['Marca']) else ''
        modelo = quitar_tildes(str(row['Modelo']).lower()) if pd.notna(row['Modelo']) else ''
        version_limpia = re.sub(rf'\b{re.escape(marca)}\b', '', version_limpia)
        version_limpia = re.sub(rf'\b{re.escape(modelo)}\b', '', version_limpia)

        # Eliminar patrón de motor (como 1.5, 1.5t, .5 l, etc.)
        version_limpia = re.sub(r'\b\d{1,2}(\.\d)?\s*[lt]?\b', '', version_limpia)
        version_limpia = re.sub(r'\b\.\d\s*[lt]?\b', '', version_limpia)

        # Eliminar HP/CV
        version_limpia = re.sub(r'\b\d{2,3}\s*(cv|hp)\b', '', version_limpia)

        # Eliminar tracción
        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_traccion)) + r')\b', '', version_limpia)

        # Eliminar transmisión
        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_transmision)) + r')\b', '', version_limpia)

        # Limpieza final de espacios
        version_limpia = re.sub(r'\s{2,}', ' ', version_limpia).strip()

        versiones_limpias.append(version_limpia)

    df['Versión'] = versiones_limpias
    return df



def limpiar_motor(df):
    """
    Procesa un DataFrame y limpia la columna 'Motor', extrayendo la cilindrada.
    Si no la encuentra en 'Motor', intenta extraerla del 'Título'.
    Si aún no se encuentra, la busca en otra fila con misma Marca, Modelo, Versión y Año.
    Devuelve el DataFrame modificado.
    """
    df = df.copy()

    motores_limpios = []

    for idx, row in df.iterrows():
        motor = str(row['Motor']).lower().replace(',', '.') if pd.notna(row['Motor']) else ''
        match = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', motor)
        if match:
            motores_limpios.append(match.group(1))
            continue

        # Buscar en el título
        titulo = str(row['Título']).lower().replace(',', '.') if pd.notna(row['Título']) else ''
        match_titulo = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', titulo)
        if match_titulo:
            motores_limpios.append(match_titulo.group(1))
            continue

        # Buscar en otra fila con misma Marca, Modelo, Versión y Año
        filtro = (
            (df['Marca'] == row['Marca']) &
            (df['Modelo'] == row['Modelo']) &
            (df['Versión'] == row['Versión']) &
            (df['Año'] == row['Año']) &
            (df['Motor'].notna())
        )
        candidatos = df.loc[filtro, 'Motor'].dropna().astype(str).str.lower().str.replace(',', '.')
        cilindrada_encontrada = None
        for candidato in candidatos:
            match_candidato = re.search(r'(?<!\d)(\d?\.\d)(?=[^0-9]|$)', candidato)
            if match_candidato:
                cilindrada_encontrada = match_candidato.group(1)
                break

        motores_limpios.append(cilindrada_encontrada)

    df['Motor'] = motores_limpios
    return df


def convertir_pesos_a_dolares(df, valor_dolar=1250):
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

    return df
