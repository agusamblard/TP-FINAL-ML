import unidecode
import pandas as pd
from difflib import get_close_matches
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN


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



# Diccionario agrupado por marca
modelos_por_marca = {
    'ford': ['ecosport', 'territory', 'kuga', 'bronco sport', 'explorer', 'bronco'],
    'chevrolet': ['tracker', 'trailblazer', 'equinox', 'spin', 'blazer', 'grand blazer'],
    'peugeot': ['2008', '3008', '4008'],
    'renault': ['duster', 'captur', 'duster oroch', 'koleos', 'sandero stepway'],
    'jeep': ['compass', 'renegade', 'grand cherokee', 'commander', 'wrangler', 'cherokee', 'patriot'],
    'nissan': ['kicks', 'x-trail', 'murano', 'pathfinder', 'x-terra', 'terrano ii'],
    'volkswagen': ['taos', 't-cross', 'tiguan', 'tiguan allspace', 'touareg'],
    'toyota': ['corolla cross', 'hilux sw4', 'sw4', 'rav4', 'land cruiser', '4runner'],
    'citroen': ['c4 cactus', 'c3 aircross', 'c5 aircross', 'c3', 'c4 aircross'],
    'hyundai': ['tucson', 'santa fe', 'creta', 'x35', 'galloper', 'kona', 'grand santa fé'],
    'fiat': ['pulse', '500x'],
    'honda': ['hr-v', 'cr-v', 'pilot'],
    'bmw': ['x1', 'x3', 'x5', 'x6', 'x4', 'x2', 'serie 4'],
    'audi': ['q5', 'q3', 'q2', 'q7', 'q3 sportback', 'q8', 'sq5', 'q5 sportback'],
    'kia': ['sportage', 'soul', 'sorento', 'seltos', 'mohave'],
    'baic': ['x55', 'x25'],
    'mercedes-benz': ['clase glc', 'clase gla', 'clase gle', 'clase glk', 'clase ml', 'clase gl', 'ml'],
    'chery': ['tiggo', 'tiggo 3', 'tiggo 4 pro', 'tiggo 5', 'tiggo 2', 'tiggo 4', 'tiggo 8 pro', 's2'],
    'dodge': ['journey'],
    'land rover': ['evoque', 'range rover sport', 'discovery', 'range rover', 'freelander', 'defender'],
    'suzuki': ['grand vitara', 'vitara', 'jimny', 'samurai'],
    'porsche': ['cayenne', 'macan', 'panamera'],
    'volvo': ['xc60', 'xc40'],
    'd.s.': ['ds7 crossback', 'ds7', 'ds3'],
    'ssangyong': ['musso', 'actyon'],
    'alfa romeo': ['stelvio'],
    'jetour': ['x70'],
    'gwm': ['jolion'],
    'isuzu': ['trooper'],
    'lifan': ['myway'],
    'lexus': ['ux', 'nx'],
    'subaru': ['outback'],
    'daihatsu': ['terios'],
    'mini': ['cooper countryman'],
    'mitsubishi': ['outlander', 'montero', 'nativa'],
    'jaguar': ['f-pace']
}

# Plano: modelo → marca
marca_por_modelo = {
    modelo: marca
    for marca, modelos in modelos_por_marca.items()
    for modelo in modelos
}

# Lista de modelos válidos
modelos_validos = list(marca_por_modelo.keys())

# Función principal
def corregir_modelo(df):
    df = df.copy()
    nuevos_modelos = []

    for _, row in df.iterrows():
        modelo = row['Modelo']
        marca = row['Marca']

        if pd.isna(modelo) or pd.isna(marca):
            nuevos_modelos.append('Otros')
            continue

        modelo_clean = unidecode.unidecode(str(modelo)).lower().strip()
        marca_clean = unidecode.unidecode(str(marca)).lower().strip()

        if modelo_clean in modelos_validos and marca_clean == marca_por_modelo[modelo_clean]:
            nuevos_modelos.append(modelo.title())
        else:
            modelos_de_marca = [m for m in modelos_validos if marca_por_modelo[m] == marca_clean]
            coincidencias = get_close_matches(modelo_clean, modelos_de_marca, n=1, cutoff=0.7)
            if coincidencias:
                nuevos_modelos.append(coincidencias[0].title())
            else:
                nuevos_modelos.append('Otros')

    df['Modelo'] = nuevos_modelos
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


# Listas agrupadas por tipo
automaticos = [
    'at', '6at', '8at', 'at6', 'atx', 'cvt', 'tiptronic', 'stronic',
    'dsg', 'automática', 'automatic', 'tronic', 'aut', 'automatico', 'automático'
]
manuales = ['mt', '6mt', 'manual']

# Lista total de términos
terminos_transmision = automaticos + manuales

# Mapa de normalización generado automáticamente
mapa_normalizado = {term: 'AUTOMÁTICA' for term in automaticos}
mapa_normalizado.update({term: 'MANUAL' for term in manuales})


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
    Guarda la versión original en 'Versión_prev', limpia 'Versión' y devuelve el DataFrame modificado.
    """
    df = df.copy()
    df['Versión_prev'] = df['Versión']  # guardar versión original

    terminos_traccion = ['4x4','awd', '4matic', 'quattro', '4m', '4wd', 'xdrive',
                         '2x4','4x2','fwd', 'rwd', '2wd']
    
    terminos_transmision = [
        'at', '6at', '8at', 'at6', 'atx', 'cvt', 'tiptronic', 'stronic',
        'dsg', 'automatica', 'automatic', 'tronic', 'aut',
        'mt', '6mt', 'manual', 'automatico', 'automático'
    ]


    terminos_combustible = [
        'electrico', 'electrica', 'electric', 'electrical',
        'hibrido', 'hibrid', 'hybrid', 'hibrida', 'mhev', 'hev', 'phev', 'hv', 'mild hybrid',
        'diesel', 'gasoil',
        'nafta', 'naftero', 'nafta/gnc',
        'gnc'
    ]

    # Regex para eliminar frases como "5p", "7 asientos", "3 puertas", etc.
    regex_asientos = re.compile(r'\b[1-9]\s?(as|p|puertas|plazas|pasajeros|asientos|pas)\b', re.IGNORECASE)

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

        # Eliminar motor y combustible
        version_limpia = re.sub(r'\b(' + '|'.join(map(re.escape, terminos_combustible)) + r')\b', '', version_limpia)

        # Eliminar expresiones de cantidad de asientos
        version_limpia = regex_asientos.sub('', version_limpia)

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


def revisar_tipos_combustible(df):
    import re

    tipos_config = {
        'Eléctrico': ['electrico', 'electrica', 'electric', 'electrical'],
        'Híbrido': ['hibrido', 'hibrid', 'hybrid', 'hibrida', 'mhev', 'hev', 'phev', 'hv', 'mild hybrid'],
        'Diésel': ['diesel', 'gasoil'],
        'Nafta': ['nafta', 'naftero'],
        'Nafta/GNC': ['gnc']
    }

    prioridad = ['Eléctrico', 'Híbrido', 'Nafta/GNC', 'Diésel', 'Nafta']

    regex_por_tipo = {
        tipo: re.compile(r'\b(' + '|'.join(map(re.escape, palabras)) + r')\b', re.IGNORECASE)
        for tipo, palabras in tipos_config.items()
    }

    df = df.copy()
    df['Tipo original'] = df['Tipo de combustible']

    # --- Paso 0: Normalizar valores atípicos en 'Tipo de combustible' ---
    for idx, fila in df.iterrows():
        tipo_actual = str(fila['Tipo de combustible']).lower()
        tipo_actual_sin_tildes = quitar_tildes(tipo_actual)

        tipos_detectados = []
        for tipo, palabras in tipos_config.items():
            for palabra in palabras:
                if palabra in tipo_actual_sin_tildes:
                    tipos_detectados.append(tipo)
                    break

        for tipo_prioritario in prioridad:
            if tipo_prioritario in tipos_detectados:
                df.at[idx, 'Tipo de combustible'] = tipo_prioritario
                break
        
    df = df.copy()
    df['Tipo original'] = df['Tipo de combustible']


    # --- Parte 1: Detectar tipo a partir del texto completo ---
    def detectar_tipo(fila):
        texto = ' '.join(
            quitar_tildes(str(v).lower())
            for k, v in fila.items()
            if k not in ['Descripción', 'Tipo de combustible', 'Tipo original'] and pd.notna(v)
        )

        tipos_detectados = [tipo for tipo, regex in regex_por_tipo.items() if regex.search(texto)]
        for tipo_prioritario in prioridad:
            if tipo_prioritario in tipos_detectados:
                return tipo_prioritario

        return fila['Tipo de combustible']

    df['Tipo de combustible'] = df.apply(detectar_tipo, axis=1)

    # --- Parte 2: Corregir falsos positivos conservadores ---
    for tipo_objetivo in prioridad:
        regex_tipo = regex_por_tipo[tipo_objetivo]
        for idx, fila in df.iterrows():
            if fila['Tipo original'] == tipo_objetivo and fila['Tipo de combustible'] == tipo_objetivo:
                texto = ' '.join(
                    quitar_tildes(str(v).lower())
                    for k, v in fila.items()
                    if k not in ['Descripción', 'Tipo de combustible', 'Tipo original'] and pd.notna(v)
                )
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

    return df


