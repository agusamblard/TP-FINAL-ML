from difflib import get_close_matches
import pandas as pd
from utils.diccionarios import MODELOS_POR_MARCA
from preprocessing.data_cleanse import normalizar
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations

def hmv_marca(df_train, df_to_input):

    df = df_to_input.copy()
    referencia = df_train[df_train['Marca'].notna()].copy()
    total_before = df.shape[0]

    df_missing = df[df['Marca'].isna()].copy()

    reemplazos = []
    for idx, row in df_missing.iterrows():
        marca_inferida = None
        modelo = row.get('Modelo')
        version = row.get('Versión')
        titulo = str(row.get('Título', '')).lower()

        # Paso 0: normalizar posibles campos
        modelo_norm = str(modelo).lower() if pd.notna(modelo) else None
        version_norm = str(version).lower() if pd.notna(version) else None

        # Paso 1: Fuzzy matching por MODELO (cutoff alto)
        if modelo_norm:
            posibles = referencia[referencia['Modelo'].notna()]
            matches = get_close_matches(modelo_norm, posibles['Modelo'].str.lower().tolist(), n=1, cutoff=0.7)
            if matches:
                match_modelo = matches[0]
                marcas = posibles[posibles['Modelo'].str.lower() == match_modelo]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]

        # Paso 2: Fuzzy matching por VERSIÓN (cutoff alto)
        if not marca_inferida and version_norm:
            posibles = referencia[referencia['Versión'].notna()]
            matches = get_close_matches(version_norm, posibles['Versión'].str.lower().tolist(), n=1, cutoff=0.8)
            if matches:
                match_version = matches[0]
                marcas = posibles[posibles['Versión'].str.lower() == match_version]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]

        # Paso 3: Segundo intento con cutoff más bajo en MODELO
        if not marca_inferida and modelo_norm:
            matches = get_close_matches(modelo_norm, referencia['Modelo'].dropna().str.lower().tolist(), n=1, cutoff=0.5)
            if matches:
                match_modelo = matches[0]
                marcas = referencia[referencia['Modelo'].str.lower() == match_modelo]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]

        # Paso 4: Buscar en TÍTULO
        if not marca_inferida:
            for m in referencia['Marca'].unique():
                if pd.notna(m) and m.lower() in titulo:
                    marca_inferida = m.lower()
                    break

        # Asignación o eliminación
        if marca_inferida:
            df.at[idx, 'Marca'] = marca_inferida
            reemplazos.append(idx)
        else:
            df.drop(index=idx, inplace=True)

    df_result = df[df['Marca'].notna()]

    return df_result


def hmv_modelo(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Modelo' de df_to_input usando df_train como referencia.

    Lógica:
    - Si hay Marca y Versión: busca versiones asociadas a modelos válidos de esa marca
    - Si no hay Marca pero sí Versión: hace fuzzy matching sobre todas las versiones
    - Si no hay Versión: elimina la fila
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Modelo'].notna() & df_train['Versión'].notna()].copy()
    total_before = df.shape[0]

    df_missing = df[df['Modelo'].isna()].copy()


    reemplazos = []
    for idx, row in df_missing.iterrows():
        modelo_inferido = None
        marca = str(row.get('Marca', '')).lower()
        version = str(row.get('Versión', '')).lower().strip()

        # Caso 1: no hay versión → eliminar
        if not version:
            df.drop(index=idx, inplace=True)
            continue

        # Caso 2: hay marca válida → restringir a modelos de esa marca
        if marca in MODELOS_POR_MARCA:
            modelos_permitidos = MODELOS_POR_MARCA[marca]
            subset = referencia[referencia['Modelo'].str.lower().isin(modelos_permitidos)]
        else:
            # Caso 3: no hay marca o marca desconocida → usar todo el entrenamiento
            subset = referencia

        if subset.empty:
            df.drop(index=idx, inplace=True)
            continue

        # Fuzzy matching con versiones
        versiones_entrenamiento = subset['Versión'].str.lower().tolist()
        matches = get_close_matches(version, versiones_entrenamiento, n=1, cutoff=0.7)

        if matches:
            match_version = matches[0]
            modelos_encontrados = subset[subset['Versión'].str.lower() == match_version]['Modelo']
            if not modelos_encontrados.empty:
                modelo_inferido = modelos_encontrados.mode().iloc[0]

        if modelo_inferido:
            df.at[idx, 'Modelo'] = modelo_inferido
            reemplazos.append(idx)
        else:
            df.drop(index=idx, inplace=True)

    df_result = df[df['Modelo'].notna()]
    return df_result



def hmv_version_train(df_train, df_to_input):
    """
    Imputa y agrupa versiones en df_to_input utilizando df_train como referencia.

    Paso 1: Imputación
        - Si hay Modelo → fuzzy matching con versiones de ese modelo en el train
        - Si no hay Modelo pero sí Marca → buscar versiones asociadas a los modelos de esa marca
        - Si no hay Marca ni Modelo → eliminar fila

    Paso 2: Fuzzy clustering por modelo (agrupa si comparten tokens similares)
        - Agrupa versiones similares dentro de cada modelo
        - Asigna el nombre más frecuente del grupo

    Paso 3: Agrupamiento final entre versiones resultantes del mismo modelo
        - Reagrupa versiones normalizadas que aún sean similares entre sí
        - Usa fuzzy matching sobre strings completas

    COSAS A TENER EN CUENTA:
    - El paso 1 puede ser mejorado. Ahora solo aplica la moda, pero podría ser más sofisticado teniendo en cuenta el precio, año
     kilometraje, etc.
     - El paso 2 y 3 agrupan algunos casos que no deberian agruparse. Son similares semanticamente, pero tienen diferencias 
     importantes en el precio. Por ejemplo advance y advance plus.
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Versión'].notna()].copy()

    df_missing = df[df['Versión'].isna()].copy()

    reemplazos = []
    for idx, row in df_missing.iterrows():
        modelo = str(row.get("Modelo", "")).lower()
        marca = str(row.get("Marca", "")).lower()

        if modelo:
            versiones_posibles = referencia[referencia["Modelo"].str.lower() == modelo]["Versión"].dropna()
        elif marca in MODELOS_POR_MARCA:
            modelos_de_marca = MODELOS_POR_MARCA[marca]
            versiones_posibles = referencia[referencia["Modelo"].str.lower().isin(modelos_de_marca)]["Versión"].dropna()
        else:
            df.drop(index=idx, inplace=True)
            continue

        if versiones_posibles.empty:
            df.drop(index=idx, inplace=True)
            continue

        version_frecuente = versiones_posibles.mode().iloc[0]
        df.at[idx, "Versión"] = version_frecuente
        reemplazos.append(idx)

    df = df[df['Versión'].notna()]  # Eliminar filas sin imputar

    # Helper: chequea si dos cadenas comparten tokens similares
    def tokens_comparten_similitud(base, candidato, cutoff=0.8):
        tokens1 = base.replace('-', ' ').replace('_', ' ').lower().split()
        tokens2 = candidato.replace('-', ' ').replace('_', ' ').lower().split()
        for t1 in tokens1:
            if t1 in tokens2:
                return True
            if get_close_matches(t1, tokens2, n=1, cutoff=cutoff):
                return True
        return False

    # Paso 2: Clustering de versiones dentro de cada modelo
    versiones_normalizadas = {}

    for modelo, sub in df[['Modelo', 'Versión']].dropna().groupby('Modelo'):
        versiones = sub['Versión'].dropna().unique().tolist()
        mapeo = {}


        while versiones:
            base = versiones.pop(0)
            similares = []

            for v in versiones:
                if tokens_comparten_similitud(base, v, cutoff=0.8):
                    similares.append(v)

            grupo = [base] + similares
            versiones = [v for v in versiones if v not in similares]

            subset = sub[sub['Versión'].isin(grupo)]
            version_mas_comun = subset['Versión'].mode().iloc[0]



            for v in grupo:
                mapeo[v] = version_mas_comun

        for original, reemplazo in mapeo.items():
            versiones_normalizadas[(modelo, original)] = reemplazo

    # Aplicar los reemplazos de clustering
    def reemplazar_version(row):
        clave = (row['Modelo'], row['Versión'])
        return versiones_normalizadas.get(clave, row['Versión'])

    df['Versión'] = df.apply(reemplazar_version, axis=1)

    # Paso 3: Agrupamiento final entre versiones ya normalizadas (por modelo)

    def tokens_similares(v1, v2, cutoff=0.8):
        """
        Devuelve True si v1 y v2 comparten tokens similares (no necesita ser exacto).
        """
        tokens1 = v1.replace('-', ' ').replace('_', ' ').lower().split()
        tokens2 = v2.replace('-', ' ').replace('_', ' ').lower().split()
        for t1 in tokens1:
            if t1 in tokens2:
                return True
            if get_close_matches(t1, tokens2, n=1, cutoff=cutoff):
                return True
        return False

    for modelo, sub in df[['Modelo', 'Versión']].dropna().groupby('Modelo'):
        versiones_finales = sub['Versión'].unique().tolist()
        finales_map = {}


        while versiones_finales:
            base = versiones_finales.pop(0)
            similares = []

            for v in versiones_finales:
                if tokens_similares(base, v, cutoff=0.8):
                    similares.append(v)

            grupo = [base] + similares
            versiones_finales = [v for v in versiones_finales if v not in similares]

            subset = sub[sub['Versión'].isin(grupo)]
            version_final = subset['Versión'].mode().iloc[0]


            for v in grupo:
                finales_map[(modelo, v)] = version_final

        # Aplicar mapeo final
        def ajustar_version_final(row):
            clave = (row['Modelo'], row['Versión'])
            return finales_map.get(clave, row['Versión'])

        df['Versión'] = df.apply(ajustar_version_final, axis=1)

    return df


def hmv_version(df_train, df_to_input):
    """
    Ejecuta limpieza y agrupamiento de versiones:
    - Primero se aplica la función completa sobre df_train.
    - Luego, se genera un diccionario de versiones válidas por marca y modelo.
    - Finalmente, se corrige df_to_input con ese conocimiento (imputación y similitud).

    Retorna df_to_input actualizado.
    """


    df_train_limpio = hmv_version_train(df_train, df_train)

    # Diccionario definitivo de versiones por (Marca, Modelo)
    versiones_por_modelo = {}
    for (marca, modelo), sub in df_train_limpio.groupby(['Marca', 'Modelo']):
        versiones_por_modelo[(marca, modelo)] = (
            sub['Versión'].dropna()
               .value_counts()
               .index
               .tolist()  
        )
    df_val = df_to_input.copy()
    referencia = df_train_limpio[df_train_limpio['Versión'].notna()].copy()

    df_missing = df_val[df_val['Versión'].isna()].copy()


    reemplazos = []
    eliminar = []

    # Paso 1: Imputar versiones faltantes según prioridad
    for idx, row in df_missing.iterrows():
        marca = str(row.get("Marca", "")).lower()
        modelo = str(row.get("Modelo", "")).lower()

        version_frecuente = None

        if marca and modelo:
            versiones_posibles = referencia[
                (referencia["Marca"].str.lower() == marca) &
                (referencia["Modelo"].str.lower() == modelo)
            ]["Versión"].dropna()
            contexto = "marca + modelo"

        elif marca:
            versiones_posibles = referencia[
                referencia["Marca"].str.lower() == marca
            ]["Versión"].dropna()
            contexto = "solo marca"

        elif modelo:
            versiones_posibles = referencia[
                referencia["Modelo"].str.lower() == modelo
            ]["Versión"].dropna()
            contexto = "solo modelo"

        else:
            versiones_posibles = pd.Series(dtype=str)
            contexto = "sin marca ni modelo"

        if not versiones_posibles.empty:
            version_frecuente = versiones_posibles.mode().iloc[0]
            df_val.at[idx, "Versión"] = version_frecuente
            reemplazos.append(idx)
        else:
            df_val.drop(index=idx, inplace=True)





    def tokens_similares(v1, v2, cutoff=0.75):
        """
        Devuelve True si al menos un token de v1 se parece (por fuzzy matching) a algún token de v2.
        """
        t1 = str(v1).replace('-', ' ').replace('_', ' ').lower().split()
        t2 = str(v2).replace('-', ' ').replace('_', ' ').lower().split()
        for token1 in t1:
            if token1 in t2 or get_close_matches(token1, t2, n=1, cutoff=cutoff):
                return True
        return False

    for idx, row in df_val.iterrows():
        marca = row.get('Marca')
        modelo = row.get('Modelo')
        version = row.get('Versión')
        key = (marca, modelo)
        versiones_validas = versiones_por_modelo.get(key, [])

        if not versiones_validas:
            eliminar.append(idx)
            continue

        if version not in versiones_validas:
            # Buscar por similitud de tokens
            candidato = None
            for v in versiones_validas:
                if tokens_similares(version, v, cutoff=0.75):
                    candidato = v
                    break

            if candidato:
                df_val.at[idx, 'Versión'] = candidato
            else:
                eliminar.append(idx)

    if eliminar:
        df_val.drop(index=eliminar, inplace=True)
    return df_val.reset_index(drop=True)

def hmv_combustible(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Tipo de combustible' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles (Marca, Modelo, Versión, Año).
    - Si no encuentra coincidencias, reduce las columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    """


    df = df_to_input.copy()
    referencia = df_train[df_train['Tipo de combustible'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Tipo de combustible'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(col in subset for col in ['Modelo', 'Versión']):
                    continue

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['Tipo de combustible'].mode()
                    if not moda.empty:
                        df.at[idx, 'Tipo de combustible'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.drop(index=idx, inplace=True)

    return df

def hmv_puertas(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Puertas' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles (Marca, Modelo, Versión, Año).
    - Si no encuentra coincidencias, reduce las columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    """


    df = df_to_input.copy()
    referencia = df_train[df_train['Puertas'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Puertas'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(col in subset for col in ['Modelo', 'Versión']):
                    continue

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['Puertas'].mode()
                    if not moda.empty:
                        df.at[idx, 'Puertas'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.drop(index=idx, inplace=True)

    return df

def hmv_transmision(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Transmisión' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para poder imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles.
    - Si no encuentra coincidencias, reduce el conjunto de columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    """


    df = df_to_input.copy()
    referencia = df_train[df_train['Transmisión'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Transmisión'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(col in subset for col in ['Modelo', 'Versión']):
                    continue

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['Transmisión'].mode()
                    if not moda.empty:
                        df.at[idx, 'Transmisión'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.drop(index=idx, inplace=True)

    return df



def hmv_motor(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Motor' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para poder imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles.
    - Si no encuentra coincidencias, reduce el conjunto de columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Motor'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Motor'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(col in subset for col in ['Modelo', 'Versión']):
                    continue  

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['Motor'].mode()
                    if not moda.empty:
                        df.at[idx, 'Motor'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.drop(index=idx, inplace=True)

    return df

def hmv_camara(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Con cámara de retroceso' de df_to_input usando df_train como referencia.

    Lógica:
    - Solo intenta imputar si están presentes Marca, Modelo, Versión y Año.
    - Busca coincidencias exactas con esas cuatro columnas en df_train o imputaciones anteriores.
    - Si hay coincidencias, asigna la moda.
    - Si no hay coincidencias, asigna 0.
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Con cámara de retroceso'].notna()].copy()
    imputados = 0
    default_0 = 0

    for idx, row in df[df['Con cámara de retroceso'].isna()].iterrows():
        if any(pd.isna(row[col]) for col in ['Marca', 'Modelo', 'Versión', 'Año']):
            df.at[idx, 'Con cámara de retroceso'] = 0
            default_0 += 1
            continue

        filtro = referencia[
            (referencia['Marca'] == row['Marca']) &
            (referencia['Modelo'] == row['Modelo']) &
            (referencia['Versión'] == row['Versión']) &
            (referencia['Año'] == row['Año'])
        ]



        if not filtro.empty:
            moda = filtro['Con cámara de retroceso'].mode()
            if not moda.empty:
                valor = moda.iloc[0]
                df.at[idx, 'Con cámara de retroceso'] = valor

                imputados += 1
            else:
                df.at[idx, 'Con cámara de retroceso'] = 0

                default_0 += 1
        else:
            df.at[idx, 'Con cámara de retroceso'] = 0

            default_0 += 1


    return df


def hmv_hp(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'HP' de df_to_input usando df_train como referencia.

    Lógica:
    - Busca combinaciones exactas con columnas: Motor, Versión, Modelo, Año, Marca (en ese orden de prioridad).
    - Evalúa combinaciones desde 5 hasta 1 columna, priorizando las que incluyan 'Motor'.
    - Requiere al menos uno entre 'Modelo', 'Versión' o 'Motor'.
    - Imputa la moda si encuentra coincidencias.
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['HP'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Motor', 'Versión', 'Modelo', 'Año', 'Marca']

    for idx, row in df[df['HP'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión', 'Motor']):
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            posibles_subsets = list(combinations(disponibles, k))
            posibles_subsets.sort(key=lambda s: 'Motor' not in s)

            for subset in posibles_subsets:
                if not any(c in subset for c in ['Modelo', 'Versión', 'Motor']):
                    continue

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['HP'].mode()
                    if not moda.empty:
                        df.at[idx, 'HP'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.drop(index=idx, inplace=True)

    return df



def hmv_traccion(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Tracción' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión'.
    - Busca coincidencias exactas con combinaciones de columnas disponibles: Marca, Modelo, Versión, Año.
    - Evalúa combinaciones desde 4 hasta 1 columna.
    - Si no encuentra coincidencias, asigna "4x2" como valor por defecto.
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Tracción'].notna()].copy()
    imputados = 0
    asignados_default = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Tracción'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            df.at[idx, 'Tracción'] = '4x2'
            asignados_default += 1
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(c in subset for c in ['Modelo', 'Versión']):
                    continue 

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]


                if not filtro.empty:
                    moda = filtro['Tracción'].mode()
                    if not moda.empty:
                        df.at[idx, 'Tracción'] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.at[idx, 'Tracción'] = '4x2'
            asignados_default += 1


    return df



def hmv_year(df_train, df_to_input):
    df = df_to_input.copy()
    referencia = df_train[df_train['Año'].notna()].copy()
    imputados = 0

    for idx, row in df[df['Año'].isna()].iterrows():


        if pd.isna(row['Kilómetros']) or pd.isna(row['Con cámara de retroceso']):
            continue

        condiciones = (
            (referencia['Marca'] == row['Marca']) &
            (referencia['Modelo'] == row['Modelo']) &
            (referencia['Versión'] == row['Versión']) &
            referencia['Kilómetros'].notna() &
            referencia['Con cámara de retroceso'].notna()
        )

        candidatos = referencia[condiciones]

        if len(candidatos) < 5:
            continue

        X = candidatos[['Kilómetros', 'Con cámara de retroceso']]
        y = candidatos['Año']

        modelo = RandomForestRegressor(random_state=42)
        modelo.fit(X, y)

        X_pred = [[row['Kilómetros'], row['Con cámara de retroceso']]]
        pred = modelo.predict(X_pred)[0]
        pred_redondeado = int(round(pred))

        df.at[idx, 'Año'] = pred_redondeado
        imputados += 1

    return df


def hmv_km(df_train, df_to_input, min_size=15, max_ext=10):
    """
    Detecta outliers en 'Kilómetros' de df_to_input basándose en la distribución de df_train.
    Marca los valores outlier como NaN en df_to_input y los imputa con la media del año
    (expandiendo hacia años vecinos si es necesario).

    Returns:
        df_result: copia de df_to_input con valores imputados en 'Kilómetros'
    """


    def ajustar_rangos_iqr(grupo, año):
        """
        Calcula límites de kilometraje adaptativos en función de la antigüedad del auto.
        """
        Q1 = grupo['Kilómetros'].quantile(0.30)
        Q3 = grupo['Kilómetros'].quantile(0.70)
        IQR = Q3 - Q1
        antiguedad = 2025 - año

        if antiguedad >= 30:
            factor_lower = 1.5
            factor_upper = 1.0
        elif antiguedad >= 20:
            factor_lower = 1.3
            factor_upper = 1.2
        elif antiguedad >= 10:
            factor_lower = 1.1
            factor_upper = 1.3
        elif antiguedad >= 3:
            factor_lower = 1.0
            factor_upper = 1.5
        else:
            factor_lower = 0.8
            factor_upper = 1.75

        lower = max(0, Q1 - factor_lower * IQR)
        upper = Q3 + factor_upper * IQR

        if antiguedad <= 1:
            upper = max(upper, 10000)

        return lower, upper

    df_result = df_to_input.copy()
    años_unicos = sorted(df_result['Año'].dropna().unique())
    evaluados = set()

    outliers_total = 0
    imputados_total = 0
    no_imputados = 0


    for año in años_unicos:
        if año in evaluados:
            continue

        año = int(año)
        ext = 0
        grupo = pd.DataFrame()
        while ext <= max_ext:
            rango = list(range(año - ext, año + ext + 1))
            grupo = df_train[df_train['Año'].isin(rango)]
            if len(grupo) >= min_size:
                break
            ext += 1

        if len(grupo) < min_size:
            continue

        lower, upper = ajustar_rangos_iqr(grupo, año)
        evaluados.update(rango)


        cond_outlier = (
            (df_result['Año'] == año) &
            ((df_result['Kilómetros'] < lower) | (df_result['Kilómetros'] > upper))
        )
        outliers_detectados = cond_outlier.sum()
        outliers_total += outliers_detectados
        df_result.loc[cond_outlier, 'Kilómetros'] = pd.NA



    for idx, row in df_result[df_result['Kilómetros'].isna()].iterrows():
        año = int(row['Año'])
        ext = 0
        grupo = pd.DataFrame()

        while ext <= max_ext:
            rango = list(range(año - ext, año + ext + 1))
            grupo = df_train[
                (df_train['Año'].isin(rango)) &
                (df_train['Kilómetros'].notna())
            ]
            if len(grupo) >= min_size:
                break
            ext += 1

        if len(grupo) >= min_size:
            imputado = round(grupo['Kilómetros'].median())
            df_result.at[idx, 'Kilómetros'] = imputado
            imputados_total += 1
        else:
            no_imputados += 1



    return df_result





def hmv_tipo_de_vendedor(df_train, df_to_input):
    """
    Imputa valores faltantes en 'Tipo de vendedor' de df_to_input usando como referencia df_train.
    Para cada valor NaN, busca la moda en el grupo coincidente por Marca + Modelo + Versión + Año.

    Args:
        df_train: DataFrame de referencia (con valores conocidos).
        df_to_input: DataFrame a imputar.

    Returns:
        df_result: copia de df_to_input con imputaciones aplicadas.
    """


    df_result = df_to_input.copy()
    imputados = 0

    for idx, row in df_result[df_result['Tipo de vendedor'].isna()].iterrows():
        marca = row['Marca']
        modelo = row['Modelo']
        version = row['Versión']
        año = row['Año']

        grupo = df_train[
            (df_train['Marca'] == marca) &
            (df_train['Modelo'] == modelo) &
            (df_train['Versión'] == version) &
            (df_train['Año'] == año) &
            (df_train['Tipo de vendedor'].notna())
        ]

        if not grupo.empty:
            moda = grupo['Tipo de vendedor'].mode().iloc[0]
            df_result.at[idx, 'Tipo de vendedor'] = moda
            imputados += 1


    return df_result

def hmv_color_train(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Color' y agrupa valores similares basándose en tokens con similitud textual.
    Orden de operaciones:
    1. Imputación de valores faltantes en 'Color' en df_to_input usando df_train (prioridad: Versión > Modelo > Marca)
    2. Reemplazo de 'morado' por 'violeta' en ambos datasets
    3. Agrupamiento de colores por tokens similares usando df_train
    4. Reemplazo de valores conocidos en df_to_input según el agrupamiento
    """

    # Copia de df_to_input
    df_result = df_to_input.copy()

    #Paso 1: Imputando valores faltantes con prioridad Versión → Modelo → Marca

    imputados = 0
    for idx, row in df_result[df_result['Color'].isna()].iterrows():
        version = row.get('Versión')
        modelo = row.get('Modelo')
        marca = row.get('Marca')

        valor_color = None

        # Intento 1: por versión (usar df_train)
        if pd.notna(version):
            coincidencias = df_train[df_train['Versión'] == version]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'versión'

        # Intento 2: por modelo (usar df_train)
        if valor_color is None and pd.notna(modelo):
            coincidencias = df_train[df_train['Modelo'] == modelo]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'modelo'

        # Intento 3: por marca (usar df_train)
        if valor_color is None and pd.notna(marca):
            coincidencias = df_train[df_train['Marca'] == marca]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'marca'

        if valor_color is not None:
            df_result.at[idx, 'Color'] = valor_color
            imputados += 1



    #  Reemplazando 'morado' por 'violeta' en ambos datasets. sino me lo escribe como dorado y esta mal
    for df in [df_train, df_result]:
        df['Color'] = df['Color'].apply(lambda c: 'violeta' if isinstance(c, str) and 'morado' in normalizar(c, eliminar_espacios=False) else c)

    # 🔍 Paso 3: Agrupando colores por tokens similares
    colores = df_train['Color'].dropna().unique()
    token_map = {}
    color_groups = {}

    for color in colores:
        tokens = normalizar(color, eliminar_espacios=False).split()
        if not tokens:
            continue
        first_token = tokens[0].strip()
        if not first_token:
            continue
        similars = get_close_matches(first_token, token_map.keys(), n=1, cutoff=0.7)
        if similars:
            group_key = token_map[similars[0]]
        else:
            group_key = first_token
        token_map[first_token] = group_key
        color_groups.setdefault(group_key, set()).add(color)




    color_map = {}
    for grupo, variantes in color_groups.items():
        for variante in variantes:
            color_map[variante] = grupo

    df_result['Color'] = df_result['Color'].map(lambda x: color_map.get(x, x))

    return df_result


def hmv_color(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Color' y agrupa valores similares basándose en tokens con similitud textual.
    Orden de operaciones:
    1. Imputación de valores faltantes en 'Color' en df_to_input usando df_train (prioridad: Versión > Modelo > Marca)
    2. Reemplazo de 'morado' por 'violeta' en ambos datasets
    3. Agrupamiento de colores por tokens similares usando df_train
    4. Reemplazo de valores conocidos en df_to_input según el agrupamiento
    """

    # ------------------------------------------------------------------ #
    # -----------------  PARTE A: APRENDER DESDE df_train --------------- #
    # ------------------------------------------------------------------ #
    # Limpiar df_train usando la misma función recursivamente
    df_train_limpio = hmv_color_train(df_train, df_train)
    colores_validos = df_train_limpio['Color'].dropna().unique().tolist()

    # Copia de df_to_input
    df_result = df_to_input.copy()

    # Paso 1: Imputando valores faltantes con prioridad Versión → Modelo → Marca
    imputados = 0
    for idx, row in df_result[df_result['Color'].isna()].iterrows():
        version = row.get('Versión')
        modelo = row.get('Modelo')
        marca = row.get('Marca')

        valor_color = None

        # Intento 1: por versión (usar df_train_limpio)
        if pd.notna(version):
            coincidencias = df_train_limpio[df_train_limpio['Versión'] == version]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'versión'

        # Intento 2: por modelo (usar df_train_limpio)
        if valor_color is None and pd.notna(modelo):
            coincidencias = df_train_limpio[df_train_limpio['Modelo'] == modelo]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'modelo'

        # Intento 3: por marca (usar df_train_limpio)
        if valor_color is None and pd.notna(marca):
            coincidencias = df_train_limpio[df_train_limpio['Marca'] == marca]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'marca'

        if valor_color is not None:
            df_result.at[idx, 'Color'] = valor_color
            imputados += 1



    for df in [df_train_limpio, df_result]:
        df['Color'] = df['Color'].apply(lambda c: 'violeta' if isinstance(c, str) and 'morado' in normalizar(c, eliminar_espacios=False) else c)

    #  Aprender colores válidos desde df_train_limpio ya procesado
    colores_validos = df_train_limpio['Color'].dropna().unique()
    color_tokens = {}
    for color in colores_validos:
        tokens = normalizar(color, eliminar_espacios=False).split()
        if not tokens:
            continue
        first_token = tokens[0]
        color_tokens[color] = first_token



    def token_similar(c1, c2, cutoff=0.75):
        t1 = normalizar(c1, eliminar_espacios=False).split()
        t2 = normalizar(c2, eliminar_espacios=False).split()
        for token1 in t1:
            if token1 in t2 or get_close_matches(token1, t2, n=1, cutoff=cutoff):
                return True
        return False

    colores_resultantes = []
    for idx, val in df_result['Color'].items():
        if pd.isna(val):
            colores_resultantes.append(val)
            continue
        if val in colores_validos:
            colores_resultantes.append(val)
            continue
        match = None
        for c_valido in colores_validos:
            if token_similar(val, c_valido, cutoff=0.75):
                match = c_valido
                break
        if match:

            colores_resultantes.append(match)
        else:

            colores_resultantes.append("otro")

    df_result['Color'] = colores_resultantes

    return df_result




def hmv_dataset(df_train, df_to_input):
    """
    Ejecuta en orden todas las funciones hmv_* para imputar datos faltantes
    sobre df_to_input, utilizando df_train como referencia.
    """
    original_count = len(df_to_input)
    df = df_to_input.copy()

    # Lista de funciones de imputación hmv
    hmv_funcs = [
        ("hmv_marca", hmv_marca),
        ("hmv_modelo", hmv_modelo),
        ("hmv_version", hmv_version),
        ("hmv_combustible", hmv_combustible),
        ("hmv_puertas", hmv_puertas),
        ("hmv_transmision", hmv_transmision),
        ("hmv_motor", hmv_motor),
        ("hmv_camara", hmv_camara),
        ("hmv_hp", hmv_hp),
        ("hmv_traccion", hmv_traccion),
        ("hmv_year", hmv_year),
        ("hmv_km", hmv_km),
        ("hmv_tipo_de_vendedor", hmv_tipo_de_vendedor),  
        ("hmv_color", hmv_color),
    ]

    for func_name, func in hmv_funcs:
        count_before = len(df)
        df = func(df_train, df)
        count_after = len(df)




    return df