from difflib import get_close_matches
import pandas as pd
from utils.diccionarios import MODELOS_POR_MARCA
from itertools import combinations
from preprocessing.utils import tokens_similares, normalizar
import numpy as np
from sklearn.ensemble import RandomForestRegressor



# ------------------  FUNCION GENERICA DE INPUTACION ------------------ #
def imputar_por_contexto(df_train, df_to_input, col_target, columnas_importancia, fallback_valor=None):
    """
    Imputa valores faltantes de `col_target` en df_to_input basándose en coincidencias
    exactas con df_train y, opcionalmente, asigna un valor por defecto.
    """
    df = df_to_input.copy()
    referencia = df_train[df_train[col_target].notna()].copy()
    imputados = 0
    fallback_modelo = 0
    fallback_global = 0
    asignados_default = 0

    for idx, row in df[df[col_target].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        # Necesita al menos Modelo o Versión
        requiere_contexto = any(c in disponibles for c in ['Modelo', 'Versión'])
        if not disponibles or not requiere_contexto:
            if fallback_valor is not None:
                df.at[idx, col_target] = fallback_valor
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
                    moda = filtro[col_target].mode()
                    if not moda.empty:
                        df.at[idx, col_target] = moda.iloc[0]
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if imputado:
            continue  # ya imputado

        modelo = row.get('Modelo')
        if modelo:
            posibles = referencia[referencia['Modelo'] == modelo][col_target]
            if not posibles.empty:
                moda_modelo = posibles.mode()
                if not moda_modelo.empty:
                    df.at[idx, col_target] = moda_modelo.iloc[0]
                    fallback_modelo += 1
                    continue

        # Fallback global
        moda_global = referencia[col_target].mode()
        if not moda_global.empty:
            df.at[idx, col_target] = moda_global.iloc[0]
            fallback_global += 1
        elif fallback_valor is not None:
            df.at[idx, col_target] = fallback_valor
            asignados_default += 1

    return df

def hmv_marca(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Marca' en df_to_input usando df_train como referencia.
    - Primero intenta imputar por 'Modelo'
    - Luego intenta con 'Versión'
    - Si falla, intenta con cutoff más bajo
    - Luego intenta detectar marca en 'Título'

    """
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

        # Paso 1: por MODELO (cutoff alto)
        if modelo_norm:
            posibles = referencia[referencia['Modelo'].notna()]
            matches = get_close_matches(modelo_norm, posibles['Modelo'].str.lower().tolist(), n=1, cutoff=0.7)
            if matches:
                match_modelo = matches[0]
                marcas = posibles[posibles['Modelo'].str.lower() == match_modelo]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]

        # Paso 2: por VERSIÓN (cutoff alto)
        if not marca_inferida and version_norm:
            posibles = referencia[referencia['Versión'].notna()]
            matches = get_close_matches(version_norm, posibles['Versión'].str.lower().tolist(), n=1, cutoff=0.8)
            if matches:
                match_version = matches[0]
                marcas = posibles[posibles['Versión'].str.lower() == match_version]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]

        # Paso 3: intento mas laxo por MODELO (cutoff bajo)
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
    - Si no hay Marca pero sí Versión: busca coincidencias sobre todas las versiones
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



def hmv_version(df_train, df_to_input):

    # ---------- PARTE A: LIMPIEZA SOBRE df_train ----------
    df_train_proc = df_train.copy()
    referencia = df_train_proc[df_train_proc['Versión'].notna()].copy()

    # Clustering de versiones por modelo
    versiones_normalizadas = {}
    for modelo, sub in referencia[['Modelo', 'Versión']].dropna().groupby('Modelo'):
        versiones = sub['Versión'].unique().tolist()
        while versiones:
            base = versiones.pop(0)
            grupo = [base] + [v for v in versiones if tokens_similares(base, v, 0.8)]
            versiones = [v for v in versiones if v not in grupo]
            label = sub[sub['Versión'].isin(grupo)]['Versión'].mode().iloc[0]
            for v in grupo:
                versiones_normalizadas[(modelo, v)] = label

    # Reemplazo de la función interna por lambda en apply
    df_train_proc['Versión'] = df_train_proc.apply(
        lambda row: versiones_normalizadas.get((row['Modelo'], row['Versión']), row['Versión']),
        axis=1
    )

    # Diccionario de versiones válidas por (Marca, Modelo)
    versiones_por_modelo = {}
    for (marca, modelo), sub in df_train_proc.groupby(['Marca', 'Modelo']):
        versiones_por_modelo[(marca, modelo)] = sub['Versión'].dropna().value_counts().index.tolist()

    # ---------- PARTE B: IMPUTACIÓN SOBRE df_to_input ----------
    df_val = df_to_input.copy()
    referencia = df_train_proc[df_train_proc['Versión'].notna()].copy()

    # Paso 1: imputar faltantes por contexto
    for idx, row in df_val[df_val['Versión'].isna()].iterrows():
        marca = str(row.get("Marca", "")).lower()
        modelo = str(row.get("Modelo", "")).lower()

        if marca and modelo:
            posibles = referencia[(referencia["Marca"].str.lower() == marca) &
                                  (referencia["Modelo"].str.lower() == modelo)]["Versión"]
        elif marca:
            posibles = referencia[referencia["Marca"].str.lower() == marca]["Versión"]
        elif modelo:
            posibles = referencia[referencia["Modelo"].str.lower() == modelo]["Versión"]
        else:
            posibles = pd.Series(dtype=str)

        if not posibles.dropna().empty:
            df_val.at[idx, "Versión"] = posibles.mode().iloc[0]

    # Paso 2: corregir valores no válidos
    for idx, row in df_val.iterrows():
        key = (row.get('Marca'), row.get('Modelo'))
        version_actual = row.get('Versión')
        opciones = versiones_por_modelo.get(key, [])
        if opciones and version_actual not in opciones:
            match = next((v for v in opciones if tokens_similares(version_actual, v, 0.75)), None)
            df_val.at[idx, 'Versión'] = match if match else np.nan

    # Paso 3: última imputación por moda del modelo
    for idx, row in df_val[df_val['Versión'].isna()].iterrows():
        modelo = row.get("Modelo")
        if modelo:
            posibles = referencia[referencia["Modelo"] == modelo]["Versión"]
            if not posibles.dropna().empty:
                df_val.at[idx, "Versión"] = posibles.mode().iloc[0]

    return df_val.reset_index(drop=True)

def hmv_combustible(df_train, df_to_input):
    """Imputa 'Tipo de combustible' usando el helper genérico."""
    return imputar_por_contexto(df_train, df_to_input,
                                'Tipo de combustible',
                                ['Marca', 'Modelo', 'Versión', 'Año'])

def hmv_puertas(df_train, df_to_input):
    """Imputa 'Puertas' usando el helper genérico."""
    return imputar_por_contexto(df_train, df_to_input,
                                'Puertas',
                                ['Marca', 'Modelo', 'Versión', 'Año'])

def hmv_transmision(df_train, df_to_input):
    """Imputa 'Transmisión' usando el helper genérico."""
    return imputar_por_contexto(df_train, df_to_input,
                                'Transmisión',
                                ['Marca', 'Modelo', 'Versión', 'Año'])



def hmv_motor(df_train, df_to_input):
    """Imputa 'Motor' usando el helper genérico."""
    return imputar_por_contexto(df_train, df_to_input,
                                'Motor',
                                ['Marca', 'Modelo', 'Versión', 'Año'])

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
    """Imputa 'HP' usando el helper genérico (prioriza 'Motor')."""
    return imputar_por_contexto(df_train, df_to_input,
                                'HP',
                                ['Motor', 'Versión', 'Modelo', 'Año', 'Marca'])



def hmv_traccion(df_train, df_to_input):
    """Imputa 'Tracción' y asigna '4x2' por defecto cuando no hay contexto."""
    return imputar_por_contexto(df_train, df_to_input,
                                'Tracción',
                                ['Marca', 'Modelo', 'Versión', 'Año'],
                                fallback_valor='4x2')


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

    Devuelve:
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

def hmv_color(df_train, df_to_input):

    # ---------- PARTE A: CLUSTERS DESDE df_train ----------
    df_train_proc = df_train.copy()

    # 'morado' -> 'violeta'
    df_train_proc['Color'] = df_train_proc['Color'].apply(
        lambda c: 'violeta' if isinstance(c, str) and 'morado' in normalizar(c, eliminar_espacios=False) else c
    )

    # Construir clusters por primer token
    color_tokens = {}
    for color in df_train_proc['Color'].dropna().unique():
        tok = normalizar(color, eliminar_espacios=False).split()[0]
        key = next((k for k in color_tokens if tokens_similares(k, tok, 0.7)), tok)
        color_tokens.setdefault(key, set()).add(color)

    # Mapeo color original -> etiqueta canon
    color_map = {}
    for canon, variantes in color_tokens.items():
        canon_label = max(list(variantes), key=lambda x: list(variantes).count(x))
        for v in variantes:
            color_map[v] = canon_label
    colores_validos = set(color_map.values())

    # ---------- PARTE B: IMPUTAR / CORREGIR df_to_input ----------
    df_res = df_to_input.copy()

    # Imputar faltantes (Versión > Modelo > Marca)
    for idx, row in df_res[df_res['Color'].isna()].iterrows():
        valor = None
        for col in ['Versión', 'Modelo', 'Marca']:
            key = row.get(col)  
            if pd.notna(key):
                cand = df_train_proc[df_train_proc[col] == key]['Color']
                if not cand.dropna().empty:
                    valor = cand.mode().iloc[0]
                    break
        if valor is not None:
            df_res.at[idx, 'Color'] = valor

    # Canonizar / etiquetar valores
    def _canonizar(c):
        if pd.isna(c):
            return c
        if c in color_map:
            return color_map[c]
        for canon in colores_validos:
            if tokens_similares(c, canon, 0.75):
                return canon
        return 'otro'

    df_res['Color'] = df_res['Color'].apply(_canonizar)

    return df_res




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
        if count_after < count_before:
            print(f"[{func_name}] Muestras eliminadas: {count_before - count_after}")

    total_eliminadas = original_count - len(df)
    print(f"[TOTAL] Muestras eliminadas en total: {total_eliminadas}")

    return df