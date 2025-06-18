from difflib import get_close_matches
import pandas as pd
from utils.diccionarios import MARCAS_VALIDAS, MODELOS_POR_MARCA
def hmv_marca(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Marca' en df_to_input usando df_train como referencia.
    - Primero intenta imputar por fuzzy matching con 'Modelo' (cutoff alto)
    - Luego intenta con 'Versión' (cutoff alto)
    - Si falla, intenta fuzzy con cutoff más bajo
    - Luego intenta detectar marca en 'Título'
    - Si todo falla, elimina la fila
    Prints:
    - Índices con marca faltante (+2)
    - Coincidencias encontradas y marcas asignadas
    - Índices de filas eliminadas (+2)
    - Resumen final
    """
    df = df_to_input.copy()
    referencia = df_train[df_train['Marca'].notna()].copy()
    total_before = df.shape[0]

    df_missing = df[df['Marca'].isna()].copy()
    print(f"🔍 Índices con marca faltante: {[i + 2 for i in df_missing.index.tolist()]}")

    reemplazos = []
    indices_eliminados = []

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
                    print(f"[{idx + 2}] Fuzzy por MODELO: '{modelo_norm}' ≈ '{match_modelo}' → Marca: {marca_inferida} ({marcas.value_counts().to_dict()})")

        # Paso 2: Fuzzy matching por VERSIÓN (cutoff alto)
        if not marca_inferida and version_norm:
            posibles = referencia[referencia['Versión'].notna()]
            matches = get_close_matches(version_norm, posibles['Versión'].str.lower().tolist(), n=1, cutoff=0.8)
            if matches:
                match_version = matches[0]
                marcas = posibles[posibles['Versión'].str.lower() == match_version]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]
                    print(f"[{idx + 2}] Fuzzy por VERSIÓN: '{version_norm}' ≈ '{match_version}' → Marca: {marca_inferida} ({marcas.value_counts().to_dict()})")

        # Paso 3: Segundo intento con cutoff más bajo en MODELO
        if not marca_inferida and modelo_norm:
            matches = get_close_matches(modelo_norm, referencia['Modelo'].dropna().str.lower().tolist(), n=1, cutoff=0.5)
            if matches:
                match_modelo = matches[0]
                marcas = referencia[referencia['Modelo'].str.lower() == match_modelo]['Marca']
                if not marcas.empty:
                    marca_inferida = marcas.mode().iloc[0]
                    print(f"[{idx + 2}] Segundo intento MODELO (cutoff 0.5): '{modelo_norm}' ≈ '{match_modelo}' → Marca: {marca_inferida}")

        # Paso 4: Buscar en TÍTULO
        if not marca_inferida:
            for m in referencia['Marca'].unique():
                if pd.notna(m) and m.lower() in titulo:
                    marca_inferida = m.lower()
                    print(f"[{idx + 2}] Marca detectada en TÍTULO: '{marca_inferida}'")
                    break

        # Asignación o eliminación
        if marca_inferida:
            df.at[idx, 'Marca'] = marca_inferida
            reemplazos.append(idx)
        else:
            indices_eliminados.append(idx)

    # Eliminar filas sin marca
    df_result = df[df['Marca'].notna()]
    print(f"\n🗑️ Índices eliminados: {[i + 2 for i in indices_eliminados]}")
    print(f"\n✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {len(indices_eliminados)}")

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
    print(f"🔍 Índices con modelo faltante: {[i + 2 for i in df_missing.index.tolist()]}")

    reemplazos = []
    indices_eliminados = []

    for idx, row in df_missing.iterrows():
        modelo_inferido = None
        marca = str(row.get('Marca', '')).lower()
        version = str(row.get('Versión', '')).lower().strip()

        # Caso 1: no hay versión → eliminar
        if not version:
            indices_eliminados.append(idx)
            continue

        # Caso 2: hay marca válida → restringir a modelos de esa marca
        if marca in MODELOS_POR_MARCA:
            modelos_permitidos = MODELOS_POR_MARCA[marca]
            subset = referencia[referencia['Modelo'].str.lower().isin(modelos_permitidos)]
        else:
            # Caso 3: no hay marca o marca desconocida → usar todo el entrenamiento
            subset = referencia

        if subset.empty:
            indices_eliminados.append(idx)
            continue

        # Fuzzy matching con versiones
        versiones_entrenamiento = subset['Versión'].str.lower().tolist()
        matches = get_close_matches(version, versiones_entrenamiento, n=1, cutoff=0.7)

        if matches:
            match_version = matches[0]
            modelos_encontrados = subset[subset['Versión'].str.lower() == match_version]['Modelo']
            if not modelos_encontrados.empty:
                modelo_inferido = modelos_encontrados.mode().iloc[0]
                print(f"[{idx + 2}] Fuzzy por VERSIÓN {'(con marca)' if marca in MODELOS_POR_MARCA else '(sin marca)'}: '{version}' ≈ '{match_version}' → Modelo: {modelo_inferido} ({modelos_encontrados.value_counts().to_dict()})")

        if modelo_inferido:
            df.at[idx, 'Modelo'] = modelo_inferido
            reemplazos.append(idx)
        else:
            indices_eliminados.append(idx)

    df_result = df[df['Modelo'].notna()]
    print(f"\n🗑️ Índices eliminados: {[i + 2 for i in indices_eliminados]}")
    print(f"✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {len(indices_eliminados)}")

    return df_result


def hmv_version(df_train, df_to_input):
    """
    Imputa y agrupa versiones en df_to_input utilizando df_train como referencia.

    Paso 1: Imputación
        - Si hay Modelo → fuzzy matching con versiones de ese modelo en el train
        - Si no hay Modelo pero sí Marca → buscar versiones asociadas a los modelos de esa marca
        - Si no hay Marca ni Modelo → eliminar fila

    Paso 2: Fuzzy clustering por modelo (solo versiones con al menos una palabra en común)
        - Agrupa versiones similares dentro de cada modelo
        - Asigna el nombre más frecuente del grupo
    """

    df = df_to_input.copy()
    referencia = df_train[df_train['Versión'].notna()].copy()

    df_missing = df[df['Versión'].isna()].copy()
    print(f"🔍 Índices con versión faltante: {[i + 2 for i in df_missing.index.tolist()]}")

    reemplazos = []
    indices_eliminados = []

    # Paso 1: Imputación
    for idx, row in df_missing.iterrows():
        modelo = str(row.get("Modelo", "")).lower()
        marca = str(row.get("Marca", "")).lower()
        version_inferida = None

        if modelo:
            versiones_posibles = referencia[referencia["Modelo"].str.lower() == modelo]["Versión"].dropna()
        elif marca in MODELOS_POR_MARCA:
            modelos_de_marca = MODELOS_POR_MARCA[marca]
            versiones_posibles = referencia[referencia["Modelo"].str.lower().isin(modelos_de_marca)]["Versión"].dropna()
        else:
            indices_eliminados.append(idx)
            continue

        if versiones_posibles.empty:
            indices_eliminados.append(idx)
            continue

        version_frecuente = versiones_posibles.mode().iloc[0]
        df.at[idx, "Versión"] = version_frecuente
        reemplazos.append(idx)
        print(f"[{idx + 2}] Imputada versión: '{version_frecuente}' usando contexto de {'modelo' if modelo else 'marca'}")

    df = df[df['Versión'].notna()]  # Eliminar filas sin imputar

    # Paso 2: Clustering con fuzzy + intersección semántica
    versiones_normalizadas = {}

    for modelo, sub in df[['Modelo', 'Versión']].dropna().groupby('Modelo'):
        versiones = sub['Versión'].dropna().unique().tolist()
        mapeo = {}

        print(f"\n🔄 Clustering de versiones para modelo: {modelo} ({len(versiones)} versiones distintas)")

        while versiones:
            base = versiones.pop(0)
            base_tokens = set(base.replace('-', ' ').replace('_', ' ').split())
            similares = []

            for v in versiones:
                v_tokens = set(v.replace('-', ' ').replace('_', ' ').split())
                if base_tokens & v_tokens:  # al menos una palabra en común
                    if get_close_matches(base, [v], n=1, cutoff=0.6):
                        similares.append(v)

            grupo = [base] + similares
            versiones = [v for v in versiones if v not in similares]

            subset = sub[sub['Versión'].isin(grupo)]
            version_mas_comun = subset['Versión'].mode().iloc[0]

            if len(grupo) > 1:
                print(f"👥 Agrupadas: {grupo} → 🏷️ '{version_mas_comun}'")

            for v in grupo:
                mapeo[v] = version_mas_comun

        for original, reemplazo in mapeo.items():
            versiones_normalizadas[(modelo, original)] = reemplazo

    def reemplazar_version(row):
        clave = (row['Modelo'], row['Versión'])
        return versiones_normalizadas.get(clave, row['Versión'])

    df['Versión'] = df.apply(reemplazar_version, axis=1)

    print(f"\n✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {len(indices_eliminados)}")

    return df
