from difflib import get_close_matches
import pandas as pd
from utils.diccionarios import MARCAS_VALIDAS, MODELOS_POR_MARCA
from preprocessing.data_cleanse import normalizar, quitar_tildes
import re
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
            df.drop(index=idx, inplace=True)

    df_result = df[df['Marca'].notna()]
    print(f"\n✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {total_before - df_result.shape[0]}")
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
                print(f"[{idx + 2}] Fuzzy por VERSIÓN {'(con marca)' if marca in MODELOS_POR_MARCA else '(sin marca)'}: '{version}' ≈ '{match_version}' → Modelo: {modelo_inferido} ({modelos_encontrados.value_counts().to_dict()})")

        if modelo_inferido:
            df.at[idx, 'Modelo'] = modelo_inferido
            reemplazos.append(idx)
        else:
            df.drop(index=idx, inplace=True)

    df_result = df[df['Modelo'].notna()]
    print(f"\n✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {total_before - df_result.shape[0]}")
    return df_result




def hmv_version(df_train, df_to_input):
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
    print(f"🔍 Índices con versión faltante: {[i + 2 for i in df_missing.index.tolist()]}")

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
        print(f"[{idx + 2}] Imputada versión: '{version_frecuente}' usando contexto de {'modelo' if modelo else 'marca'}")

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

        print(f"\n🔄 Clustering de versiones para modelo: {modelo} ({len(versiones)} versiones distintas)")

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

            if len(grupo) > 1:
                print(f"👥 Agrupadas: {grupo} → 🏷️ '{version_mas_comun}'")

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
    print("\n🧩 Paso final: Agrupamiento entre versiones ya normalizadas")

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

        print(f"\n🔍 Agrupamiento final para modelo: {modelo} ({len(versiones_finales)} versiones)")

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

            if len(grupo) > 1:
                print(f"🔗 Agrupadas finales por token: {grupo} → 🏷️ '{version_final}'")

            for v in grupo:
                finales_map[(modelo, v)] = version_final

        # Aplicar mapeo final
        def ajustar_version_final(row):
            clave = (row['Modelo'], row['Versión'])
            return finales_map.get(clave, row['Versión'])

        df['Versión'] = df.apply(ajustar_version_final, axis=1)


    print(f"\n✔️ Muestras imputadas: {len(reemplazos)}")
    print(f"🗑️ Muestras eliminadas por no poder imputar: {df_missing.shape[0] - len(reemplazos)}")
    return df


def hmv_combustible(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Tipo de combustible' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles (Marca, Modelo, Versión, Año).
    - Si no encuentra coincidencias, reduce las columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    - Si logra imputar, agrega la muestra al set de referencia.
    - Si no logra imputar, elimina la fila y la guarda en 'combustible_deleted.csv'.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['Tipo de combustible'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Tipo de combustible'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin Modelo o Versión.")
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

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['Tipo de combustible'].mode()
                    if not moda.empty:
                        df.at[idx, 'Tipo de combustible'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            print(f"   ❌ No se pudo imputar, se eliminará.")
            df.drop(index=idx, inplace=True)

    print(f"\n✔️ Imputaciones realizadas: {imputados}")
    print(f"🗑️ Muestras eliminadas: {df_to_input.shape[0] - df.shape[0]} (guardadas en 'combustible_deleted.csv')")
    return df

def hmv_puertas(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Puertas' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles (Marca, Modelo, Versión, Año).
    - Si no encuentra coincidencias, reduce las columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    - Si logra imputar, agrega la muestra al set de referencia.
    - Si no logra imputar, elimina la fila y la guarda en 'puertas_deleted.csv'.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['Puertas'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Puertas'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin Modelo o Versión.")
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

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['Puertas'].mode()
                    if not moda.empty:
                        df.at[idx, 'Puertas'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            print(f"   ❌ No se pudo imputar, se eliminará.")
            df.drop(index=idx, inplace=True)

    print(f"\n✔️ Imputaciones realizadas: {imputados}")
    print(f"🗑️ Muestras eliminadas: {df_to_input.shape[0] - df.shape[0]} (guardadas en 'puertas_deleted.csv')")
    return df

def hmv_transmision(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Transmisión' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para poder imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles.
    - Si no encuentra coincidencias, reduce el conjunto de columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    - Si logra imputar, agrega la muestra al set de referencia.
    - Si no logra imputar, elimina la fila y la guarda en 'transmision_deleted.csv'.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['Transmisión'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Transmisión'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin Modelo o Versión.")
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

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['Transmisión'].mode()
                    if not moda.empty:
                        df.at[idx, 'Transmisión'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            print(f"   ❌ No se pudo imputar, se eliminará.")
            df.drop(index=idx, inplace=True)

    print(f"\n✔️ Imputaciones realizadas: {imputados}")
    print(f"🗑️ Muestras eliminadas: {df_to_input.shape[0] - df.shape[0]} (guardadas en 'transmision_deleted.csv')")
    return df



def hmv_motor(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Motor' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión' para poder imputar.
    - Busca coincidencias exactas con las columnas no nulas disponibles.
    - Si no encuentra coincidencias, reduce el conjunto de columnas (manteniendo Modelo/Versión) hasta encontrar alguna.
    - Si logra imputar, la muestra se agrega al set de referencia para futuras imputaciones.
    - Si no logra imputar, elimina la fila y la guarda en 'motor_deleted.csv'.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['Motor'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Motor'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin Modelo o Versión.")
            df.drop(index=idx, inplace=True)
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(col in subset for col in ['Modelo', 'Versión']):
                    continue  # Requiere al menos Modelo o Versión

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['Motor'].mode()
                    if not moda.empty:
                        df.at[idx, 'Motor'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            print(f"   ❌ No se pudo imputar, se eliminará.")
            df.drop(index=idx, inplace=True)

    print(f"\n✔️ Imputaciones realizadas: {imputados}")
    print(f"🗑️ Muestras eliminadas: {df_to_input.shape[0] - df.shape[0]} (guardadas en 'motor_deleted.csv')")
    return df

def hmv_camara(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Con cámara de retroceso' de df_to_input usando df_train como referencia.

    Lógica:
    - Solo intenta imputar si están presentes Marca, Modelo, Versión y Año.
    - Busca coincidencias exactas con esas cuatro columnas en df_train o imputaciones anteriores.
    - Si hay coincidencias, asigna la moda.
    - Si no hay coincidencias, asigna 0.
    - Imputaciones exitosas se agregan al set de referencia paso a paso.
    """
    import pandas as pd

    df = df_to_input.copy()
    referencia = df_train[df_train['Con cámara de retroceso'].notna()].copy()
    imputados = 0
    default_0 = 0

    for idx, row in df[df['Con cámara de retroceso'].isna()].iterrows():
        if any(pd.isna(row[col]) for col in ['Marca', 'Modelo', 'Versión', 'Año']):
            print(f"[{idx + 2}] ❌ Faltan columnas clave. Se asigna 0.")
            df.at[idx, 'Con cámara de retroceso'] = 0
            default_0 += 1
            continue

        filtro = referencia[
            (referencia['Marca'] == row['Marca']) &
            (referencia['Modelo'] == row['Modelo']) &
            (referencia['Versión'] == row['Versión']) &
            (referencia['Año'] == row['Año'])
        ]

        print(f"[{idx + 2}] 🔍 Buscando coincidencias exactas en Marca, Modelo, Versión y Año")
        print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

        if not filtro.empty:
            moda = filtro['Con cámara de retroceso'].mode()
            if not moda.empty:
                valor = moda.iloc[0]
                df.at[idx, 'Con cámara de retroceso'] = valor
                print(f"   ✅ Imputado con valor: {valor}")
                imputados += 1
            else:
                df.at[idx, 'Con cámara de retroceso'] = 0
                print(f"   ⚠️ Sin moda. Se asigna 0.")
                default_0 += 1
        else:
            df.at[idx, 'Con cámara de retroceso'] = 0
            print(f"   ❌ Sin coincidencias. Se asigna 0.")
            default_0 += 1

    print(f"\n✔️ Imputaciones realizadas por coincidencia exacta: {imputados}")
    print(f"🔧 Muestras sin coincidencias (asignadas 0): {default_0}")

    return df


def hmv_hp(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'HP' de df_to_input usando df_train como referencia.

    Lógica:
    - Busca combinaciones exactas con columnas: Motor, Versión, Modelo, Año, Marca (en ese orden de prioridad).
    - Evalúa combinaciones desde 5 hasta 1 columna, priorizando las que incluyan 'Motor'.
    - Requiere al menos uno entre 'Modelo', 'Versión' o 'Motor'.
    - Imputa la moda si encuentra coincidencias.
    - Si no encuentra coincidencias, elimina la muestra y la guarda en 'hp_deleted.csv'.
    - Las imputaciones exitosas se agregan a la base de referencia en cada paso.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['HP'].notna()].copy()
    imputados = 0

    columnas_importancia = ['Motor', 'Versión', 'Modelo', 'Año', 'Marca']

    for idx, row in df[df['HP'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión', 'Motor']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin al menos Modelo, Versión o Motor.")
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

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['HP'].mode()
                    if not moda.empty:
                        df.at[idx, 'HP'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            print(f"   ❌ No se pudo imputar. Se eliminará la muestra.")
            df.drop(index=idx, inplace=True)

    print(f"\n✔️ Imputaciones realizadas: {imputados}")
    print(f"🗑️ Muestras eliminadas: {df_to_input.shape[0] - df.shape[0]} ")
    return df



def hmv_traccion(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Tracción' de df_to_input usando df_train como referencia.

    Lógica:
    - Requiere al menos 'Modelo' o 'Versión'.
    - Busca coincidencias exactas con combinaciones de columnas disponibles: Marca, Modelo, Versión, Año.
    - Evalúa combinaciones desde 4 hasta 1 columna.
    - Si encuentra coincidencias, imputa la moda y actualiza el set de referencia.
    - Si no encuentra coincidencias, asigna "4x2" como valor por defecto.
    """
    import pandas as pd
    from itertools import combinations

    df = df_to_input.copy()
    referencia = df_train[df_train['Tracción'].notna()].copy()
    imputados = 0
    asignados_default = 0

    columnas_importancia = ['Marca', 'Modelo', 'Versión', 'Año']

    for idx, row in df[df['Tracción'].isna()].iterrows():
        disponibles = [col for col in columnas_importancia if pd.notna(row.get(col))]

        if not disponibles or not any(col in disponibles for col in ['Modelo', 'Versión']):
            print(f"[{idx + 2}] ❌ No se puede imputar sin Modelo o Versión. Se asigna '4x2'.")
            df.at[idx, 'Tracción'] = '4x2'
            asignados_default += 1
            continue

        imputado = False

        for k in range(len(disponibles), 0, -1):
            for subset in combinations(disponibles, k):
                if not any(c in subset for c in ['Modelo', 'Versión']):
                    continue  # Requiere al menos Modelo o Versión

                filtro = referencia.copy()
                for col in subset:
                    filtro = filtro[filtro[col] == row[col]]

                print(f"[{idx + 2}] 🔍 Buscando con columnas: {', '.join(subset)}")
                print(f"   ↪️ Coincidencias encontradas: {len(filtro)}")

                if not filtro.empty:
                    moda = filtro['Tracción'].mode()
                    if not moda.empty:
                        df.at[idx, 'Tracción'] = moda.iloc[0]
                        print(f"   ✅ Imputado con valor: {moda.iloc[0]}")
                        imputados += 1
                        imputado = True
                        break
            if imputado:
                break

        if not imputado:
            df.at[idx, 'Tracción'] = '4x2'
            print(f"   ❌ No se pudo imputar. Se asigna valor por defecto: '4x2'.")
            asignados_default += 1

    print(f"\n✔️ Imputaciones realizadas por coincidencia: {imputados}")
    print(f"🔧 Asignaciones por defecto ('4x2'): {asignados_default}")

    return df

def hmv_year(df_train, df_to_input):
    """
    Imputa valores faltantes en la columna 'Año' de df_to_input usando regresión lineal
    entrenada con muestras de df_train que coincidan en Marca, Modelo y Versión.

    Requiere que Kilómetros y Precio estén presentes.

    - Si hay al menos 5 coincidencias con año conocido, se entrena una regresión.
    - Si no hay suficientes coincidencias o faltan predictores, se mantiene NaN.

    Retorna:
        pd.DataFrame con los valores imputados en la columna 'Año'.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df = df_to_input.copy()
    referencia = df_train[df_train['Año'].notna()].copy()
    imputados = 0

    for idx, row in df[df['Año'].isna()].iterrows():
        print(f"\n[{idx + 2}] 🔍 Intentando imputar Año para muestra:")

        if pd.isna(row['Kilómetros']) or pd.isna(row['Precio']):
            print(f"   ⚠️ Faltan datos clave (Kilómetros o Precio). Se omite.")
            continue

        print(f"   Marca: {row['Marca']}, Modelo: {row['Modelo']}, Versión: {row['Versión']}")
        print(f"   Kilómetros: {row['Kilómetros']}, Precio: {row['Precio']}")

        condiciones = (
            (referencia['Marca'] == row['Marca']) &
            (referencia['Modelo'] == row['Modelo']) &
            (referencia['Versión'] == row['Versión']) &
            referencia['Kilómetros'].notna() &
            referencia['Precio'].notna()
        )

        candidatos = referencia[condiciones]

        print(f"   ↪️ Coincidencias para regresión: {len(candidatos)}")

        if len(candidatos) < 5:
            print("   ❌ No hay suficientes datos para entrenar regresión.")
            continue

        X = candidatos[['Kilómetros', 'Precio']]
        y = candidatos['Año']

        modelo = LinearRegression()
        modelo.fit(X, y)

        pred = modelo.predict([[row['Kilómetros'], row['Precio']]])[0]
        pred_redondeado = int(round(pred))

        print(f"   ✅ Predicción cruda: {pred:.2f} → Imputado como: {pred_redondeado}")

        df.at[idx, 'Año'] = pred_redondeado
        imputados += 1

    print(f"\n✔️ Años imputados por regresión: {imputados}")
    return df

def hmv_km(df_train, df_to_input):
    '''
    HAY QUE IMPLEMENTARLO. QUIEOR ANALIZAR CORRELACION PARA VER SI SE PUEDE HACER UNA REGRESION
    
    '''
    return df_to_input

def hmv_precio(df_train, df_to_input):
    '''
    HAY QUE IMPLEMENTARLO. QUIEOR ANALIZAR CORRELACION PARA VER SI SE PUEDE HACER UNA REGRESION
    
    '''
    return df_to_input

def hmv_tipo_de_vendedor(df_train,df_to_input):
    '''
    HAY QUE IMPLEMENTARLO. QUIEOR ANALIZAR CORRELACION PARA VER SI SE PUEDE HACER UNA REGRESION

    '''
    return df_to_input


def hmv_color(df_train, df_to_input):
    """
    Agrupa valores similares en la columna 'Color' basándose en tokens con similitud textual.
    Luego imputa NaNs utilizando este orden de prioridad:
    1. Moda de 'Color' para esa Versión
    2. Moda de 'Color' para ese Modelo
    3. Moda de 'Color' para esa Marca
    """

    print("🎨 Paso 0: Reemplazando 'morado' por 'violeta' en ambos datasets...\n")
    for df in [df_train, df_to_input]:
        df['Color'] = df['Color'].apply(lambda c: 'violeta' if isinstance(c, str) and 'morado' in normalizar(c, eliminar_espacios=False) else c)

    print("🔍 Paso 1: Agrupando colores por tokens similares...\n")
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

    print(f"📦 Grupos de colores formados: {len(color_groups)}")
    for grupo, variantes in color_groups.items():
        print(f"🔗 Token base: '{grupo}' → {sorted(variantes)}")

    print("\n🧼 Paso 2: Reemplazando valores conocidos en df_to_input...\n")
    color_map = {}
    for grupo, variantes in color_groups.items():
        for variante in variantes:
            color_map[variante] = grupo

    df_result = df_to_input.copy()
    df_result['Color'] = df_result['Color'].map(lambda x: color_map.get(x, x))

    print("🩹 Paso 3: Imputando valores faltantes con prioridad Versión → Modelo → Marca\n")
    imputados = 0
    for idx, row in df_result[df_result['Color'].isna()].iterrows():
        version = row.get('Versión')
        modelo = row.get('Modelo')
        marca = row.get('Marca')

        valor_color = None

        # Intento 1: por versión
        if pd.notna(version):
            coincidencias = df_result[df_result['Versión'] == version]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'versión'

        # Intento 2: por modelo
        if valor_color is None and pd.notna(modelo):
            coincidencias = df_result[df_result['Modelo'] == modelo]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'modelo'

        # Intento 3: por marca
        if valor_color is None and pd.notna(marca):
            coincidencias = df_result[df_result['Marca'] == marca]['Color'].dropna()
            if not coincidencias.empty:
                valor_color = coincidencias.mode().iloc[0]
                origen = 'marca'

        if valor_color is not None:
            df_result.at[idx, 'Color'] = valor_color
            imputados += 1
            print(f"[{idx + 2}] 🖌️ Imputado color '{valor_color}' por {origen}")
        else:
            print(f"[{idx + 2}] ⚠️ No se pudo imputar. Se mantiene como NaN")

    print(f"\n✅ Total de colores imputados por contexto: {imputados}")
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
        ("hmv_km", hmv_km),  # aún sin implementar
        ("hmv_precio", hmv_precio),  # aún sin implementar
        ("hmv_tipo_de_vendedor", hmv_tipo_de_vendedor),  # aún sin implementar
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