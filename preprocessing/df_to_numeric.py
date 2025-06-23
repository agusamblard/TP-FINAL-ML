import pandas as pd

def df_to_numeric(df):
    """
    Transforma un DataFrame con columnas categóricas a formato numérico y aplica renombramientos específicos.

    - Convierte a enteros las columnas: 'Marca', 'Modelo', 'Versión', 'Color', 'Tipo de combustible', 'Tipo de vendedor'
    - Guarda los mapeos usados en 'conversiones.txt'
    - Renombra 'Con cámara de retroceso' a 'tiene_camara'
    - Mapea 'Tracción': '4x4' → 1, '4x2' → 0 → 'is_4x4'
    - Mapea 'Transmisión': 'manual' → 1, 'automatica' → 0 → 'is_manual'
    - Reporta columnas no numéricas restantes
    """
    cols_a_mapear = ['Versión','Marca', 'Modelo', 'Color', 'Tipo de combustible', 'Tipo de vendedor']
    mapeos = {}

    for col in cols_a_mapear:
        if col == 'Versión':
            versiones = df[['Marca', 'Modelo', 'Versión']].dropna().drop_duplicates()
            claves = [(row['Marca'], row['Modelo'], row['Versión']) for _, row in versiones.iterrows()]
            mapeo = {v: i for i, v in enumerate(sorted(claves))}
            df['Versión'] = df.apply(lambda row: mapeo.get((row['Marca'], row['Modelo'], row['Versión'])), axis=1)
            mapeos['Versión'] = mapeo
        else:
            valores = df[col].astype(str).unique()
            mapeo = {v: i for i, v in enumerate(sorted(valores))}
            df[col] = df[col].map(mapeo)
            mapeos[col] = mapeo

    with open('conversiones.txt', 'w', encoding='utf-8') as f:
        for col, mapeo in mapeos.items():
            f.write(f'Columna: {col}\n')
            if col == 'Versión':
                for (marca, modelo, version), v in mapeo.items():
                    f.write(f'  Marca: {marca} | Modelo: {modelo} | Versión: {version} -> {v}\n')
            else:
                for k, v in mapeo.items():
                    f.write(f'  {k} -> {v}\n')
            f.write('\n')

    if 'Con cámara de retroceso' in df.columns:
        df.rename(columns={'Con cámara de retroceso': 'tiene_camara'}, inplace=True)

    if 'Tracción' in df.columns:
        df['is_4x4'] = df['Tracción'].map({'4x4': 1, '4x2': 0})
        df.drop(columns=['Tracción'], inplace=True)

    if 'Transmisión' in df.columns:
        df['is_manual'] = df['Transmisión'].map({'manual': 1, 'automatica': 0})
        df.drop(columns=['Transmisión'], inplace=True)

    if 'Puertas' in df.columns:
        df['is_coupe'] = df['Puertas'].map({3: 1, 5: 0})
        df.drop(columns=['Puertas'], inplace=True)

    cols_no_numericas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cols_no_numericas:
        print("Columnas no numéricas restantes:", cols_no_numericas)
    else:
        print("No quedan columnas no numéricas.")

    return df