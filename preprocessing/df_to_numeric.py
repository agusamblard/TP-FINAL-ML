import pandas as pd

def df_to_numeric(df_train, df_to_input):
    """
    Transforma df_to_input a numérico usando mapeos aprendidos de df_train.

    - Convierte columnas categóricas a números según df_train.
    - Valores no vistos en df_to_input se reemplazan por NaN.
    - Guarda los mapeos en 'conversiones.txt'.
    - Aplica transformaciones adicionales:
        - 'Con cámara de retroceso' → 'tiene_camara'
        - 'Tracción' → 'is_4x4'
        - 'Transmisión' → 'is_manual'
        - 'Puertas' (3 → coupé, 5 → no) → 'is_coupe'
    - Retorna df_to_input transformado.
    """
    cols_a_mapear = ['Versión','Marca', 'Modelo', 'Color', 'Tipo de combustible', 'Tipo de vendedor']
    mapeos = {}
    df_train = df_train.copy()
    df_to_input = df_to_input.copy()

    for col in cols_a_mapear:
        if col == 'Versión':
            versiones = df_train[['Marca', 'Modelo', 'Versión']].dropna().drop_duplicates()
            claves = [(row['Marca'], row['Modelo'], row['Versión']) for _, row in versiones.iterrows()]
            mapeo = {v: i for i, v in enumerate(sorted(claves))}
            df_train['Versión'] = df_train.apply(lambda row: mapeo.get((row['Marca'], row['Modelo'], row['Versión'])), axis=1)
            df_to_input['Versión'] = df_to_input.apply(lambda row: mapeo.get((row['Marca'], row['Modelo'], row['Versión'])), axis=1)
            mapeos['Versión'] = mapeo
        else:
            valores = df_train[col].astype(str).unique()
            mapeo = {v: i for i, v in enumerate(sorted(valores))}
            df_train[col] = df_train[col].astype(str).map(mapeo)
            df_to_input[col] = df_to_input[col].astype(str).map(mapeo)
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

    def transformar(df):
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

        return df

    df_to_input = transformar(df_to_input)

    # Reporte final
    cols_no_numericas = df_to_input.select_dtypes(include=['object', 'category']).columns.tolist()
    if cols_no_numericas:
        print("df_to_input - Columnas no numéricas restantes:", cols_no_numericas)
    else:
        print("df_to_input - No quedan columnas no numéricas.")

    return df_to_input