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
    df['Marca_original'] = df['Marca']

    for idx, row in df.iterrows():
        marca = row['Marca']
        titulo = str(row['Titulo']).lower()
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
