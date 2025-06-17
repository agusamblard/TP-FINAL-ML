import pandas as pd

def handle_missing_values( df):
    df = hmv_marca(df)
    df = hmv_modelo(df)
    df = hmv_ano(df)
    df = hmv_km(df)

    