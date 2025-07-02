
import pandas as pd
from preprocessing.data_cleanse import limpiar_dataset
from preprocessing.hmv import hmv_dataset
from preprocessing.df_to_numeric import df_to_numeric


def preprocessing_full(df_to_input,train_complete=pd.read_csv('data/processed/train_processed.csv'), precio=False):
    df_to_input = limpiar_dataset(df_to_input, precio=precio)
    df_to_input = hmv_dataset(train_complete,df_to_input)
    df_to_input = df_to_numeric(train_complete, df_to_input)

    return df_to_input