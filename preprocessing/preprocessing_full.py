
import pandas as pd
from preprocessing.data_cleanse import limpiar_dataset
from preprocessing.hmv import hmv_dataset
from preprocessing.df_to_numeric import df_to_numeric

def preprocessing_full(df_train, df_to_input, cleaned = True):
    df_to_input = limpiar_dataset(df_to_input)
    df_to_input = hmv_dataset(df_train,df_to_input)
    df_to_input = df_to_numeric(df_train, df_to_input)

    return df_to_input