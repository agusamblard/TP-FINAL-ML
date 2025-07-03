
import pandas as pd
from preprocessing.data_cleanse import limpiar_dataset
from preprocessing.hmv import hmv_dataset
from preprocessing.df_to_numeric import df_to_numeric
from preprocessing.hmv_final_test import hmv_dataset_final


def preprocessing_full(df_to_input,train_complete=pd.read_csv('data/processed/train_processed.csv'), precio=False, final_test=False):
    df_to_input = limpiar_dataset(df_to_input, precio=precio)
    if not(final_test):
        df_to_input = hmv_dataset(train_complete,df_to_input)
    else:
        df_to_input = hmv_dataset_final(train_complete, df_to_input)
    df_to_input = df_to_numeric(train_complete, df_to_input)
    
    for col in df_to_input.select_dtypes(include=['float', 'int']).columns:
        if df_to_input[col].isna().sum() > 0:
            mean_value = df_to_input[col].mean()
            df_to_input[col].fillna(mean_value, inplace=True)

    return df_to_input