import pandas as pd
import numpy as np
import sys
import dotenv
import os


def open_and_prepare_df(ds='main'):

    dotenv.load_dotenv()
    main_dir = os.getenv('MAINDIR')

    path = f'{main_dir}datasets/'

    if ds == 'main':
        df = pd.read_excel(io=f'{path}NLP_CLEAN.xlsx')
        print(len(df))
        df = df[df['time'] > 300]
        print(len(df))
        df = df[df['label'] != 1]
        print(len(df))
        for item in [f'nlp_{i}' for i in range(2, 6)]:
            df = df[df[item].apply(lambda x: x.count(' ') > 1)]
        print(len(df))
        
    elif ds == 'features':
        df = pd.read_excel(io=f'{path}NLP_FEATURES.xlsx')

    return df

def X_y_split(df, X_string):
    
    X = []
    X_raw = df[X_string].values

    for item in X_raw:
        if X_string == 'nlp_all':
            X.append(item)
        else:
            X.append(eval(item))

    X = np.asarray(X)
    y = df['label'].values

    return X, y