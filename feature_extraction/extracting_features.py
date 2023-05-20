import sys
import dotenv
import os
dotenv.load_dotenv()
sys.path.append(os.getenv('MAINDIR'))
from helpers.ds_helpers import open_and_prepare_df
from helpers.models_helpers import set_device
import numpy as np
import pandas as pd
from feature_extraction.transformer_class import BertTransformer
from sklearn import decomposition
from transformers import AutoTokenizer, AutoModel


def initialize_herBERT_transformer(transformer, device, max_length=60):
    tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')
    bert_model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    bert_model = bert_model.to(device)
    bert_transformer = transformer(tokenizer, bert_model, max_length=max_length)
    return bert_transformer


def transform_and_stack(df, transformer, arguments):
    tokenized_X_list = []

    for item in arguments:

        item_processed = transformer.transform(list(df[item]))
        tokenized_X_list.append(item_processed.cpu())

    tokized_X_tuple = tuple(tokenized_X_list)

    X = np.hstack(tokized_X_tuple)
    return X

def perform_PCA(X, n_comp=300):
    pca = decomposition.PCA(n_components=n_comp)
    X_PCA = pca.fit_transform(X)
    return X_PCA

if __name__ == '__main__':

    DATASET = 'main'
    NAME = 'NLP_FEATURES'

    device = set_device()
    df = open_and_prepare_df(DATASET)

    bert_transformer = initialize_herBERT_transformer(BertTransformer, device)
    X_strings = [('nlp_2',), 
                 ('nlp_3',),
                 ('nlp_4',),
                 ('nlp_5',)]
    
    features_df = pd.DataFrame()
    
    for arg in X_strings:
        X = transform_and_stack(df, bert_transformer, arg)

        features_df[f'{arg[0]}{len(arg)}'] = X.tolist()
        print(f'{arg} added')

    features_df['label'] = df['GDT_score'].values
    features_df.columns = ['nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'label']
    features_df.to_excel(f'datasets/{NAME}.xlsx')