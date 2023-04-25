import sys
import dotenv
import os
dotenv.load_dotenv()
sys.path.append(os.getenv('MAINDIR'))
from helpers.ds_helpers import X_y_split, open_and_prepare_df
from helpers.models_helpers import random_seeds
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings


def get_kfold_results(model_class, X, y, params):

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=8)

    true_y = []
    preds = []

    for train_index, test_index in kfold.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_class(neurons=params['neurons'], activation_1=params['activation_1'], activation_2=params['activation_2'],
                            activation_3=params['activation_3'], learning_rate=params['learning_rate'], optimizer=params['optimizer'])
        
        model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
        pred = model.predict(X_test, verbose=0).flatten()
        preds = np.concatenate((preds, pred))

        true_y = np.concatenate((true_y, y_test))

    corr = round(stats.pearsonr(preds, true_y)[0], 3)
    mae = round(mean_absolute_error(preds, true_y), 3)
    print(f'Corr = {corr}, MAE = {mae}')

    return corr, mae

def create_model(neurons, activation_1, activation_2, activation_3, learning_rate, optimizer):
    random_seeds()
    
    keras_model = tf.keras.Sequential([
                        tf.keras.layers.Dense(neurons, activation=activation_1),
                        tf.keras.layers.Dense(neurons/2, activation=activation_2),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(neurons/3, activation=activation_3),
                        tf.keras.layers.Dense(units=1)
                        ])

    keras_model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mean_absolute_error')
    return keras_model


def run_grid(create_model, X, y, activations_1, activations_2, activations_3, learning_rates, neurons_count, epochs, batch_sizes, optimizers):
    best_corr = 0
    best_params = None
    loop = 1
    loops_count = len(activations_1) * len(activations_2) * len(activations_3) * len(learning_rates) *\
                  len(neurons_count) * len(epochs) * len(batch_sizes) * len(optimizers)

    for epoch in epochs:
        for batch_size in batch_sizes:
            for activation_1 in activations_1:
                for activation_2 in activations_2:
                    for activation_3 in activations_3:
                        for lr in learning_rates:
                            for neurons in neurons_count:
                                for optimizer in optimizers:
                                
                                    params = {}
                                    params['epochs'] = epoch
                                    params['batch_size'] = batch_size
                                    params['activation_1'] = activation_1
                                    params['activation_2'] = activation_2
                                    params['activation_3'] = activation_3
                                    params['learning_rate'] = lr
                                    params['neurons'] = neurons
                                    params['optimizer'] = optimizer

                                    corr, mae = get_kfold_results(create_model, X, y, params)
                                    print(params)
                                    if corr > best_corr:
                                        best_corr = corr
                                        best_params = params

                                    print(f'{loop}/{loops_count}')
                                    loop += 1
    return best_params, best_corr


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    X, y = X_y_split(open_and_prepare_df('features'), 'nlp_all')

    activations_1 = ['relu', 'tanh', 'sigmoid']
    activations_2 = ['relu', 'tanh', 'sigmoid']
    activations_3 = ['relu']
    learning_rates = [0.001, 0.005, 0.01]
    neurons_count = [256, 512]
    epochs = [100, 150, 200]
    batch_sizes = [30, 60]
    optimizers = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam]

    best_params, best_corr = run_grid(create_model, X, y, activations_1, activations_2, activations_3, learning_rates, neurons_count, epochs, batch_sizes, optimizers)