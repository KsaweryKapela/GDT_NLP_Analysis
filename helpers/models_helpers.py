import torch
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_device():
    print(f'Cuda available: {torch.cuda.is_available()}')
    return "cuda:0" if torch.cuda.is_available() else "cpu"
    
def random_seeds():
    np.random.seed(123)
    random.seed(123)
    tf.random.set_seed(1234)

def plot_results(prediction, true_values):

    prediction = [item if item >= 4 else 4 for item in prediction]
    prediction = [item if item <= 20 else 20 for item in prediction]

    answers_tuples = [(y, res) for y, res in zip(true_values, prediction)]
    sorted_tuples = sorted(answers_tuples, key=lambda x: x[0])

    sorted_results = [x[1] for x in sorted_tuples]

    true_values = [x[0] for x in sorted_tuples]
    prediction = [round(int(item)) for item in sorted_results]

    plt.figure(figsize=(10, 7))
    plt.plot(range(len(prediction)), prediction, 'o', color=[1, 0, 0, 0.7], label="Prediction")
    plt.plot(range(len(true_values)), true_values, 'o', color=[0, 1, 0, 0.5], label='Observable data')
    plt.legend(loc='upper left')
    plt.title('Connected predictions vs real data')
    plt.show()