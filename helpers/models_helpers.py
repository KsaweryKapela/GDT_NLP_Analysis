import torch
import random
import tensorflow as tf
import numpy as np

def set_device():
    print(f'Cuda available: {torch.cuda.is_available()}')
    return "cuda:0" if torch.cuda.is_available() else "cpu"
    
def random_seeds():
    np.random.seed(123)
    random.seed(123)
    tf.random.set_seed(1234)