import nn_npy as nn
import os

model = nn.network()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist")
model.load_model(MODEL_PATH)

model.summary()