from tensorflow import keras
from tensorflow.keras import layers

LAYERS = keras.Sequential(
    [
        layers.Dense(32, activation="relu", name="layer1"),
        layers.Dense(32, activation="relu", name="layer2"),
    ]
)