import numpy as np
from singleton_decorator import singleton
from einops.layers.tensorflow import Rearrange

from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    Conv1D,
    Dense,
    InputLayer,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2, L2


@singleton
class PoseCommand:
    def __init__(self, model_path="Gesto_Command_Weight.h5"):
        self.thresh = 0.98
        self.no_seqs = 50  # No. of Videos per Action
        self.seqs_length = 30  # No. if Frames in Videos

        self.actions = np.array(["Play Music", "Hello", "Alarm 2"])

        self.model = Sequential(
            [
                InputLayer(input_shape=(self.seqs_length, 192, 1)),
                TimeDistributed(
                    Conv1D(
                        1,
                        1,
                        activation="relu",
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Rearrange("b s d t -> b s (d t)"),
                Bidirectional(
                    GRU(
                        64,
                        return_sequences=True,
                        activation="relu",
                        input_shape=(30, 128),
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Bidirectional(
                    GRU(
                        128,
                        return_sequences=True,
                        activation="relu",
                        input_shape=(30, 128),
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Bidirectional(
                    GRU(
                        64,
                        return_sequences=False,
                        activation="relu",
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    self.actions.shape[0],
                    activation="softmax",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
            ]
        )

        self.load_model(f'.models/{model_path}')

    def warm_up(self):
        print("Warming Up Model")
        self.model.predict(np.zeros((1, 30, 192, 1)))
        print("Model Warmed up !!")

    def load_model(self, path):
        self.warm_up()
        self.model.load_weights(path)
        print("Pretrained weights loaded !!")
