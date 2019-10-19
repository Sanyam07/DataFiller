"""
    Class that builds a Keras model based on features and target data passed
"""

import tensorflow as tf
import datatypes as dt

# TODO: Remove or abstract dictionaries
TYPE_ACTIVATION_MAP = {
    dt.NUMBER_TYPE: 'sigmoid',
    dt.LABEL_TYPE: 'softmax',
    dt.PHRASE_TYPE: 'softmax'
}

TYPE_LOSS_MAP = {
    dt.NUMBER_TYPE: 'mean_squared_error',
    dt.LABEL_TYPE: 'categorical_crossentropy',
    dt.PHRASE_TYPE: 'categorical_crossentropy'
}

TYPE_METRICS_MAP = {
    dt.NUMBER_TYPE: [],
    dt.LABEL_TYPE: ['accuracy'],
    dt.PHRASE_TYPE: ['accuracy']
}


class ModelFactory:
    """ Factory that creates a Keras model based on feature and target specs """
    def __init__(self, features_dim, target_dim, target_type, depth=1):
        self.features_dim = features_dim
        self.target_dim = target_dim
        self.target_type = target_type
        self.depth = depth
        self.scaling_factor = \
            (self.target_dim / self.features_dim) ** (1 / (self.depth + 1))
        self.neurons = int(self.features_dim * self.scaling_factor)

    def build_model(self, optimizer='adam'):
        """ Build a Keras model based on features and target specs """
        layers = self._build_input_layer()
        layers += self._build_hidden_layers()
        layers += self._build_output_layer()
        model = tf.keras.models.Sequential(layers)
        model.compile(optimizer=optimizer,
                      loss=TYPE_LOSS_MAP[self.target_type],
                      metrics=TYPE_METRICS_MAP[self.target_type])
        return model

    def _build_input_layer(self, activation='relu'):
        return [
            tf.keras.layers.Dense(self.neurons,
                                  activation=activation,
                                  input_dim=self.features_dim),
        ]

    def _build_hidden_layers(self, activation='relu'):
        hidden_layers = []
        hidden_neurons = self.neurons
        for d in range(1, self.depth):
            hidden_neurons = hidden_neurons * self.scaling_factor
            hidden_layers += [
                tf.keras.layers.Dense(hidden_neurons, activation=activation)
            ]
        return hidden_layers

    def _build_output_layer(self):
        return [tf.keras.layers.Dense(
            self.target_dim,activation=TYPE_ACTIVATION_MAP[self.target_type]
        )]


def get_callbacks(min_improvement=10**-2, patience=3):
    return [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=min_improvement,
                                             patience=patience,
                                             restore_best_weights=True)]
