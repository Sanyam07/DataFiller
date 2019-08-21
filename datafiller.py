"""
    Class that completes and validates a dataset field.
    It takes an input Dataframe with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.6
"""

import math
import copy
import scipy
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

import datatype as df

TARGET_SUFFIX = '_NEW'


class DataFiller:
    """
    Complete and validate a target column in a dataset through machine learning
    """

    def __init__(self, input_data, target, sparse_matrix=False):
        """
        Class constructor, initialized with Dataset to be filled

        :param input_data: Dataframe with data to be used
        :param target: String with column to be evaluated and completed
        :param sparse_matrix: Boolean to limit memory usage via sparse matrices
        (for large datasets)
        """
        self.target = target
        self.input_data = input_data
        self.features_data = input_data.drop(target, axis=1)
        self.target_data = input_data[target]
        self.sparse = sparse_matrix
        self.data_types = df.classify_fields(self.input_data)
        self.features_encoder = self.build_features_encoder()
        self.target_type = self.get_field_type(self.target)
        self.target_encoder = self.build_target_encoder()
        self.model = None
        self.target_pred = None

    def get_field_type(self, field):
        for _, type in self.data_types.items():
            if field in type['fields']:
                return type['datatype']

    def build_features_encoder(self):
        """Prepare multi-feature processing pipeline"""
        transformers = []
        for name, type in self.data_types.items():
            if name is not 'ignore':
                features = [f for f in type['fields'] if f != self.target]
                transformers += [(name,
                                  type['datatype'].get_feature_transformer(),
                                  features)]
        return ColumnTransformer(transformers)

    def build_target_encoder(self):
        """Prepare the target encoder based on its type"""
        return make_pipeline(self.target_type.get_target_transformer())

    def build_model(self, target_type, features_dim, target_dim, depth=1):
        """Prepare the training model"""
        model = Sequential()
        input_dim = features_dim
        neurons_scaling_factor = (target_dim/features_dim)**(1/(depth+1))
        for d in range(0, depth):
            neurons = int(input_dim * neurons_scaling_factor)
            model.add(Dense(neurons, activation='relu', input_dim=input_dim))
            input_dim = neurons
        model.add(Dense(target_dim, activation=target_type.activation))
        model.compile(optimizer='adam',
                      loss=target_type.loss,
                      metrics=target_type.metrics)
        return model

    def predict_target(self):
        """Trains a model and predict target values (after data encoding)"""
        X = self.features_encoder.fit_transform(self.features_data)
        Y = self.target_encoder.fit_transform(
            self.target_data.values.reshape(-1, 1)
        )
        training_rows = self.target_data.notnull()
        X_train, Y_train = X[training_rows], Y[training_rows]

        self.model = self.build_model(self.target_type,
                                      X_train.shape[1], Y_train.shape[1],
                                      depth=2)
        self.train_model(X_train, Y_train)
        pred_sequence = training_data_sequence(X)
        Y_pred = self.model.predict_generator(pred_sequence)
        self.target_pred = pd.DataFrame(
            self.target_encoder.inverse_transform(Y_pred),
            index=self.target_data.index,
            columns=[self.target+TARGET_SUFFIX]
        )

    def train_model(self, x, y, min_improvement=10**-2, patience=3, epochs=100):
        calls = [EarlyStopping(monitor='loss', min_delta=min_improvement,
                               patience=patience, restore_best_weights=True)]
        steps_for_epoch = 10
        batch_size = math.ceil(x.shape[0] / steps_for_epoch)
        train_sequence = training_data_sequence(x, y, batch_size=batch_size)
        self.model.fit_generator(train_sequence, steps_per_epoch=steps_for_epoch,
                                 verbose=2, callbacks=calls, epochs=epochs)

    def save_dataset(self, output_filename):
        """
        Join original dataset with additional target prediction
        and save result in a file

        :param output_filename: Output filename with full path
        """
        output = copy.deepcopy(self.features_data)
        output = output.join(self.target_data)
        output = output.join(self.target_pred)
        # TODO: Apply to Targets dataframe and then join to features to avoid slicing
        if output_filename.endswith(".xlsx") \
                or output_filename.endswith(".xls"):
            index_slice = pd.IndexSlice[[self.target,
                                         self.target + TARGET_SUFFIX]]
            styler = output.style.apply(color_targets, axis=1,
                                        target_type=self.target_type,
                                        subset=index_slice)
            styler.to_excel(output_filename, sheet_name='DataFiller Output',
                            index=False, freeze_panes=(1, 1))
        else:
            output.to_csv(output_filename)


class training_data_sequence(Sequence):
    def __init__(self, x, y=None, batch_size=32):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.__makebatch__(self.x, idx), self.__makebatch__(self.y, idx) \
            if self.y is not None else self.__makebatch__(self.x, idx)

    def __makebatch__(self, variable, idx):
        batch = variable[idx * self.batch_size:(idx + 1) * self.batch_size]
        if scipy.sparse.issparse(variable):
            batch = batch.todense()
        return batch


# TODO: Move function to datatype class
def color_targets(row, target_type, tolerance=0.1, color_new='#98FF99',
                  color_different='#FF9999', color_match='transparent'):
    """
        Colors predicted target values depending on the original values.

        :param row: Dataframe row to be used with dataframe.style.apply
        :param target_type: Target type to be formatted ('label','number')
        :param tolerance: Percentage delta to consider numeric fields equal
        :param color_new: Color for new calculated values (default: '#98FF99')
        :param color_different: Color for calculated values not matching (default: '#FF9999')
        :param color_match: Color for calculated values matching (default: 'transparent')
        :return: List with background-color specification for the dataframe row
    """
    old_value = row[0]
    if old_value is np.NAN:  # TODO: Breakdown, first color NAN then different
        color = color_new
    elif target_type.name == 'number':  # TODO: Not orthogonal, use Datatype or move there
        new_value = np.float(row[1])
        if abs(old_value - new_value) < \
                tolerance * abs(old_value + new_value) / 2:
            color = color_match
        else:
            color = color_different
    elif target_type.name in ['label','phrase']:
        new_value = row[1]
        if old_value == new_value:
            color = color_match
        else:
            color = color_different
    else:
        raise Exception(f"Target type {target_type} not recognized")
    return ['', f'background-color: {color}']
