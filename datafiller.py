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
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

INTEGERS_TYPES = ['int16', 'int32', 'int64']
DECIMALS_TYPES = ['float16', 'float32', 'float64']
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
        self.data_fields = classify_fields(self.input_data)
        self.features_encoder = self.build_features_encoder()
        self.target_encoder = self.build_target_encoder()
        self.model = None
        self.target_pred = None

    def build_features_encoder(self):
        """Prepare multi-feature processing pipeline"""
        data_types = set(self.data_fields.values())-set(['ignore'])
        transformer = []
        for t in data_types:
            features_per_type = [feat for feat in self.features_data.columns
                                 if self.data_fields[feat] == t]
            transformer += [(t, get_feature_transformer(t), features_per_type)]
        return ColumnTransformer(transformer)

    def build_target_encoder(self):
        """Set the target encoder based on its type"""
        target_type = self.data_fields[self.target]
        return make_pipeline(get_field_encoder(target_type))

    def build_keras_model(self, target_type, features_dim, target_dim, depth=1):
        neurons = int(math.sqrt(features_dim+target_dim))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(neurons,
                                        activation='relu',
                                        input_dim=features_dim))
        for d in range(1, depth):
            model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        if target_type == 'label':  # TODO: Not orthogonal, create data_structure
            model.add(tf.keras.layers.Dense(target_dim, activation="sigmoid"))
            model.compile(optimizer='adam', loss='binary_crossentropy')
        elif target_type == 'number':
            model.add(tf.keras.layers.Dense(target_dim, activation="sigmoid"))
            model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            raise ValueError(f"Value {target_type} not recognized")
        return model

    def predict_target(self, min_improvement=10**-2, patience=3, epochs=100):
        """Trains a model and predict target values (after data encoding)"""
        features_train = self.features_data[self.target_data.notnull()]
        target_train = self.target_data[self.target_data.notnull()]
        x_train = self.features_encoder.fit_transform(
            features_train
        )
        y_train = self.target_encoder.fit_transform(
            target_train.values.reshape(-1,1),
        )
        self.model = self.build_keras_model(self.data_fields[self.target],
                                            x_train.shape[1], y_train.shape[1],
                                            depth=2)
        calls = [tf.keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=min_improvement,
            patience=patience, restore_best_weights=True)]
        steps_for_epoch = 10
        batch_size = math.ceil(x_train.shape[0]/ steps_for_epoch)
        train_sequence = datafiller_sequence(x_train, y_train, batch_size=batch_size)
        self.model.fit_generator(train_sequence, steps_per_epoch=steps_for_epoch,
                                 verbose=2, callbacks=calls, epochs=epochs)

        x_pred = self.features_encoder.transform(self.features_data)
        pred_sequence = datafiller_sequence(x_pred)
        y_pred = self.model.predict_generator(pred_sequence)
        self.target_pred = pd.DataFrame(
            self.target_encoder.inverse_transform(y_pred),
            index=self.target_data.index,
            columns=[self.target+TARGET_SUFFIX]
        )

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
            target_type = self.data_fields[self.target]
            styler = output.style.apply(color_targets, axis=1,
                                        target_type=target_type,
                                        subset=index_slice)
            styler.to_excel(output_filename, sheet_name='DataFiller Output',
                            index=False, freeze_panes=(1, 1))
        else:
            output.to_csv(output_filename)


class datafiller_sequence(tf.keras.utils.Sequence):
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


def classify_fields(input_data):
    """Classify the dataset fields by DataType class"""
    data_fields = {}
    for field_name in input_data.columns:
        temp_values = input_data[field_name].dropna()
        temp_values = temp_values.apply(pd.to_numeric, errors='ignore')
        if len(temp_values.unique()) <= 1:  # ignore single or no value fields
            data_fields[field_name] = 'ignore'
        elif temp_values.dtype in DECIMALS_TYPES:
            data_fields[field_name] = 'number'
        elif len(temp_values.unique()) < len(temp_values) / 10:
            data_fields[field_name] = 'label'
        elif temp_values.dtype in INTEGERS_TYPES:
            data_fields[field_name] = 'number'
        else:
            data_fields[field_name] = 'label'  # TODO: Implement TFIDF vectorizer
    return data_fields


def get_feature_transformer(field_type):
    return make_pipeline(
        get_field_imputer(field_type),
        get_field_encoder(field_type)
    )


def get_field_imputer(field_type):
    if field_type == 'number':
        imputer = SimpleImputer(strategy='median')
    elif field_type == 'label':
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
    else:
        raise ValueError(f"Field type {field_type} not recognized")
    return imputer


def get_field_encoder(field_type):
    if field_type == 'number':
        encoder = MinMaxScaler()
    elif field_type == 'label':
        encoder = OneHotEncoder(handle_unknown='ignore')
    else:
        raise ValueError(f"Field type {field_type} not recognized")
    return encoder


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
    elif target_type == 'number':  # TODO: Not orthogonal, use Datatype or move there
        new_value = np.float(row[1])
        if abs(old_value - new_value) < \
                tolerance * abs(old_value + new_value) / 2:
            color = color_match
        else:
            color = color_different
    elif target_type == 'label':
        new_value = row[1]
        if old_value == new_value:
            color = color_match
        else:
            color = color_different
    else:
        raise Exception(f"Target type {target_type} not recognized")
    return ['', f'background-color: {color}']
