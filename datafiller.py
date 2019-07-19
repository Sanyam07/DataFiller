"""
    Class that completes and validates a dataset field.
    It takes an input Dataframe with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.6
"""

import copy
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
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
        self.dataset = input_data
        self.target = target
        self.sparse = sparse_matrix
        self.data_fields = {'ignore': [], 'number': [],
                            'phrase': [], 'label': []}
        self.encoded_features = None
        self.pipeline = None

    def classify_fields(self):
        """Classify the dataset fields by DataType class"""
        for field_name in self.dataset.columns:
            test_values = self.dataset[field_name].dropna()
            test_values = test_values.apply(pd.to_numeric, errors='ignore')
            if len(test_values.unique()) <= 1:  # ignore single or no value fields
                self.data_fields['ignore'] += [field_name]
            elif test_values.dtype in DECIMALS_TYPES:
                self.data_fields['number'] += [field_name]
            elif len(test_values.unique()) < len(test_values) / 10:
                self.data_fields['label'] += [field_name]
            elif test_values.dtype in INTEGERS_TYPES:
                self.data_fields['number'] += [field_name]
            else:
                self.data_fields['phrase'] += [field_name]

    def prepare_pipeline(self):  # TODO: Manage Sparse Output
        """Prepare multi-feature processing pipeline"""
        number_transformer = make_pipeline(
            SimpleImputer(strategy='median'),
            MinMaxScaler()
        )
        label_transformer = make_pipeline(
            SimpleImputer(strategy='constant', fill_value='missing'),
            OneHotEncoder(sparse=self.sparse)
        )
        phrase_transformer = make_pipeline(
            CountVectorizer()
            #TfidfTransformer,
            #DenseTransformer()
        )
        preprocessor = ColumnTransformer([  # TODO: Make orthogonal
                ('number', number_transformer, self.data_fields['number']),
                ('label', label_transformer, self.data_fields['label']),
                ('phrase', phrase_transformer, self.data_fields['phrase'])
        ])

        results = preprocessor.fit_transform(self.dataset)
        keras_model = tf.keras.wrappers.scikit_learn.KerasRegressor(self.prepare_model)
        self.pipeline = make_pipeline(preprocessor, keras_model)

    def prepare_model(self, neurons_per_layer=100, depth=1,
                      min_improvement=10**-2, patience=2):
        """Creates Keras model as a last step of the Pipeline"""
        model = tf.keras.models.Sequential()
        input_layer = tf.keras.layers.Dense(neurons_per_layer)
        model.add(input_layer)
        for d in range(1, depth):
            self.model.add(tf.keras.layers.Dense(neurons_per_layer))
        calls = [tf.keras.callbacks.EarlyStopping(monitor='loss',  # TODO: Add to Model
                                                  min_delta=min_improvement,
                                                  patience=patience,
                                                  restore_best_weights=True)]
        target_datatype = self.data_fields[self.target]
        output_layer = tf.keras.layers.Dense(activation=target_datatype.activation)
        model.add(output_layer)
        model.compile(optimizer='adam',
                      loss=target_datatype.loss,
                      metrics=target_datatype.metrics)
        return model

    def predict_target(self):
        """Trains a model and predict target values (after data encoding)"""
        training_data = self.dataset[self.dataset[self.target].notnull()]
        self.pipeline.fit_transform(training_data)
        self.target_predicted = self.pipeline.fit_transform(self.dataset)

    def save_dataset(self, output_filename):
        """
        Join original dataset with additional target prediction
        and save result in a file

        :param output_filename: Output filename with full path
        """
        output = copy.deepcopy(self.dataset)
        original_target = pd.DataFrame(output.pop(self.target))
        output = output.join(original_target)
        output = output.join(self.target_predicted)

        if output_filename.endswith(".xlsx") \
                or output_filename.endswith(".xls"):
            index_slice = pd.IndexSlice[[self.target, self.target + TARGET_SUFFIX]]
            target_type = self.data_fields[self.target].type
            styler = output.style.apply(color_targets, axis=1,
                                        target_type=target_type,
                                        subset=index_slice)
            styler.to_excel(output_filename, sheet_name='Output',
                            index=False, freeze_panes=(1, 1))
        else:
            output.to_csv(output_filename)

class DenseTransformer():
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def batch_generator(x, y, batch_size):
    """Generate batches for low memory mode using sparse matrices"""
    batches_for_epoch = math.ceil(x.shape[0] / batch_size)
    i = 0
    while True:
        index_batch = range(x.shape[0])[batch_size * i:batch_size * (i + 1)]
        x_batch = np.array(x[index_batch, :].todense())
        y_batch = np.array(y[index_batch, :].todense())
        yield(np.array(x_batch), y_batch)
        i = (i + 1) % batches_for_epoch


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
