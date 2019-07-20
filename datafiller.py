"""
    Class that completes and validates a dataset field.
    It takes an input Dataframe with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.6
"""

import copy
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor

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
        self.data_fields = {}
        self._classify_fields()
        self.pipeline = make_pipeline(
            self._prepare_preprocessor(),
            self._prepare_model()
        )
        self.target_encoder = None
        self._prepare_target_encoder()
        self.target_pred = None

    def _classify_fields(self):
        """Classify the dataset fields by DataType class"""
        for field_name in self.input_data.columns:
            temp_values = self.input_data[field_name].dropna()
            temp_values = temp_values.apply(pd.to_numeric, errors='ignore')
            if len(temp_values.unique()) <= 1:  # ignore single or no value fields
                self.data_fields[field_name] = 'ignore'
            elif temp_values.dtype in DECIMALS_TYPES:
                self.data_fields[field_name] = 'number'
            elif len(temp_values.unique()) < len(temp_values) / 10:
                self.data_fields[field_name] = 'label'
            elif temp_values.dtype in INTEGERS_TYPES:
                self.data_fields[field_name] = 'number'
            else:
                self.data_fields[field_name] = 'label'

    def _prepare_preprocessor(self):
        """Prepare multi-feature processing pipeline"""
        number_features = [f for f in self.features_data.columns
                           if self.data_fields[f] == 'number']
        number_transformer = make_pipeline(
            SimpleImputer(strategy='median'),
            MinMaxScaler()
        )
        label_features = [f for f in self.features_data.columns
                          if self.data_fields[f] == 'label']
        label_transformer = make_pipeline(
            SimpleImputer(strategy='constant', fill_value='missing'),
            OneHotEncoder(sparse=self.sparse, handle_unknown='ignore')
        )
        return ColumnTransformer([  # TODO: Make orthogonal
            ('number', number_transformer, number_features),
            ('label', label_transformer, label_features),
        ])

    def _prepare_target_encoder(self):
        """Set the target encoder based on its type"""
        target_type = self.data_fields[self.target]
        if target_type == 'label':
            self.target_encoder = OneHotEncoder(handle_unknown='ignore')
        elif target_type == 'number':
            self.target_encoder = MinMaxScaler()
        else:
            raise ValueError(f"Target type {target_type} not recognized")

    def _prepare_model(self):
        target_type = self.data_fields[self.target]
        if target_type == 'label':
            model = MLPClassifier()
        elif target_type == 'number':
            model = MLPRegressor()
        else:
            raise ValueError(f"Value {target_type} not recognized")
        return model

    def predict_target(self):
        """Trains a model and predict target values (after data encoding)"""
        features_train = self.features_data[self.target_data.notnull()]
        target_train = self.target_data[self.target_data.notnull()]
        y_train = self.target_encoder.fit_transform(
            target_train.values.reshape(-1,1)
        )
        self.pipeline.fit(features_train, y_train)
        y_pred = self.pipeline.predict(self.features_data)
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


class DenseTransformer():
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


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
