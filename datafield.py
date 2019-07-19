"""
    Class that represents a data field for machine learning tasks

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.5
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

INTEGERS_TYPES = ['int16', 'int32', 'int64']
DECIMAL_TYPES = ['float16', 'float32', 'float64']


class DataField:
    """Generic parent class that it's inherited by datatypes"""
    def __init__(self, field_type, field_name, values, sparse_output=False,
                 loss=None, metrics=None, activation=None):
        """
        Construct the type class starting from the values

        :param field_type: Field type ('label', 'number', ...)
        :param field_name: Name of the data field store
        ('sku', 'category', 'price', ...)
        :param values: Input values of the datatype
        :param sparse_output: Boolean to use sparse matrix (for large datasets)
        :param loss: String to select which Keras loss to use
        if the type is a target field
        :param metrics: String to select which Keras metrics to use
        if the type is a target field
        :param activation: String to select which Keras activation to use
        if the type is a target field
        """
        self.type = field_type
        self.name = field_name
        self.values = np.reshape(np.array(values), (-1, 1))
        self.sparse = sparse_output
        self.loss = loss
        self.metrics = metrics
        self.activation = activation

    #  TODO: Use pipelines?
    #  TODO: Implement decode_values function
    #  TODO: Generate Keras model here


class Label(DataField):
    """ Class to manage text data types"""
    def __init__(self, field_name, values, sparse_output=False):
        DataField.__init__(self, field_type='label', field_name=field_name,
                           values=values, sparse_output=sparse_output,
                           loss='binary_crossentropy', metrics=['accuracy'],
                           activation='softmax')
        self.imputer = SimpleImputer(strategy='constant',
                                     fill_value='missing')
        self.encoder = LabelBinarizer(sparse_output=sparse_output)

    def encode_values(self):
        """Encode the values to use them in a machine learning model"""
        filled_values = self.imputer.fit_transform(self.values).astype('str')
        return self.encoder.fit_transform(filled_values)


class Number:
    """ Class to manage number data types"""
    def __init__(self, name, values, sparse_output=False):
        DataField.__init__(self, field_type='number', field_name=name,
                           values=values, sparse_output=sparse_output,
                           loss='mean_squared_error',
                           metrics=None, activation='linear')
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = MinMaxScaler()

    def encode_values(self):
        """Encode the values to use them in a machine learning model"""
        encoded_values = self.imputer.fit_transform(self.values)
        encoded_values = self.encoder.fit_transform(encoded_values)
        if self.sparse:
            return sparse.csr_matrix(encoded_values)
        else:
            return encoded_values


class Phrase(DataField):
    # TODO: Implement inverse transform
    # https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
    # https://towardsdatascience.com/generating-text-with-lstms-b0cdb6fc7bca
    """ Class to manage text data types"""
    def __init__(self, field_name, values, sparse_output=False):
        DataField.__init__(self, field_type='phrase', field_name=field_name, values=values,
                           sparse_output=sparse_output,
                           loss='binary_crossentropy', metrics=['accuracy'],
                           activation='softmax')
        self.imputer = SimpleImputer(strategy='constant',
                                fill_value='missing')
        self.encoder = TfidfVectorizer(decode_error='replace',
                                  strip_accents='unicode')

    def encode_values(self):
        """Encode the values to use them in a machine learning model"""
        filled_values = self.imputer.fit_transform(self.values).ravel()
        encoded_values = self.encoder.fit_transform(filled_values)
        if self.sparse:
            return encoded_values
        else:
            return encoded_values.toarray()


class Ignore(DataField):
    """ Class to manage data types to be ignored"""
    def __init__(self, field_name, values):
        DataField.__init__(self, field_type='ignore', field_name=field_name, values=values)


def classify_data(field_name, values, is_target, sparse_output):
    """
        Classify the dataset fields by datatype:
        - "label": any string or number that is used as a category
        (e.g. number_of_car_doors = {2,4})
        - "number": any discrete or continuous quantity
        - "phrase": any long string with multiple labels
        - "ignore": columns that should not be considered
        (e.g. unique record IDs)

        :param field_name: Name of the field to be classified
        :param values: Series or list with values to classify
        :param is_target: Boolean to specify if the field is the target
        :param sparse_output: Boolean to use sparse matrixes for encoding
        :return DataType instance of the datatype identified
    """
    test_values = pd.Series(values).dropna()
    test_values = test_values.apply(pd.to_numeric, errors='ignore')
    if len(test_values.unique()) <= 1:  # ignore single or no value fields
        datatype = Ignore(field_name, values)
    elif values.dtype in DECIMAL_TYPES:
        datatype = Number(field_name, values, sparse_output)
    elif len(test_values.unique()) < len(values) / 10:
        datatype = Label(field_name, values, sparse_output)
    elif test_values.dtype in INTEGERS_TYPES:
        datatype = Number(field_name, values, sparse_output)
    else:
        datatype = Label(field_name, values, sparse_output) if is_target \
            else Phrase(field_name, values, sparse_output)
    return datatype


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