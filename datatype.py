"""
    Class that represents the different data types used in DataFiller

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.4
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


INTEGERS_TYPES = ['int16', 'int32', 'int64']
DECIMAL_TYPES = ['float16', 'float32', 'float64']


def classify_data(values, sparse_output):
    """
        Classify the dataset fields by datatype:
        - "label": any string or number that is used as a category
        (e.g. number_of_car_doors = {2,4})
        - "number": any discrete or continuous quantity
        - "ignore": columns that should not be considered
        (e.g. unique record IDs)

        :param values: Series or list with values to classify
        :param sparse_output: Boolean to use sparse matrixes for encoding
        :return DataType instance of the datatype identified
    """
    # TODO: EAN and Customs Code in Full Export are mislabeled as numbers
    field_values = pd.Series(values).dropna()
    field_values = field_values.apply(pd.to_numeric, errors='ignore')
    if len(field_values.unique()) <= 1:  # ignore single or no value fields
        datatype = Ignore(values)
    elif field_values.dtype in DECIMAL_TYPES:
        datatype = Number(values, sparse_output)
    elif field_values.dtype in INTEGERS_TYPES \
            and len(field_values.unique()) > len(field_values) / 100:
        # integer fields with many unique values (more than 1%)
        # are considered numbers, otherwise they are 'label'
        datatype = Number(values, sparse_output)
    elif len(field_values.unique()) == len(field_values):
        datatype = Ignore(values, sparse_output)
    else:
        datatype = Label(values, sparse_output)
    return datatype


class DataType:
    """Generic parent class that it's inherited by datatypes"""
    def __init__(self, type_name, values, sparse_output=False, loss=None,
                 metrics=None, activation=None, imputer=None, encoder=None):
        """
        Construct the type class starting from the values

        :param type_name: Name of the Datatype ('number', 'label', ...)
        :param values: Input values of the datatype
        :param sparse_output: Boolean to use sparse matrix (for large datasets)
        :param loss: String to select which Keras loss to use
        if the type is a target field
        :param metrics: String to select which Keras metrics to use
        if the type is a target field
        :param activation: String to select which Keras activation to use
        if the type is a target field
        :param imputer: Sklearn imputer to be used during encoding
        :param encoder: Sklearn encoder to be used during encoding
        """
        self.type = type_name
        self.values = np.reshape(np.array(values), (-1, 1))
        self.sparse = sparse_output
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.imputer = imputer
        self.encoder = encoder

    def encode_values(self):
        """Encode the values to use them in a machine learning model"""
        output_values = self.imputer.fit_transform(self.values)
        return self.encoder.fit_transform(output_values)

    #  TODO: Implement decode_values function


class Label(DataType):
    """ Class to manage text data types"""
    def __init__(self, values, sparse_output=False):
        DataType.__init__(self, type_name='label', values=values, sparse_output=sparse_output,
                          loss='binary_crossentropy', metrics=['accuracy'],
                          activation='softmax',
                          imputer=SimpleImputer(strategy='constant',
                                                fill_value='missing'),
                          encoder=LabelBinarizer(sparse_output=sparse_output))


class Number:
    """ Class to manage number data types"""
    def __init__(self, values, sparse_output=False):
        DataType.__init__(self, type_name='number', values=values,
                          sparse_output=sparse_output,
                          loss='mean_squared_error',
                          metrics=None, activation='linear',
                          imputer=SimpleImputer(strategy='median'),
                          encoder=MinMaxScaler())

    def encode_values(self):
        encoded_values = DataType.encode_values(self)
        if self.sparse:
            return sparse.csr_matrix(encoded_values)
        else:
            return encoded_values


class Phrase(DataType):  # TODO: Implement phrase datatype (multilabel)
    """ Class to manage text data types"""
    def __init__(self, values, sparse_output=False):
        DataType.__init__(self, type_name='phrase', values=values, sparse_output=sparse_output,
                          loss='binary_crossentropy', metrics=['accuracy'],
                          activation='softmax',
                          imputer=SimpleImputer(strategy='constant',
                                                fill_value='missing'),
                          encoder=LabelBinarizer(sparse_output=sparse_output))


class Ignore:
    """ Class to manage data types to be ignored"""
    def __init__(self, values, sparse=False):
        DataType.__init__(self, values, sparse)
        self.type = 'ignore'
