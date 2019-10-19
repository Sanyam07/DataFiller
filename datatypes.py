"""
    Classes to classify data types for machine learning

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.5
"""

import pandas as pd

INTEGERS = ['int16', 'int32', 'int64']
DECIMALS = ['float16', 'float32', 'float64']

NUMBER_TYPE = 'number'
LABEL_TYPE = 'label'
PHRASE_TYPE = 'phrase'
IGNORE_TYPE = 'ignore'


def classify_field(field_values):
    """Classify the dataset fields by DataType class"""
    temp_values = field_values.dropna()
    temp_values = temp_values.apply(pd.to_numeric, errors='ignore')
    if len(temp_values.unique()) <= 1:  # ignore fields with 0 or 1 values
        data_type = IGNORE_TYPE
    elif temp_values.dtype in DECIMALS:
        data_type = NUMBER_TYPE
    elif len(temp_values.unique()) < len(temp_values) / 10:
        data_type = LABEL_TYPE
    elif temp_values.dtype in INTEGERS:
        data_type = NUMBER_TYPE
    else:
        data_type = PHRASE_TYPE
    return data_type
