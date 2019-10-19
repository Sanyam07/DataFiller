"""
    Classes to transform different data types for machine learning

    author: Francesco Baldisserri
    email: fbaldisserri@gmail.com
    version: 0.6
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import datatypes as dt


def build_feature_transformer(data_values):
    """ Build data transformer based on a feature type """
    data_type = dt.classify_field(data_values)
    if data_type == dt.NUMBER_TYPE:
        transformer = NumberTransformer()
    elif data_type == dt.LABEL_TYPE:
        transformer = LabelTransformer()
    elif data_type == dt.PHRASE_TYPE:
        transformer = PhraseTransformer()
    else:
        transformer = None
    return transformer


def build_target_transformer(data_values):
    """
    Build an data transformer based on a target type
    which is invertible given that the target prediction
    will need to be decoded
    """
    data_type = dt.classify_field(data_values)
    if data_type == dt.NUMBER_TYPE:
        transformer = NumberTransformer()
    elif data_type == dt.LABEL_TYPE or data_type == dt.PHRASE_TYPE:
        transformer = LabelTransformer()
    else:
        transformer = None
    return transformer


class BaseTransformer(BaseEstimator, TransformerMixin):
    """ General Transformer class for all data types """
    def __init__(self, type_name, imputer, encoder):
        self.type_name = type_name
        self.encoder = make_pipeline(imputer, encoder)  # Encoding can be lossy

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X, **kwargs):
        return self.encoder.transform(X, **kwargs)

    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X)


class NumberTransformer(BaseTransformer):
    """ Transformer for quantities """
    def __init__(self):
        BaseTransformer.__init__(
            self,
            type_name=dt.NUMBER_TYPE,
            imputer=InvertibleImputer(strategy='median'),
            encoder=MinMaxScaler()
        )


class LabelTransformer(BaseTransformer):
    """ Transformer for categories """
    def __init__(self):
        BaseTransformer.__init__(
            self,
            type_name=dt.LABEL_TYPE,
            imputer=InvertibleImputer(strategy='constant',
                                      fill_value='n/a'),
            encoder=OneHotEncoder(handle_unknown='ignore')
        )


# TODO: Select encoding type (onehot, tfid, sequence)
# TODO: Make it invertible
class PhraseTransformer(BaseTransformer):
    """ Transformer for text sequences """
    def __init__(self):
        BaseTransformer.__init__(
            self,
            type_name=dt.PHRASE_TYPE,
            imputer=InvertibleImputer(strategy='constant',
                                      fill_value='n/a'),
            encoder=TfidfVectorizer(preprocessor=_column_concatenate,
                                    strip_accents='unicode',
                                    decode_error='ignore')
        )


class InvertibleImputer(SimpleImputer):  # TODO: Add transformer Mixin inheritance?
    """ Imputer that can be used also in Inverse_Transform tasks """
    def inverse_transform(self, X):
        return X


def _column_concatenate(x):
    return ' '.join([value for value in x])
