"""
Tester class for transformers

author: Francesco Baldisserri
"""

import unittest
import numpy as np
import pandas as pd
import transformers as tf


class NumberTransformerTests(unittest.TestCase):
    def setUp(self):
        self.transformer = tf.NumberTransformer()
        self.value = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        self.result = np.array([0, 0, 0.5, 0.5, 1, 1]).reshape(3, 2)

    def test_transform(self):
        output = tf.NumberTransformer().fit_transform(self.value)
        self.assertTrue(np.all(self.result == output))

    def test_inverse_transform(self):
        temp_output = self.transformer.fit_transform(self.value)
        output = self.transformer.inverse_transform(temp_output)
        self.assertTrue(np.allclose(self.value, output, equal_nan=True))


class LabelTransformerTests(unittest.TestCase):
    def setUp(self):
        self.transformer = tf.LabelTransformer()
        self.value = np.array(['one', 'two', 'three'],
                              dtype=object).reshape(3, 1)
        self.result = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).reshape(3, 3)

    def test_transform(self):
        output = self.transformer.fit_transform(self.value)
        self.assertTrue(np.all(self.result == output))

    def test_inverse_transform(self):
        temp_output = self.transformer.fit_transform(self.value)
        output = self.transformer.inverse_transform(temp_output)
        self.assertTrue(np.all(self.value == output))

"""
class PhraseTransformerTests(unittest.TestCase):
    def setUp(self):
        self.transformer = tf.PhraseTransformer()
        self.value = np.array(['I go home', 'You go home'],
                              dtype=object).reshape(2, 1)
        self.result = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0],
                                [0,0,0,1], [0,1,0,0], [0,0,1,0]]).reshape(2,3,4)

    def test_transform(self):
        output = self.transformer.fit_transform(self.value)
        self.assertTrue(np.all(self.result == output))

    def test_inverse_transform(self):
        temp_output = self.transformer.fit_transform(self.value)
        output = self.transformer.inverse_transform(temp_output)
        self.assertTrue(np.all(self.value == output))
"""

class InverseImputerTests(unittest.TestCase):
    def setUp(self):
        self.string_imputer = tf.InvertibleImputer(strategy='constant',
                                                   fill_value='n/a')
        self.string_value = np.array(['one', 'two',
                                      np.nan, 'four',
                                      np.nan, 'six'],
                                     dtype=object).reshape(3, 2)
        self.string_result = np.array(['one', 'two',
                                       'n/a', 'four',
                                       'n/a', 'six']).reshape(3, 2)

        self.number_value = np.array([1, 2, np.nan, 4, np.nan, 6],
                                     dtype='float32').reshape(3, 2)
        self.number_imputer = tf.InvertibleImputer(strategy='constant',
                                                   fill_value=999)
        self.number_result = np.array([1, 2, 999, 4, 999, 6],
                                      dtype='float32').reshape(3, 2)

    def test_transform_strings(self):
        output = self.string_imputer.fit_transform(self.string_value)
        self.assertTrue(np.all(self.string_result == output))

    def test_inverse_transform_strings(self):
        value_transformed = self.string_imputer.fit_transform(self.string_value)
        output = self.string_imputer.inverse_transform(value_transformed)
        fill_mask = pd.notnull(self.string_value)
        self.assertTrue(
            np.all(self.string_value[fill_mask] == output[fill_mask])
        )

    def test_transform_numbers(self):
        output = self.number_imputer.fit_transform(self.number_value)
        self.assertTrue(np.all(self.number_result == output))

    def test_inverse_transform_numbers(self):
        value_transformed = self.number_imputer.fit_transform(self.number_value)
        output = self.number_imputer.inverse_transform(value_transformed)
        fill_mask = pd.notnull(self.number_value)
        self.assertTrue(
            np.all(self.number_value[fill_mask] == output[fill_mask])
        )


if __name__ == '__main__':
    unittest.main()
