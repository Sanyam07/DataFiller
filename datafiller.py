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
from sklearn.compose import ColumnTransformer

import datatypes as dt
import transformers as tr
import modelfactory as mf

TARGET_SUFFIX = '_NEW'


class DataFiller:
    """
    Complete and validate a target column in a dataset through machine learning
    """

    def __init__(self, input_data, target):
        """
        Class constructor, initialized with Dataset to be filled

        :param input_data: Dataframe with data to be used
        :param target: String with column to be evaluated and completed
        """
        self.target = target
        self.features_data = input_data.drop(target, axis=1)
        self.target_data = input_data[target]
        self.features_transformer = self.build_features_transformer()
        self.target_type = dt.classify_field(self.target_data)
        self.target_transformer = tr.build_target_transformer(self.target_data)
        self.model = None   # TODO: Revisit model creation at init

    def build_features_transformer(self):
        """Prepare multi-feature processing pipeline"""
        transformers = []
        for feature in self.features_data.columns:
            transformer = tr.build_feature_transformer(self.features_data[feature])
            if transformer is not None:
                transformers += [(feature, transformer, [feature])]
        return ColumnTransformer(transformers)

    def predict_target(self):
        """ Trains a model and predict target values (after data encoding) """
        training_rows = self.target_data.index[self.target_data.notnull()]
        X = self.features_transformer.fit_transform(self.features_data)
        X_train = X[training_rows]
        Y_train = self.target_transformer.fit_transform(
            self.target_data.values.reshape(-1, 1)[training_rows]
        )
        model_factory = mf.ModelFactory(X_train.shape[1],
                                        Y_train.shape[1],
                                        self.target_type,
                                        depth=2)
        self.model = model_factory.build_model()
        print(f"Model:\n{self.model.summary()}")
        self.train_model(X_train, Y_train)
        Y_pred = self.model.predict(X)
        return pd.DataFrame(
            self.target_transformer.inverse_transform(Y_pred),
            index=self.target_data.index,
            columns=[self.target+TARGET_SUFFIX]
        )

    def train_model(self, X, Y, epochs=100):
        # TODO: Insert data shuffling before training?
        callbacks = mf.get_callbacks()
        self.model.fit(X, Y, verbose=2, callbacks=callbacks,
                       epochs=epochs, validation_split=0.2)

    def save_dataset(self, output_filename, target_pred):
        """
        Join original dataset with additional target prediction
        and save result in a file
        """
        output = copy.deepcopy(self.features_data)
        output = output.join(self.target_data)
        output = output.join(target_pred)
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


# TODO: Where to put it?
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
    if pd.isnull(old_value):  # TODO: Breakdown, first color NAN then different
        color = color_new
    elif target_type == dt.NUMBER_TYPE:  # TODO: Not orthogonal, use Datatype or move there
        old_value, new_value = np.float(row[0]), np.float(row[1])  # TODO: Datafield OLD_VALUE loosing formatting
        if abs(old_value - new_value) < \
                tolerance * abs(old_value + new_value) / 2:
            color = color_match
        else:
            color = color_different
    elif target_type in [dt.LABEL_TYPE, dt.PHRASE_TYPE]:
        new_value = row[1]
        if old_value == new_value:
            color = color_match
        else:
            color = color_different
    else:
        raise Exception(f"Target type {target_type} not recognized")
    return ['', f'background-color: {color}']
