# TODO: Profile and optimize memory use
# TODO: Add example data and script
# TODO: Recognize Categories
# TODO: Recognize Model
# TODO: Recognize Colors
# TODO: Date category

"""
    Tool that completes and validates a dataset field.
    It takes an input file with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    @author: Francesco Baldisserri
    @email: fbaldisserri@gmail.com
    @version: 1.3
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, LabelBinarizer

FILETYPES = ["xlsx", "xls", "csv"]
SUFFIX = '_NEW'
INTEGERS_TYPES = ['int16', 'int32', 'int64']
DECIMAL_TYPES = ['float16', 'float32', 'float64']

def main():
    # Read parameters and input data
    master = tk.Tk()  # TODO: Use "with ..."?
    if "-i" in sys.argv:
        input_path = sys.argv[sys.argv.index("-i") + 1]
    else:
        filetypes = [(ext.upper()+" files", "*."+ext) for ext in FILETYPES]
        input_path = askopenfilename(title="Select input file",
                                     filetypes=filetypes,
                                     parent=master)

    filename, file_extension = os.path.splitext(input_path)
    if file_extension in [".csv"]:
        data = pd.read_csv(input_path, low_memory=False, dtype=str)
    elif file_extension in [".xls", ".xlsx"]:
        data = pd.read_excel(input_path, dtype=str)
    if "-t" in sys.argv:
        target = sys.argv[sys.argv.index("-t") + 1]
    else:
        target = select_target(tuple(data.columns.values), master)
    master.destroy()

    print(f"Calculating {target}")
    filler = DataFiller(data, target)
    filler.classify_fields()
    filler.encode_values()
    filler.predict_target()
    print("Process complete\n%s" % filler.predicted_target.head())
    filler.save_dataset(f"{filename}_OUTPUT_{filler.target}.xlsx")


class DataFiller:
    """
    Complete and validate a dataset target column through machine learning
    """

    def __init__(self, input_data, target):
        """
        Class constructor, initialized with Dataset to be filled

        :param input_data: Dataframe with data to be used
        :param target: String with column to be evaluated and completed
        """
        self.input_data = input_data
        self.target = target

    def classify_fields(self):
        """
            Classify the dataset fields by type:
            - "label": any string or number that is used as a category
            (e.g. number_of_car_doors = {2,4})
            - "number": any discrete or continuous quantity
            - "ignore": columns that should not be considered
            (e.g. unique record IDs)
        """
        # TODO: EAN and Customs Code in Full Export are mislabeled as numbers
        self.data_types = {}
        for field in self.input_data.columns:
            field_values = self.input_data[field].dropna()
            field_values = field_values.apply(pd.to_numeric,
                                              errors='ignore')
            if len(field_values.unique()) <= 1:  # ignore single or no value fields
                self.data_types[field] = DataType("ignore")
            elif field_values.dtype in DECIMAL_TYPES:
                self.data_types[field] = DataType("number")
            elif field_values.dtype in INTEGERS_TYPES \
                    and len(field_values.unique()) > len(field_values) / 100:
                # integer fields with many unique values (more than 1%)
                # are considered numbers, otherwise they are 'label'
                self.data_types[field] = DataType("number")
            elif len(field_values.unique()) == len(field_values):
                self.data_types[field] = DataType("ignore")
            else:
                self.data_types[field] = DataType("label")

    def encode_values(self):
        """Prepare dataset by normalizing and encoding the dataset fields"""
        # TODO: Process multiple targets at the same time
        valid_fields = [f for f in self.input_data.columns
                        if self.data_types[f].name != 'ignore']
        encoded_values = pd.DataFrame(index=self.input_data.index)
        for field in valid_fields:
            print(f"Encoding {field}")
            type = self.data_types[field]
            raw_value = self.input_data[field].values.reshape(-1, 1)
            encoded_value = type.imputer.fit_transform(raw_value)
            encoded_value = type.encoder.fit_transform(encoded_value)
            if field == self.target:
                self.encoded_target = encoded_value
                self.target_encoder = type.encoder
            else:
                encoded_values = encoded_values.join(
                    pd.DataFrame(encoded_value,
                                 index=self.input_data.index),
                    rsuffix="_" + field)

        self.encoded_features = encoded_values.values

    def predict_target(self):
        """Trains a model and predict target values after data encoding"""
        target_values = self.input_data[self.target]
        training_rows = self.input_data.index[target_values.notnull()]
        x = self.encoded_features[training_rows]
        y = self.encoded_target[training_rows]
        x_features, y_features = x.shape[1], y.shape[1]
        neurons = int((x_features + y_features)**(1/2))
        depth = 1
        patience = 3
        min_improvement = 10 ** -3
        epochs = 100

        # Model training
        self.model = tf.keras.models.Sequential()
        input_layer = tf.keras.layers.Dense(neurons, input_shape=x.shape[1:])
        self.model.add(input_layer)
        for d in range(1, depth):
            self.model.add(tf.keras.layers.Dense(neurons))

        calls = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  min_delta=min_improvement,
                                                  patience=patience,
                                                  mode='auto',
                                                  restore_best_weights=True)]
        target_datatype = self.data_types[self.target]
        output_layer = tf.keras.layers.Dense(y.shape[1],
                                             activation=target_datatype.activation)
        self.model.add(output_layer)
        self.model.compile(optimizer='adam',
                           loss=target_datatype.loss,
                           metrics=target_datatype.metrics)

        # train model and predict targets
        print(f"Training Model\nSamples: {len(x)}\nFeatures: {x_features}\t"
              f"Targets: {y_features}\n{self.model.summary()}")
        self.model.fit(x, y, epochs=epochs, verbose=2, callbacks=calls)
        pred_y = self.model.predict(self.encoded_features)
        self.predicted_target = pd.DataFrame(
            self.target_encoder.inverse_transform(pred_y),
            columns=[self.target+SUFFIX],
            index=self.input_data.index
        )

    def save_dataset(self, output_filename):
        """
        Join original dataset with additional target prediction
        and save result in a file

        :param output_filename: Output filename with full path
        """
        output = copy.deepcopy(self.input_data)
        original_target = pd.DataFrame(output.pop(self.target))
        output = output.join(original_target)
        output = output.join(self.predicted_target)

        if output_filename.endswith(".xlsx") \
                or output_filename.endswith(".xls"):
            index_slice = pd.IndexSlice[[self.target, self.target + SUFFIX]]
            target_type = self.data_types[self.target].name
            styler = output.style.apply(color_targets,
                                        target_type=target_type,
                                        axis=1,
                                        subset=index_slice)
            styler.to_excel(output_filename, sheet_name='Output',
                            index=False, freeze_panes=(1, 1))
        else:
            output.to_csv(output_filename)


class DataType:
    """ Micro class to manager different data fields"""
    def __init__(self, type_name):
        """
        Construct a data type given its type

        :param type_name: Type of data to be built ('label', 'number' or 'ignore')
        """
        self.name = type_name
        if self.name == 'label':
            self.imputer = SimpleImputer(strategy='constant',
                                         fill_value='missing')
            self.encoder = LabelBinarizer()
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
            self.activation = 'softmax'
        elif self.name == 'number':
            self.imputer = SimpleImputer(strategy='median')
            self.encoder = MinMaxScaler()
            self.loss = 'mean_squared_error'
            self.metrics = [],
            self.activation = 'linear'


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
    if old_value is np.NAN:
        color = color_new
    elif target_type == 'number':  # TODO: Format numbers as integers, floats, ...
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


def select_target(fields, master):
    """
    Graphical dropdown selector for target field

    :param fields: List of strings with all possible fields for the target
    :param master: Tkinter master to use for the window
    :return: String with selected target
    """
    master.title("Select the desired target field")
    selector = tk.StringVar(master)
    selector.set(fields[0])  # default value
    tk.OptionMenu(master, selector, *fields).pack()
    tk.Button(master, text="Select", command=master.quit).pack()
    tk.mainloop()
    return selector.get()


if __name__ == '__main__':
    main()