# TODO: Add example data and script
# TODO: Recognize Categories
# TODO: Recognize Model
# TODO: Recognize Colors

"""
    Tool that completes and validates a dataset field.
    It takes an input file with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    @author: Francesco Baldisserri
    @email: fbaldisserri@gmail.com
    @version: 1.4
"""

import os
import sys
import copy
import math
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
from scipy import sparse
from tkinter.filedialog import askopenfilename

import datatype as dt

FILETYPES = ["xlsx", "xls", "csv"]
SUFFIX = '_NEW'
LOW_MEMORY_MODE = False  # TODO: Revisit LowMemory option


def main():
    input_file = get_inputfile()

    filename, file_extension = os.path.splitext(input_file)
    if file_extension in [".csv"]:
        input_data = pd.read_csv(input_file, low_memory=False, dtype=str)
    elif file_extension in [".xls", ".xlsx"]:
        input_data = pd.read_excel(input_file, dtype=str)

    target = get_target(input_data)
    print(f"Calculating {target}")
    filler = DataFiller(input_data, target, low_memory=LOW_MEMORY_MODE)
    filler.classify_fields()
    filler.encode_values()
    filler.predict_target()
    print("Process complete\n%s" % filler.target_predicted.head())
    filler.save_dataset(f"{filename}_OUTPUT_{filler.target}.xlsx")


def get_inputfile():
    """
    Get source filename from arguments or through graphical selector

    :return: Input filename
    """
    master= tk.Tk()
    if "-i" in sys.argv:
        input_path = sys.argv[sys.argv.index("-i") + 1]
    else:
        filetypes = [(ext.upper() + " files", "*." + ext) for ext in FILETYPES]
        input_path = askopenfilename(title="Select input file",
                                     filetypes=filetypes,
                                     parent=master)
    master.destroy()
    return input_path


def get_target(input_data):
    """
    Read parameters and input data

    :param input_data: Input file path
    :return: Dataset, Target columns and filename
    """
    master = tk.Tk()
    if "-t" in sys.argv:
        target = sys.argv[sys.argv.index("-t") + 1]
    else:
        target = select_target(tuple(input_data.columns.values), master)
    master.destroy()
    return target


class DataFiller:
    """
    Complete and validate a target column in a dataset through machine learning
    """

    def __init__(self, input_data, target, low_memory=False):
        """
        Class constructor, initialized with Dataset to be filled

        :param input_data: Dataframe with data to be used
        :param target: String with column to be evaluated and completed
        :param low_memory: Boolean to limit memory usage via sparse matrixes
        (for large datasets)
        """
        self.input_data = input_data
        self.target = target
        self.low_memory = low_memory

    def classify_fields(self):
        """Classify the dataset fields by DataType class"""
        self.data_types = {}
        for field in self.input_data.columns:
            self.data_types[field] = dt.classify_data(self.input_data[field],
                                                      self.low_memory)

    def encode_values(self):
        """Prepare dataset by normalizing and encoding the dataset fields"""
        valid_fields = [f for f in self.input_data.columns
                        if self.data_types[f].type != 'ignore']

        if self.low_memory:
            hstack_function = lambda x,y: sparse.hstack((x,y), format='csr') \
                if x is not None else y
        else:
            hstack_function = lambda x,y: np.hstack((x,y)) \
                if x is not None else y

        self.features_values = None
        for field in valid_fields:
            print(f"Encoding {field}")
            data_type = self.data_types[field]
            if field == self.target:
                self.target_values = data_type.encode_values()
                self.target_encoder = data_type.encoder  # TODO: Create DECODE function in Datatype
            else:
                encoded_value = data_type.encode_values()
                self.features_values = hstack_function(self.features_values,
                                                       encoded_value)

    def predict_target(self):
        """Trains a model and predict target values (after data encoding)"""
        target_values = self.input_data[self.target]
        training_rows = self.input_data.index[target_values.notnull()]
        x = self.features_values[training_rows]
        y = self.target_values[training_rows]
        x_features, y_features = x.shape[1], y.shape[1]
        depth = 1
        neurons = int((x_features + y_features)**(1/2))  # TODO: Use geometric mean
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
                                                  restore_best_weights=True)]
        target_datatype = self.data_types[self.target]
        output_layer = tf.keras.layers.Dense(y.shape[1],
                                             activation=target_datatype.activation)
        self.model.add(output_layer)
        self.model.compile(optimizer='adam',
                           loss=target_datatype.loss,
                           metrics=target_datatype.metrics)

        # train model and predict targets
        print(f"Training Model\nSamples: {x.shape[0]}\nFeatures: {x_features}\t"
              f"Targets: {y_features}\n{self.model.summary()}")
        if self.low_memory:
            batch_size = int(x.shape[0] / 10)
            steps_per_epoch = math.ceil(x.shape[0] / batch_size)
            generator = batch_generator(x, y, batch_size)
            self.model.fit_generator(generator=generator, epochs=epochs, verbose=2,
                                     callbacks=calls, steps_per_epoch=steps_per_epoch)
        else:
            self.model.fit(x, y, epochs=epochs, verbose=2, callbacks=calls)
        y_predicted = self.model.predict(self.features_values)
        self.target_predicted = pd.DataFrame(
            self.target_encoder.inverse_transform(y_predicted),
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
        output = output.join(self.target_predicted)

        if output_filename.endswith(".xlsx") \
                or output_filename.endswith(".xls"):
            index_slice = pd.IndexSlice[[self.target, self.target + SUFFIX]]
            target_type = self.data_types[self.target].type
            styler = output.style.apply(color_targets, axis=1,
                                        target_type=target_type,
                                        subset=index_slice)
            styler.to_excel(output_filename, sheet_name='Output',
                            index=False, freeze_panes=(1, 1))
        else:
            output.to_csv(output_filename)


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


def batch_generator(x, y, batch_size):
    """Generate batches for low memory mode using sparse matrixes"""
    batches_for_epoch = math.ceil(x.shape[0] / batch_size)
    i = 0
    while True:
        index_batch = range(x.shape[0])[batch_size * i:batch_size * (i + 1)]
        x_batch = np.array(x[index_batch, :].todense())
        y_batch = np.array(y[index_batch, :].todense())
        yield(np.array(x_batch), y_batch)
        i = (i + 1) % batches_for_epoch


if __name__ == '__main__':
    main()
