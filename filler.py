# TODO: Recognize Categories
# TODO: Recognize Model
# TODO: Recognize Colors
# TODO: Date category
# TODO: Profile and optimize memory use

"""
    Tool that completes and validates CSV, XLS and XLSX datasets.
    It takes files with values missing from one or more columns (targets)
    and completes them with a Neural Network Algorithm using existing examples.
"""

import os
import copy
import numpy as np
import pandas as pd
from tkinter import *
import tensorflow as tf
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SUFFIX = '_NEW'
__version__ = '1.1'


def color_targets(row, target_type, tolerance=0.1, color_new='#98FF99',
                  color_different='#FF9999', color_match='transparent'):
    """
        Colors predicted target values depending on the original values.

        :param row: Dataframe row to be used with dataframe.style.apply
        :param target_type: Target type to be formatted ('label','number')
        :param tolerance: Percentage to consider numeric fields equal
        :param color_new: Color for new calculated values (default: '#98FF99')
        :param color_different: Color for calculated values not matching (default: '#FF9999')
        :param color_match: Color for calculated values matching (default: 'transparent')
        :return: List with background-color specification for the dataframe row
    """
    old_value, new_value = row[0], row[1]
    if old_value is np.NAN:
        color = color_new
    elif target_type == 'number':  # TODO: Format numbers as integers, floats, ...
        new_value = np.float(new_value)
        if abs(old_value - new_value) < tolerance * abs(old_value + new_value) / 2:
            color = color_match
        else:
            color = color_different
    elif target_type == 'label':  # In case old_value value is not a number only equality is considered a color_match
        if old_value == new_value:
            color = color_match
        else:
            color = color_different
    else:
        raise Exception(f"Target type {target_type} not recognized")
    return ['', f'background-color: {color}']


def classifyColumns(data):
    """
        Classify the dataset columns by type:
        - "label": any string or number that is used as a category
        (e.g. number_of_car_doors = {2,4})
        - "number": any discrete or continuous quantity
        - "ignore": columns that should not be considered
        (e.g. unique record IDs)

        :param data: Dataframe with dataset data to be analyzed
        :return: Dictionary with all Dataframe columns and their
        string classification: "label","number","ignore"
    """
    classes = {}
    int = ['int16', 'int32', 'int64']
    floats = ['float16', 'float32', 'float64']
    columns = list(data.columns)
    for column in columns:
        df = data[column].dropna()
        df = df.apply(pd.to_numeric, errors='ignore')
        if len(df.unique()) <= 1:  # ignore single value column
            classes[column] = "ignore"
        # TODO: EAN and Customs Code in Full Export are mislabeled as numbers
        # "number" are all float fields and integer fields with
        # "many" (more than 1%) total unique values
        elif df.dtype in int and len(df.unique()) > len(df) / 100 \
                or df.dtype in floats:
            classes[column] = "number"
        elif len(df.unique()) == len(df):
            classes[column] = "ignore"
        else:
            classes[column] = "label"
    return classes


def data_calculation(data, target):
    """
    Predict existing and missing values for targets fields

    :param data: Dataframe with complete dataset (including targets)
    :param target: String with column to be evaluated and completed
    :return: Dataframe with original data plus calculated target columns
    at the end (followed by '_NEW' suffix)
    """
    classes = classifyColumns(data)
    output = copy.deepcopy(data)
    # TODO: Process multi-targets at the same time
    print(f"Calculating values for {target}")

    # identify features, targets, training and test rows
    features = list(data.columns)
    features.remove(target)
    target_rows = list(data[data[target].isna()].index)
    train_rows = list(set(data.index) - set(target_rows))

    # normalize raw data
    label_features = [f for f in features if classes[f] == "label"]
    number_features = [f for f in features if classes[f] == "number"]
    feature_data = pd.DataFrame(index=data.index)
    label_imp = SimpleImputer(strategy='constant', fill_value='missing')
    label_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # TODO: Word vector instead of TFID
    # label_enc = TfidfVectorizer(token_pattern="\w", strip_accents="unicode",
    #                          decode_error="replace")

    for lab in label_features:
        print(f"Encoding {lab}")
        ldata = label_imp.fit_transform(data[lab].values.reshape(-1, 1))
        ldata = label_enc.fit_transform(ldata)
        feature_data = feature_data.join(
            pd.DataFrame(ldata, index=data.index), rsuffix="_" + lab)

    number_imp = SimpleImputer(strategy='median')
    number_sca = MinMaxScaler()
    for num in number_features:
        print(f"Encoding {num}")
        ndata = number_imp.fit_transform(data[num].values.reshape(-1, 1))
        ndata = number_sca.fit_transform(ndata)
        ndata = ndata.reshape(1, -1)[0]
        feature_data = feature_data.join(
            pd.DataFrame(ndata, index=data.index), rsuffix="_" + num)

    X = feature_data.loc[train_rows].values
    print(f"Calculating values for {target}")
    # TODO: Output should be multilabel (bag of words) and not a label
    if classes[target] == "label":
        Y_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
        activation = "softmax"
    elif classes[target] == "number":
        Y_encoder = MinMaxScaler()
        loss = "mean_squared_error"
        metrics = []
        activation = "linear"
    else:
        raise Exception("Error! Target %s cannot be calculated" % target)

    Y = Y_encoder.fit_transform(
        data.loc[train_rows, target].values.reshape(-1, 1))
    X_features, Y_features = X.shape[1], Y.shape[1]
    neurons = int((X_features + Y_features)/2)
    depth = 2
    patience = 3
    tolerance = 10 ** -3
    epochs = 100
    # Model training
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons,
                                    input_shape=X.shape[1:]))
    for d in range(1, depth):
        model.add(tf.keras.layers.Dense(neurons))

    calls = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                              min_delta=tolerance,
                                              patience=patience,
                                              mode='auto',
                                              restore_best_weights=True)]

    model.add(tf.keras.layers.Dense(Y.shape[1], activation=activation))
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    # train model and predict targets
    print(f"Training Model\nSamples: {len(X)}\nFeatures: {X_features}\t"
          f"Targets: {Y_features}\n{model.summary()}")
    model.fit(X, Y, epochs=epochs, verbose=2, callbacks=calls)
    X_target = feature_data.values
    predictions = Y_encoder.inverse_transform(model.predict(X_target))
    target_data = pd.DataFrame(output.pop(target))
    target_data = target_data.join(pd.DataFrame(predictions,
                                                columns=[target + SUFFIX],
                                                index=data.index))
    output = output.join(target_data)
    print("Process complete\n%s" % output.head())

    return output, classes


def save_file(output_file, results, target, classes, format="xlsx"):
    """
    Save result in a file
    :param output_file: Filename without extension
    :param results: Dataframe to be saved
    :param target: String with target field
    :param classes: Dictionary with fields classification
    :param format: File format: 'xlsx' (default), 'xls', 'csv'
    :return: Filename with extension
    """
    filename = output_file + "_OUTPUT." + format
    if format == "xlsx" or format == "xls":
        styler = results.style.apply(color_targets, target_type=classes[target],
                                         axis=1, subset=pd.IndexSlice[[target, target + SUFFIX]])
        styler.to_excel(filename, sheet_name='Output',
                        index=False, freeze_panes=(1, 0))
    else:
        results.to_csv(filename)
    return filename


def selectTarget(fields,master):
    """
    Graphical selector for target field
    :param fields: List of strings with all possible fields for the target
    :param master: Tkinter master to use for the window
    :return: String with selected target
    """
    master.title("Select the desired target field")
    variable = StringVar(master)
    variable.set(fields[0])  # default value
    OptionMenu(master, variable, *fields).pack()
    Button(master, text="Select", command=master.quit).pack()
    mainloop()
    return variable.get()


def main():
    # parameters
    master = Tk()

    if "-i" in sys.argv:
        input_path = sys.argv[sys.argv.index("-i") + 1]
    else:
        filetypes = (("XLSX files", "*.xlsx"),("XLS files", "*.xls"),
                     ("CSV files", "*.csv"),("all files", "*.*"))
        input_path = askopenfilename(title="Select input file",
                                     filetypes=filetypes,
                                     parent=master)

    filename, file_extension = os.path.splitext(input_path)
    if file_extension in [".csv"]:
        raw_data = pd.read_csv(input_path, low_memory=False, dtype=str)
    elif file_extension in [".xls", ".xlsx"]:
        raw_data = pd.read_excel(input_path, dtype=str)

    if "-t" in sys.argv:
        target = sys.argv[sys.argv.index("-t") + 1]
    else:
        target = selectTarget(tuple(raw_data.columns.values),master)
    master.destroy()
    results, classes = data_calculation(raw_data, target)
    save_file(filename, results, target, classes)


if __name__ == '__main__':
    main()