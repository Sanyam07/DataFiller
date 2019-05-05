# TODO: Test with ADwords
# TODO: APP FRIENDLY RESTRUCTURING
# TODO: FEATURE FREEZE - GIT CODE REVIEW
# TODO: Recognize dates
# TODO: Multi-target processed at the same time
# TODO: READ EXCEL NUMBER FORMAT
# TODO: USE SPARSE MATRIX

"""
    Tool that completes and validates CSV, XLS and XLSX datasets.
    It takes files with values missing from one or more columns (targets)
    and completes them with a Neural Network Algorithm using existing examples.
"""

import os
import sys
import numpy as np
import pandas as pd
import copy
import tensorflow as tf
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TGT_SUFFIX = '_NEW'
version = '19'


def color_predictions(row, tolerance=0.1, new='#98FF99', different='#FF9999', match='transparent'):
    """
        Dataframe style function for coloring dataset calculated targets values depending on the old_value values.

        :param row: Dataset row, to be used with DATAFRAME.style.apply(color_predictions, axis=1)
        :param tolerance: % tolerance to consider old_value and calculated values a match for numeric fields
        :param new: color for calculated values missing in the old_value targets column (default is green)
        :param different: color for calculated values different from the old_value targets column (default is red)
        :param match: color for calculated values matching the old_value targets column (default is transparent)
        :return: list with background-color specification for the dataframe row
    """
    old_value, new_value = row[0], row[1]
    if old_value is np.NAN:
        color = new
    else:  # TODO: Format numbers as integers, floats, ...
        try: # Trying to convert the old_value value to a number
            old_value, new_value = np.float(old_value), np.float(new_value)  # TODO: np.float affecting performance?
            if abs(old_value-new_value)<tolerance*abs(old_value + new_value)/2:
                color = match
            else:
                color = different
        except: # In case old_value value is not a number only equality is considered a match
            if old_value == new_value:
                color = match
            else:
                color = different
    return ['','background-color: {}'.format(color)]

def classifyColumns(data, verbose=False):
    """
        Classify the dataset columns by type:\n
        - "label": any string or even number that is used as a category (e.g. number_of_car_doors = {2,4})\n
        - "number": any discrete or continuous quantity\n
        - "ignore": columns that should not be considered in the calculations (e.g. unique record IDs)

        :param data: Dataframe with dataset data to be analyzed
        :param verbose: Boolean to cursor verbose mode
        :return: Dictionary with all Dataframe columns and their string classification: "label","number","ignore"
    """
    classes = {}
    integers, floats = ['int16', 'int32', 'int64'], ['float16', 'float32', 'float64']
    columns = list(data.columns)
    for column in columns:
        df = copy.deepcopy(data[column]).dropna()
        df = df.apply(pd.to_numeric, errors='ignore')
        # ignore columns with only one possible value
        #if len(df.unique()) == 1 or len(df.unique()) == len(df):
        if len(df.unique()) <= 1:
            classes[column]= "ignore"
        # classify as "label" all string fields and numeric fields with
        # total unique values less than 0.1% of total cursor
        # TODO: EAN and Customs Code in Full Export are mislabeled as numbers
        elif df.dtype in floats or df.dtype in integers and len(df.unique()) > len(df) / 100:
            classes[column] = "number"
        else:
            classes[column] = "label"
    if verbose:
        print("Classification results:\n%s" % classes)
    return classes

def identifyTargets(data, verbose=False):
    """
        Identify fields with gaps to be filled by checking all columns
        with empty cells (np.NAN).

        :param data: Dataframe with dataset data to be analyzed
        :param verbose: Boolean to cursor verbose mode
        :return: List with all column names considered targets columns
    """
    columns = list(data.columns)
    targets = [c for c in columns if any(data[c].isna())]
    if verbose:
        print("Targets identified:\n%s" % targets)
    return targets

def dataCalculation(raw_data,targets=None,classes=None,verbose=False):
    # Loop for all targets
    if targets == None:
        targets=identifyTargets(raw_data, verbose)
    if classes == None:
        classes=classifyColumns(raw_data,verbose=verbose)
    results = raw_data.drop(columns=targets)
    for target in targets:
        if classes[target] == 'ignore':
            print("Error! Target %s contains a single value and it cannot be calculated"
                  % target)
            break
        if verbose:
            print("Calculating values for %s" % target)
        # identify features, targets, training and test rows
        features = list(raw_data.columns)
        features.remove(target)
        target_rows = list(raw_data[raw_data[target].isna()].index)
        train_rows = list(set(raw_data.index) - set(target_rows))

        # normalize raw data
        label_features = [f for f in features if classes[f] == "label"]
        number_features = [f for f in features if classes[f] == "number"]
        feature_data = pd.DataFrame(index=raw_data.index)
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
        # TODO: Word vector instead of TFID
        encoder = TfidfVectorizer(token_pattern="\w", strip_accents="unicode",
                                  decode_error="replace")
        #encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)

        for f in label_features:
            if verbose:
                print("Encoding %s" % f)
            ldata = imputer.fit_transform(raw_data[f].values.reshape(-1, 1))
            ldata = encoder.fit_transform([str(l) for l in ldata]).toarray()
            feature_data = feature_data.join(pd.DataFrame(ldata, index=raw_data.index), rsuffix="_" + f)

        imputer = SimpleImputer(strategy='median')
        mms = MinMaxScaler()
        for f in number_features:
            if verbose:
                print("Encoding %s" % f)
            ndata = imputer.fit_transform(raw_data[f].values.reshape(-1, 1))
            ndata = mms.fit_transform(ndata)
            ndata = ndata.reshape(1, -1)[0]
            feature_data = feature_data.join(pd.DataFrame(ndata, index=raw_data.index), rsuffix="_" + f)

        X = feature_data.loc[train_rows].values
        if verbose:
            print("Calculating values for %s" % target)
        #if classes[target] == "phrase": # TODO: Output should be multilabel (bag of words) and not a label
            #Y_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            #Y_encoder = TfidfVectorizer(strip_accents="unicode", decode_error="replace")
            #type = "regressor"
        if classes[target] == "label":
            Y_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
            activation = "softmax"
        else:
            Y_encoder = MinMaxScaler()
            loss = "mean_squared_error"
            metrics = []
            activation = "linear"

        Y = Y_encoder.fit_transform(raw_data.loc[train_rows, target].values.reshape(-1, 1))
        #Y = Y_encoder.fit_transform(raw_data.loc[train_rows, target].values).toarray()  # Phrase setup
        X_features, Y_features = X.shape[1], Y.shape[1]
        neurons = int((X_features + Y_features))
        depth = 2
        patience = 10
        tolerance = 10**-5
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
        model.fit(X, Y, epochs=epochs, verbose=2, callbacks=calls)
        X_target = feature_data.values
        predictions = Y_encoder.inverse_transform(model.predict(X_target))
        target_data = pd.DataFrame(raw_data[target])
        target_data = target_data.join(pd.DataFrame(predictions, columns=[target + TGT_SUFFIX], index=raw_data.index))
        results = results.join(target_data)

    if verbose:
        print("Process complete\n%s" % results.head())

    return results

def saveFile(output_file, results, targets, format="xlsx",):
    filename = output_file + "_OUTPUT." + format
    if format == "xlsx":
        for t in targets:
            # TODO: Check styler application to all targets and not just last one
            styler = results.style.apply(color_predictions, axis=1, subset=pd.IndexSlice[[t, t + TGT_SUFFIX]])
        styler.to_excel(filename, sheet_name='Output',
                        index=False, freeze_panes=(1, 0))
    else:
        results.to_csv(filename)
    return filename

def main():
    # parameters
    verbose = True
    if "-i" in sys.argv:
        input_path = sys.argv[sys.argv.index("-i")+1]
    else:
        input_path = askopenfilename()

    filename, file_extension = os.path.splitext(input_path)
    if file_extension in [".csv"]:
        raw_data = pd.read_csv(input_path, low_memory=False, dtype=str)
    elif file_extension in [".xls",".xlsx"]:
        raw_data = pd.read_excel(input_path, dtype=str)

    # Checks if classes are specified in the input file (1st row after header)
    classes = classifyColumns(raw_data, verbose)

    # Checks if targets are specified in the command line
    if "-t" in sys.argv:
        targets = sys.argv[sys.argv.index("-t")+1].split(",")
    else:
        targets = None # TODO: Graphic dialog with target selector
    results = dataCalculation(raw_data,targets,classes,verbose=verbose)
    # write back csv
    saveFile(filename, results,targets)

if __name__ == '__main__':
    main()