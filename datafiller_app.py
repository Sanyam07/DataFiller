# TODO: Recognize Model
# TODO: Recognize Colors
# TODO: Add example data and script

"""
    Tool that completes and validates a dataset field.
    It takes an input file with values missing from one column (target)
    and completes them with a Neural Network trained on existing examples.

    @author: Francesco Baldisserri
    @email: fbaldisserri@gmail.com
    @version: 0.9
"""

import os
import sys
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename

from datafiller import DataFiller

FILE_EXTENSIONS = ["xlsx", "xls", "csv"]


def main():
    input_file = get_input_file()
    input_data = read_input_file(input_file)
    target = get_target_name(input_data)
    print(f"Calculating {target}")
    filler = DataFiller(input_data, target)
    predictions = filler.predict_target()
    output_filename = os.path.splitext(input_file)[0]
    filler.save_dataset(f"{output_filename}_OUTPUT_{target}.xlsx", predictions)
    print("Process complete\n%s" % predictions.head())


def get_input_file():
    """ Get source filename from arguments or through graphical selector """
    master = tk.Tk()
    input_path = get_argument("-i") if has_argument("-i") \
        else select_file(master, FILE_EXTENSIONS)
    master.destroy()
    return input_path


def read_input_file(input_file):
    """ Read input file from CSV, XLS or XLSX and return Dataframe """
    filename, file_extension = os.path.splitext(input_file)
    if file_extension in [".csv"]:
        input_data = pd.read_csv(input_file, dtype=str)
    elif file_extension in [".xls", ".xlsx"]:
        input_data = pd.read_excel(input_file, dtype=str)
    else:
        raise ValueError(f"File extension {file_extension} not recognized")
    return input_data


def get_target_name(input_data):
    """
    Read parameters and input data

    :param input_data: Dataframe with input data
    :return: Dataset, Target columns and filename
    """
    master = tk.Tk()
    target = get_argument("-t") if has_argument("-t") \
        else select_target(tuple(input_data.columns.values), master)
    master.destroy()
    return target


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


def has_argument(argument):
    return argument in sys.argv


def get_argument(argument):
    return sys.argv[sys.argv.index(argument) + 1]


def select_file(master, extensions):
    file_types = [(e.upper() + " files", "*." + e) for e in extensions]
    return askopenfilename(title="Select input file",
                           filetypes=file_types,
                           parent=master)


if __name__ == '__main__':
    main()
