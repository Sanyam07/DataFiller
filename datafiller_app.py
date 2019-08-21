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

EXTENSIONS = ["xlsx", "xls", "csv"]
SPARSE_MATRIX = True


def main():
    input_file = get_input_file()
    filename, file_extension = os.path.splitext(input_file)
    if file_extension in [".csv"]:
        input_data = pd.read_csv(input_file, low_memory=SPARSE_MATRIX, dtype=str)
    elif file_extension in [".xls", ".xlsx"]:
        input_data = pd.read_excel(input_file, dtype=str)
    else:
        raise ValueError(f"File extension {file_extension} not recognized")
    target = get_target_name(input_data)

    print(f"Calculating {target}")
    filler = DataFiller(input_data, target, sparse_matrix=SPARSE_MATRIX)
    filler.predict_target()
    filler.save_dataset(f"{filename}_OUTPUT_{filler.target}.xlsx")
    print("Process complete\n%s" % filler.target_pred.head())


def get_input_file():
    """
    Get source filename from arguments or through graphical selector

    :return: Input filename
    """
    master= tk.Tk()
    if "-i" in sys.argv:
        input_path = sys.argv[sys.argv.index("-i") + 1]
    else:
        filetypes = [(e.upper() + " files", "*." + e) for e in EXTENSIONS]
        input_path = askopenfilename(title="Select input file",
                                     filetypes=filetypes, parent=master)
    master.destroy()
    return input_path


def get_target_name(input_data):
    """
    Read parameters and input data

    :param input_data: Dataframe with input data
    :return: Dataset, Target columns and filename
    """
    master = tk.Tk()
    if "-t" in sys.argv:
        target = sys.argv[sys.argv.index("-t") + 1]
    else:
        target = select_target(tuple(input_data.columns.values), master)
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


if __name__ == '__main__':
    main()
