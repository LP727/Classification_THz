#misc.py

import os
import pandas as pd

def list_files(path: str, complete_path: bool = False,
               hidden_files: bool = False):

    if hidden_files:
        def isfile_condition(p, f): return os.path.isfile(os.path.join(p, f))
    else:
        def isfile_condition(
            p, f): return os.path.isfile(
            os.path.join(
                p, f)) and not f.startswith('.')

    if complete_path:
        def result_format(p, f): return os.path.join(p, f)
    else:
        def result_format(p, f): return f

    file_list = [
        result_format(
            path,
            f) for f in os.listdir(path) if isfile_condition(
            path,
            f)]
    return sorted(file_list)

def gen_dataframe(images: list, labels: list):
    # dictionary of lists
    dict = {'filename': images, 'label': labels}

    df = pd.DataFrame(dict)
    print(df.iloc[0])
    return df