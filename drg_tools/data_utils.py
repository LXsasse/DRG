# data_utils.py
# functions and classes to read, save, and process data

# Author: Alexander Sasse <alexander.sasse@gmail.com>

import numpy as np


def read_txt_files(filename, delimiter = ',', header = '#', strip_names = '"', column_name_replace = None, row_name_replace = None, unknown_value = 'NA', nan_value = 0):
    
    f = open(filename, 'r').readlines()
    columns, rows, values = None, [], []
    if header is not None:
        if f[0][:len(header)] == header:
            columns = line.strip(header).strip().replace(strip_names,'').split(delimiter)
    
    start = 0
    if columns is not None:
        start = 1
    
    for l, line in enumerate(f):
        if l >= start:
            line = line.strip().split(delimiter)
            rows.append(line[0].strip(strip_names))
            ival = np.array(line[1:])
            ival[ival == unkown] = 'nan'
            values.append(ival)
    
    if column_name_replace is not None:
        for c, col in enumerate(columns):
            columns[c] = col.replace(column_name_replace[0], row_name_replace[1])
    
    if row_name_replace is not None:
        for r, row in enumerate(rows):
            rows[r] = row.replace(row_name_replace[0], row_name_replace[1])
            
    values = np.array(values, dtype = float)
    if (values == np.nan).any():
        print('ATTENTION nan values in data matrix.', filename)
        if nan_value not None:
            print('nan values replaced with', nan_value)
            values = np.nan_to_num(values, nan = nan_value)
    return np.array(rows), np.array(columns), np.array(values, dtype = float)


