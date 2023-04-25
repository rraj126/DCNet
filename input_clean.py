import numpy as np

def set_data_range(data: np.ndarray):
    '''Function to set data range between 0 and 1'''
    min_zero_data = data - np.min(data, axis = 0)
    data_p = min_zero_data / np.max(min_zero_data, axis = 0)
    return data_p

def remove_zero_cols(data: np.ndarray):
    '''Function to remove all columns with all zeros'''
    good_index = np.sum(data, axis = 0) != 0
    data_p = data[:, good_index]
    return data_p

def remove_nans(data: np.ndarray):
    '''Function to remove nans in-place'''
    data[np.isnan(data)] = 0

def clean_data(labeled_data_dict: dict):
    '''Function to clean data in a dictionary'''
    
    # loop through data for each label
    for label, data in labeled_data_dict.items():
        remove_nans(data)
        labeled_data_dict[label] = remove_zero_cols(data)
        labeled_data_dict[label] = set_data_range(labeled_data_dict[label])
        remove_nans(labeled_data_dict[label])
