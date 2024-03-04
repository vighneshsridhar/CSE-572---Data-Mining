import pandas as pd
import numpy as np
from numpy.fft import fft, ifft
import datetime
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from scipy.fftpack import fft, ifft, rfft
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import KFold
from sklearn import metrics
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle

meal_data = pd.read_csv('test.csv', header = None)

def make_meal_feature_matrix(meal_data):
    meal_feature_matrix = pd.DataFrame()
    power_first_fourier = []
    power_second_fourier = []

    first_differential = []
    second_differential = []
    standard_deviation = []

    for i in range(meal_data.shape[0]):
        temp1 = meal_data.iloc[:, 0:24].iloc[i]
        temp_array = temp1.to_numpy()
        #temp2 = temp1.iloc[i]
        temp3 = temp1.values.tolist()
        temp_array = temp_array.tolist()
        #print ("i = ", i, "temp array ", temp_array)
        int_time = pd.to_datetime(temp_array).astype('int64')/10**9
        #fourier = np.fft.fft2(int_time, axes = (-1,))
        fourier = abs(np.fft.rfft(int_time)).tolist()
        fourier_sorted = np.sort(fourier)
        #print ("fourier shape ", fourier_sorted.shape)
        power_first_fourier.append(fourier_sorted[-2])
        power_second_fourier.append(fourier_sorted[-3])

    for i in range(meal_data.shape[0]):
        int_time_diff = pd.to_datetime(meal_data.iloc[:, 0:24].iloc[i]).astype('int64')/10**9
        first_differential.append(np.diff(int_time_diff).max())
        second_differential.append(np.diff(np.diff(int_time_diff)).max())
        int_time = pd.to_datetime(meal_data.iloc[i]).astype('int64')/10**9
        standard_deviation.append(np.std(int_time))
    meal_feature_matrix['first_differential'] = first_differential
    meal_feature_matrix['second_differential'] = second_differential
    meal_feature_matrix['standard_deviation'] = standard_deviation
    meal_feature_matrix['fourier_first_power'] = power_first_fourier
    meal_feature_matrix['fourier_second_power'] = power_second_fourier
    return meal_feature_matrix

def make_no_meal_feature_matrix(no_meal_data):
    no_meal_feature_matrix = pd.DataFrame()
    power_first_fourier = []
    power_second_fourier = []
    first_differential = []
    second_differential = []
    standard_deviation = []
    for i in range(no_meal_data.shape[0]):
        temp1 = no_meal_data.iloc[:, 0:24].iloc[i]
        temp_array = temp1.to_numpy()
        #temp2 = temp1.iloc[i]
        temp3 = temp1.values.tolist()
        #temp_array = temp_3.tolist()
        #print ("i = ", i, "temp array ", temp_array)
        int_time = pd.to_datetime(temp_array).astype('int64')/10**9
        #fourier = np.fft.fft2(int_time, axes = (-1,))
        fourier = abs(np.fft.rfft(int_time)).tolist()
        fourier_sorted = np.sort(fourier)
        #print ("fourier shape ", fourier_sorted.shape)
        power_first_fourier.append(fourier_sorted[-2])
        power_second_fourier.append(fourier_sorted[-3])
    for i in range(len(no_meal_data)):
        int_time_diff = pd.to_datetime(no_meal_data.iloc[:, 0:24].iloc[i]).astype('int64')/10**9
        first_differential.append(np.diff(int_time_diff).max())
        second_differential.append(np.diff(np.diff(int_time_diff)).max())
        int_time = pd.to_datetime(no_meal_data.iloc[i]).astype('int64')/10**9
        standard_deviation.append(np.std(int_time))
    no_meal_feature_matrix['first_differential'] = first_differential
    no_meal_feature_matrix['second_differential'] = second_differential
    no_meal_feature_matrix['standard_deviation'] = standard_deviation
    no_meal_feature_matrix['fourier_first_power'] = power_first_fourier
    no_meal_feature_matrix['fourier_second_power'] = power_second_fourier
    return no_meal_feature_matrix

data = make_no_meal_feature_matrix(meal_data)
#data_2 = make_meal_feature_matrix(meal_data)
#data = pd.concat([data_1, data_2])
data = data.reset_index(drop = True)
# print ("data ", data)

with open('Random.pickle', 'rb') as r:
    pickle_model = pickle.load(r)
predictions = pickle_model.predict(data)

pd.DataFrame(predictions).to_csv('Result.csv', index = False, header = False)
