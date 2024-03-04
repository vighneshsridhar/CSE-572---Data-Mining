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


meals = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date', 'Time', 'BWZ Carb Input (grams)'])
CGM_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date', 'Time', 'Sensor Glucose (mg/dL)'])
meals['Date and Time'] = pd.to_datetime(meals['Date'] + ' ' + meals['Time'])
CGM_data['Date and Time'] = pd.to_datetime(CGM_data['Date'] + ' ' + CGM_data['Time'])
CGM = CGM_data.sort_values(by = 'Date and Time', ascending = True)
meals.dropna(subset = 'BWZ Carb Input (grams)', inplace = True)
meals = meals.loc[~(meals['BWZ Carb Input (grams)'] == 0)]
meals_sorted = meals.sort_values(by = 'Date and Time', ascending = True)
meals_sorted.reset_index(drop=True, inplace = True)



meals_patient = pd.read_csv('Insulin_patient2.csv', low_memory = False, usecols = ['Date', 'Time', 'BWZ Carb Input (grams)'])
CGM_patient = pd.read_csv('CGM_patient2.csv', low_memory = False, usecols = ['Date', 'Time', 'Sensor Glucose (mg/dL)'])
meals_patient['Date and Time'] = pd.to_datetime(meals_patient['Date'] + ' ' + meals_patient['Time'])
CGM_patient['Date and Time'] = pd.to_datetime(CGM_patient['Date'] + ' ' + CGM_patient['Time'])
meals_patient.dropna(subset = 'BWZ Carb Input (grams)', inplace = True)
meals_patient = meals_patient.loc[~(meals_patient['BWZ Carb Input (grams)'] == 0)]
meals_patient_sorted = meals_patient.sort_values(by = 'Date and Time', ascending = True)
meals_patient_sorted.reset_index(drop=True, inplace = True)



meals_sorted.to_csv('meal_sorted.csv')
#quit()



def is_meal_stretch(meals_sorted, i, CGM):
    meal_data = pd.DataFrame()
    start_meal_str = meals_sorted['Time'][i]
    start_meal = pd.to_datetime(meals_sorted['Date and Time'][i])
    date = meals_sorted['Date'][i]
    if (i != len(meals_sorted) - 1):
        next_meal = pd.to_datetime(meals_sorted['Date and Time'][i + 1])
    else:
        next_meal = start_meal + pd.Timedelta(hours = 3)
    if (next_meal >= start_meal + pd.Timedelta(hours = 2)):
        #print ("i = ", i)
        #print ("date ", date)
        #print("CGM date ", CGM['Date'])
        potential_time = start_meal
        meal_data_CGM = CGM.loc[(pd.to_datetime(CGM['Date and Time']) >= potential_time - pd.Timedelta(minutes = 30))
        & (pd.to_datetime(CGM['Date and Time']) <= potential_time + pd.Timedelta(hours = 2))]['Sensor Glucose (mg/dL)']
        meal_data_CGM = meal_data_CGM.reset_index()
        meal_data_T = meal_data_CGM.transpose()
        meal_data_T = meal_data_T.iloc[1:]
        meal_data = pd.concat([meal_data, meal_data_T])
        #print("shape ", meal_data.shape)
        return meal_data

def is_not_meal_stretch(meals_sorted, i, CGM):
    no_meal_data = pd.DataFrame()
    time_iterator = pd.to_timedelta(meals_sorted['Time'][i]) + pd.Timedelta(hours = 2)
    time_iterator_str = str(time_iterator)
    time_iterator_str = time_iterator_str.split('days ')[-1]
    date = meals_sorted['Date'][i]
    if (i != len(meals_sorted) - 1):
        date_time_next_meal = pd.to_datetime(meals_sorted['Date and Time'][i + 1])
        date_next_meal = meals_sorted['Date'][i + 1]
    else:
        date_next_meal = pd.to_datetime(meals_sorted['Date'][i])
        date_time_next_meal = pd.to_datetime(meals_sorted['Date and Time'][i]) + pd.Timedelta(hours = 10)
    j = 0
    curr_date = date
    curr_date = pd.to_datetime(curr_date)
    potential_time = pd.to_datetime(date + ' ' + time_iterator_str)
    #print ("potential_time ", potential_time)
    while (curr_date != pd.to_datetime(date_next_meal)):
        no_meal_CGM = CGM.loc[(pd.to_datetime(CGM['Date and Time']) >= potential_time)
        & (pd.to_datetime(CGM['Date and Time']) <= potential_time + pd.Timedelta(hours = 2))]['Sensor Glucose (mg/dL)']
        no_meal_CGM = no_meal_CGM.reset_index()
        no_meal_CGM_T = no_meal_CGM.transpose()
        no_meal_CGM_T = no_meal_CGM_T.iloc[1:]
        #print ("no meal shape ", no_meal_CGM_T.shape)
        #row_count = no_meal_CGM_T.shape[0]
        #column_count = no_meal_CGM_T.shape[1]
        #print ("no meal row transpose ", row_count, "no meal column count ", column_count)
        no_meal_data = pd.concat([no_meal_data, no_meal_CGM_T])
        if (time_iterator + datetime.timedelta(hours = 2) > datetime.timedelta(hours = 22)):
            potential_time = potential_time + datetime.timedelta(days = 1)
            curr_date = curr_date + datetime.timedelta(days = 1)
            potential_time.replace(hour = 0)
            time_iterator = pd.Timedelta(hours = 0)
        else:
            potential_time = potential_time + pd.Timedelta(hours = 2)
            time_iterator = time_iterator + pd.Timedelta(hours = 2)
        j += 1
    while ((pd.to_datetime(date_time_next_meal) >= potential_time + pd.Timedelta(hours = 2)) & (time_iterator < datetime.timedelta(hours = 22))):
        no_meal_CGM = CGM.loc[(pd.to_datetime(CGM['Date and Time']) >= potential_time)
        & (pd.to_datetime(CGM['Date and Time']) <= potential_time + pd.Timedelta(hours = 2))]['Sensor Glucose (mg/dL)']
        no_meal_CGM = no_meal_CGM.reset_index()
        no_meal_CGM_T = no_meal_CGM.transpose()
        no_meal_CGM_T = no_meal_CGM_T.iloc[1:]
        no_meal_data = pd.concat([no_meal_data, no_meal_CGM_T])
        #print (no_meal_data)
        potential_time = potential_time + pd.Timedelta(hours = 2)
        time_iterator = time_iterator + pd.Timedelta(hours = 2)
        j += 1
    return no_meal_data



#####################################################################
# Make the meal and no meal feature matrices
#####################################################################



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

    for i in range(0, no_meal_data.shape[0]):
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



#####################################################################
# Extract meal and no meal data from using the Insulin and CGM data.
#####################################################################



no_meal_data = pd.DataFrame()
meal_data = pd.DataFrame()
for i in range(len(meals_sorted)):
    meal_data = pd.concat([meal_data, is_meal_stretch(meals_sorted, i, CGM)])
    no_meal_data = pd.concat([no_meal_data, is_not_meal_stretch(meals_sorted, i, CGM)])


no_meal_data_patient = pd.DataFrame()
meal_data_patient = pd.DataFrame()
for j in range(len(meals_patient_sorted)):
    meal_data_patient = pd.concat([meal_data_patient, is_meal_stretch(meals_patient_sorted, j, CGM_patient)])
    no_meal_data_patient = pd.concat([no_meal_data_patient, is_not_meal_stretch(meals_patient_sorted, j, CGM_patient)])




no_meal_data = no_meal_data.reset_index(drop = True)
# print("shape of no meal data, ", no_meal_data.shape)
no_meal_data.to_csv('no_meal_results.csv', index = False)
no_meal_feature_matrix = make_no_meal_feature_matrix(no_meal_data)
no_meal_feature_matrix_patient = make_no_meal_feature_matrix(no_meal_data_patient)
#no_meal_feature_matrix.to_csv('no_meal_feature_matrix.csv')
no_meal_feature_matrix_patient.to_csv('no_meal_feature_matrix_patient.csv')
no_meal_feature_matrix = pd.concat([no_meal_feature_matrix, no_meal_feature_matrix_patient])



meal_data = meal_data.iloc[:, 0:24]
meal_data = meal_data.reset_index(drop = True)
meal_data.to_csv('meal_results.csv', index = False)
meal_feature_matrix = make_meal_feature_matrix(meal_data)
meal_feature_matrix_patient = make_meal_feature_matrix(meal_data_patient)
#meal_feature_matrix.to_csv('meal_feature_matrix.csv')
meal_feature_matrix_patient.to_csv('meal_feature_matrix_patient.csv')
meal_feature_matrix = pd.concat([meal_feature_matrix, meal_feature_matrix_patient])


#####################################################################
# Train the machine to classify meal data using a decision tree
#####################################################################



meal_feature_matrix['label'] = 1
no_meal_feature_matrix['label'] = 0
feature_data = pd.concat([meal_feature_matrix, no_meal_feature_matrix])
#feature_data.to_csv('data_concat.csv')
feature_data = feature_data.sample(frac = 1)
feature_data = feature_data.reset_index(drop = True)
unlabeled_data = feature_data.drop(columns = 'label')
list_of_classifiers = ['DecisionTree', 'XGBClassifier']
#model = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
#model2 = DecisionTreeClassifier(max_depth = 5)
model = AdaBoostClassifier()
scores = []

kf = KFold(n_splits = 10, shuffle = True)
i = 0
for train_index, test_index in kf.split(unlabeled_data):
    X_train = unlabeled_data.loc[train_index]
    X_test = unlabeled_data.loc[test_index]
    Y_train = feature_data.label.loc[train_index]
    Y_test = feature_data.label.loc[test_index]
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    scores.append(score)

#classifier2 = DecisionTreeClassifier(max_depth = 5)
#classifier =  SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
classifier = AdaBoostClassifier()
X, Y = unlabeled_data, feature_data['label']
classifier.fit(X, Y)
pickle.dump(classifier, open('Random.pickle', 'wb'))
