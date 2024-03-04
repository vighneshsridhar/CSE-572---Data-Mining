import pandas as pd
import numpy as np
import datetime
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
#from scipy.stats import entropy




meals = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date', 'Time', 'BWZ Carb Input (grams)'])
meals['Date and Time'] = pd.to_datetime(meals['Date'] + ' ' + meals['Time'])
meals.dropna(subset = 'BWZ Carb Input (grams)', inplace = True)
meals = meals.loc[~(meals['BWZ Carb Input (grams)'] == 0)]
meals_sorted = meals.sort_values(by = 'BWZ Carb Input (grams)', ascending = True)
meals_sorted.reset_index(drop=True, inplace = True)



CGM_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date', 'Time', 'Sensor Glucose (mg/dL)'])
CGM_data['Date and Time'] = pd.to_datetime(CGM_data['Date'] + ' ' + CGM_data['Time'])
CGM = CGM_data.sort_values(by = 'Date and Time', ascending = True)







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




def get_meal_data(meals_sorted, CGM):
    meal_data = pd.DataFrame()
    for i in range(len(meals_sorted)):
        meal_data = pd.concat([meal_data, is_meal_stretch(meals_sorted, i, CGM)])

    return meal_data



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


meal_data = get_meal_data(meals_sorted, CGM)
meal_features = make_meal_feature_matrix(meal_data)





SSE = []
entropy = []
purity = []




#####################################################################
# Partitions meals into n bins
#####################################################################



def make_n_bins(meals, min_carbs, max_carbs):
    #max_carbs = meals.loc[meals['BWZ Carb Input (grams)'].idxmax()]['BWZ Carb Input (grams)']
    #print (max_carbs)
    n = math.ceil((max_carbs - min_carbs)/20)
    bins = [min_carbs]

    for i in range(n):
        bins.append(min_carbs + 20 * (i + 1))

    meals = meals.copy()
    meals['bins'] = pd.cut(meals['BWZ Carb Input (grams)'], bins)

    return meals

min_carbs = meals.loc[meals['BWZ Carb Input (grams)'].idxmin()]['BWZ Carb Input (grams)']
max_carbs = meals.loc[meals['BWZ Carb Input (grams)'].idxmax()]['BWZ Carb Input (grams)']
#print ("min carbs ", min_carbs)
bins = make_n_bins(meals_sorted, min_carbs, max_carbs)
#print ("max carb ", max_carbs)
#print ("bins ", bins)




def k_means(meal_features, k):
    kmeans = KMeans(n_clusters = k)
    kmeans_df = StandardScaler().fit_transform(meal_features)
    kmeans.fit(kmeans_df)
    SSE.append(kmeans.inertia_)

    return kmeans




#####################################################################
# Computes entropy and purity
#####################################################################


def compute_entropy(meals, clusters, bins):
    total_entropy = 0

    for i in range(clusters.groupby('clusters').ngroups):
        sum_entropy = 0
        for bin in bins['bins'].unique():
            cluster_indices = clusters.index[clusters['clusters'] == i].to_list()
            if not cluster_indices:
                continue
            meals_in_cluster = meals.iloc[cluster_indices]
            meals_in_cluster = make_n_bins(meals_in_cluster, min_carbs, max_carbs)
            #print ("min carbs 2", min_carbs)
            #print ("meal bin ", meals_in_cluster['bins'])
            #print ("bin ", bin)
            meals_in_cluster_bin = meals_in_cluster.loc[meals_in_cluster['bins'] == bin]
            P = len(meals_in_cluster_bin)/len(meals_in_cluster)
            if (P == 0):
                continue
            sum_entropy = sum_entropy - P * math.log2(P)
        total_entropy = total_entropy + len(cluster_indices) * sum_entropy

    total_entropy = total_entropy/len(bins)

    return total_entropy



def compute_purity(meals, clusters, bins):
    total_purity = 0

    for i in range(clusters.groupby('clusters').ngroups):
        sum_purity = 0
        max = 0

        for bin in bins['bins'].unique():
            cluster_indices = clusters.index[clusters['clusters'] == i].to_list()
            if not cluster_indices:
                continue
            meals_in_cluster = meals.iloc[cluster_indices]
            meals_in_cluster = make_n_bins(meals_in_cluster, min_carbs, max_carbs)
            #print ("meal bin ", meals_in_cluster['bins'])
            #print ("bin ", bin)
            meals_in_cluster_bin = meals_in_cluster.loc[meals_in_cluster['bins'] == bin]
            P = len(meals_in_cluster_bin)/len(meals_in_cluster)
            sum_purity = sum_purity + P
            if (max < P):
                max = P
        if (sum_purity == 0):
            continue
        P_cluster = max/sum_purity
        total_purity = total_purity + len(cluster_indices) * P_cluster

    total_purity = total_purity/len(bins)

    return total_purity



def compute_SSE(bin):
    if (len(bin) != 0):
        SSE = 0
        print ("bin ", bin)
        average = sum(bin)/len(bin)
        average = 0

        for point in bin:
            SSE = SSE + math.dist(point, average)**2
        return SSE * len(bin)
    return 0




def compute_SSE_DBSCAN(data, core_sample_indices):
    sum_squared_distance = 0
    length = 0

    for core_index in core_sample_indices:
        core_point = data.iloc[core_index]
        cluster = data.loc[data['clusters'] == core_point['clusters']]
        #print ("core point ", core_point)
        #sum_squared_distance = sum_squared_distance + compute_SSE(cluster)
        length += len(cluster)

        for x in range(len(cluster)):
            point = cluster.iloc[x, :]
            distance = math.dist(point, core_point)**2
            #print ("point ", point)
            #print ("core point ", core_point)
            #print ("distance ", distance)
            sum_squared_distance = sum_squared_distance + distance
    #sum_squared_distance = sum_squared_distance/length
    return sum_squared_distance



def get_DBSCAN_clusters(meal_features, num_clusters):
    clusters_DBSCAN = DBSCAN(eps = 0.03, min_samples = 10).fit(meal_features)
    DBSCAN_df = StandardScaler().fit_transform(meal_features)
    clusters_DBSCAN.fit(DBSCAN_df)
    init_centers = np.asarray(clusters_DBSCAN.components_)
    #bisect_means = BisectingKMeans(init = init_centers).fit(meal_features)
    data_DBSCAN = meal_features.copy()
    #data_DBSCAN['clusters'] = bisect_means.labels_
    data_DBSCAN['clusters'] = clusters_DBSCAN.labels_
    #SSE_DBSCAN = bisect_means.inertia_
    SSE_DBSCAN = compute_SSE_DBSCAN(data_DBSCAN, clusters_DBSCAN.core_sample_indices_)
    SSE.append(SSE_DBSCAN)
    #SSE.append(SSE_DBSCAN)
    return data_DBSCAN



#####################################################################
# Run DBSCAN and k-means clustering and record the metrics
#####################################################################



k = bins['bins'].nunique()
#print ("k = ", k)
#print ("bin types", bins['bins'].dtypes)
kmeans = k_means(meal_features, k)

clusters = meal_features
clusters['clusters'] = kmeans.labels_
entropy_kmeans = compute_entropy(meals_sorted, clusters, bins)
entropy.append(entropy_kmeans)

purity_kmeans = compute_purity(meals_sorted, clusters, bins)
purity.append(purity_kmeans)



data_DBSCAN = get_DBSCAN_clusters(meal_features, k)
#print ("DBSCAN ", data_DBSCAN)


print ("Sum of squared error for k-means and DBSCAN = ", SSE)

entropy_DBSCAN = compute_entropy(meals_sorted, data_DBSCAN, bins)
entropy.append(entropy_DBSCAN)
print ("Entropy for k-means and DBSCAN = ", entropy)

purity_DBSCAN = compute_purity(meals_sorted, data_DBSCAN, bins)
purity.append(purity_DBSCAN)
print("Purity for k means and DBSCAN = ", purity)

SSE_df = pd.DataFrame(SSE)
entropy_df = pd.DataFrame(entropy)
purity_df = pd.DataFrame(purity)
SSE_df.reset_index(drop = True, inplace = True)
entropy_df.reset_index(drop = True, inplace = True)
purity_df.reset_index(drop = True, inplace = True)
SSE_df.to_csv('SSE.csv')

SSE_df_T = SSE_df.transpose()
entropy_df_T = entropy_df.transpose()
purity_df_T = purity_df.transpose()

results_metrics = SSE_df_T
results_metrics = pd.concat([results_metrics, entropy_df_T], axis = 1)
results_metrics = pd.concat([results_metrics, purity_df_T], axis = 1)
results_metrics = results_metrics.reset_index(drop = True)
#print ("Results ", results_metrics)
results_metrics.to_csv('Result.csv', index = False, header = False)
