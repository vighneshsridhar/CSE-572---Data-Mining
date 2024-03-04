import pandas
import numpy
from datetime import time

df = pandas.read_csv('CGMData.csv', low_memory = False)
insulin_data = pandas.read_csv('InsulinData.csv', low_memory = False)

def count_CGM_in_range_strict_inequality(CGM, lower_bound, upper_bound):
    # print(lower_bound)
    # print(upper_bound)
    temp = CGM.copy()
    temp1 = CGM.copy()
    temp1.loc[(temp['Sensor Glucose (mg/dL)'] >= lower_bound) & (temp['Sensor Glucose (mg/dL)'] <= upper_bound)] = 1
    temp1.loc[(temp['Sensor Glucose (mg/dL)'] < lower_bound) | (temp['Sensor Glucose (mg/dL)'] > upper_bound)] = 0
    # print(temp1['Sensor Glucose (mg/dL)'])
    # num_between = temp1.groupby(['Date'])['Sensor Glucose (mg/dL)'].sum()
    num_between = temp1['Sensor Glucose (mg/dL)'].sum()
    #print("num between")
    #print (num_between)
    #print("temp date")
    #print(temp['Date'])
    dates = temp.groupby('Date')
    total_rows = dates.ngroups
    metric = num_between/total_rows
    #print("total rows")
    #print(total_rows)
    #print ("metric")
    #print(metric)
    return metric

def count_CGM_in_range(CGM, lower_bound, upper_bound):
    temp = CGM.copy()
    temp1 = CGM.copy()
    temp1.loc[(temp['Sensor Glucose (mg/dL)'] > lower_bound) & (temp['Sensor Glucose (mg/dL)'] < upper_bound)] = 1
    temp1.loc[(temp['Sensor Glucose (mg/dL)'] <= lower_bound) | (temp['Sensor Glucose (mg/dL)'] >= upper_bound)] = 0
    # print(temp1['Sensor Glucose (mg/dL)'])
    # num_between = temp1.groupby(['Date'])['Sensor Glucose (mg/dL)'].sum()
    num_between = temp1['Sensor Glucose (mg/dL)'].sum()
    #print("num between")
    #print (num_between)
    #print("temp date")
    #print(temp['Date'])
    dates = temp.groupby('Date')
    total_rows = dates.ngroups
    metric = num_between/total_rows
    #print("total rows")
    #print(total_rows)
    #print ("metric")
    #print(metric)
    return metric

df['Date and Time'] = pandas.to_datetime(df['Date'] + ' ' + df['Time'])
#date_to_remove = df[df['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()
#df = df.set_index('Date').drop(index = date_to_remove).reset_index()
insulin_data['Date and Time'] = pandas.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
insulin_data_sorted = insulin_data.sort_values(by = 'Date and Time', ascending = True)
# insulin_data_sorted.to_csv('insulin_data_sorted.csv')
#start_auto_mode = insulin_data.sort_values(by = 'Date and Time', ascending = True).loc[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]['Date and Time']
start_auto_mode = insulin_data_sorted.loc[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]['Date and Time']
#print(insulin_data.sort_values(by = 'Date and Time', ascending = True).loc[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0])
#print (start_auto_mode)
#df_auto_mode = df.sort_values(by = 'Date and Time', ascending = True).loc[df['Date and Time'] >= start_auto_mode]
##df_manual_mode = df.sort_values(by = 'Date and Time', ascending = True).loc[df['Date and Time'] < start_auto_mode]
#df_auto_mode.to_csv('df_auto.csv')
#df_manual_mode = df.loc[df['Date and Time'] < start_auto_mode]
#df_manual_mode.to_csv('df_manual.csv')
#quit()

##################################
## Computes metrics in manual mode
##################################

header = ["Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)",	"Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
"Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",	"Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
"Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",	"Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
"Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)", "Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
"Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)", "Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
"Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",	"Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
"Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)",	"Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
"Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)", "Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
"Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)", "Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"]

df_manual_mode = df.loc[df['Date and Time'] < start_auto_mode].copy()
#daytime_CGM_manual_mode = df_manual_mode[(df_manual_mode['Time'] >= '6:00:00') & (df_manual_mode['Time'] <= '23:59:59')].copy()
#overnight_CGM_manual_mode = df_manual_mode[(df_manual_mode['Time'] >= '00:00:00') & (df_manual_mode['Time'] <= '5:59:59')].copy()
#whole_day_CGM_manual_mode = df_manual_mode[(df_manual_mode['Time'] >= '00:00:00') & (df_manual_mode['Time'] <= '23:59:59')].copy()
#daytime_CGM_manual_mode = df_manual_mode.query("`Time` >= '06:00:00' and `Time` <= '23:59:59'")
#overnight_CGM_manual_mode = df_manual_mode.query("`Time` >= '00:00:00' and `Time` <= '06:00:00'")
#whole_day_CGM_manual_mode = df_manual_mode.query("`Time` >= '00:00:00' and `Time` <= '23:59:59'")
df_manual_mode['Time'] = pandas.to_datetime(df_manual_mode['Time']).dt.time
daytime_CGM_manual_mode = df_manual_mode[df_manual_mode['Time'].between(time(6, 0), time(23, 59))].copy()
#print("daytime")
#print(daytime_CGM_manual_mode)
overnight_CGM_manual_mode = df_manual_mode[df_manual_mode['Time'].between(time(0, 0), time(6, 0))].copy()

#print("overnight")
#print(overnight_CGM_manual_mode)
whole_day_CGM_manual_mode = df_manual_mode[df_manual_mode['Time'].between(time(0, 0), time(23, 59))].copy()

#print("whole day")
#print(whole_day_CGM_manual_mode)
#quit()

overnight_CGM_manual_mode['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_manual_mode, 180, 1000)/288)
#print(overnight_CGM)
overnight_CGM_manual_mode['Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_manual_mode, 250, 1000)/288).mean()
overnight_CGM_manual_mode['Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(overnight_CGM_manual_mode, 70, 180)/288).mean()
overnight_CGM_manual_mode['Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(overnight_CGM_manual_mode, 70, 150)/288).mean()
overnight_CGM_manual_mode['Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_manual_mode, 0, 70)/288).mean()
overnight_CGM_manual_mode['Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_manual_mode, 0, 54)/288).mean()
overnight_CGM_manual_mode.dropna()
overnight_CGM_manual_mode = overnight_CGM_manual_mode.reset_index(drop=True)
#overnight_CGM_manual_mode.to_csv('overnight_CGM_manual_mode.csv', index = True, header = False)
#print(overnight_CGM_manual_mode.iloc[0,:])

daytime_CGM_manual_mode['Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_manual_mode, 180, 1000)/288).mean()
daytime_CGM_manual_mode['Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_manual_mode, 250, 1000)/288).mean()
daytime_CGM_manual_mode['Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(daytime_CGM_manual_mode, 70, 180)/288).mean()
daytime_CGM_manual_mode['Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(daytime_CGM_manual_mode, 70, 150)/288).mean()
daytime_CGM_manual_mode['Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_manual_mode, 0, 70)/288).mean()
daytime_CGM_manual_mode['Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_manual_mode, 0, 54)/288).mean()
daytime_CGM_manual_mode = daytime_CGM_manual_mode.reset_index(drop=True)
daytime_CGM_manual_mode.dropna()
#print(daytime_CGM_manual_mode.iloc[0,:])

whole_day_CGM_manual_mode['Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_manual_mode, 180, 1000)/288).mean()
whole_day_CGM_manual_mode['Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_manual_mode, 250, 1000)/288).mean()
whole_day_CGM_manual_mode['Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(whole_day_CGM_manual_mode, 70, 180)/288).mean()
whole_day_CGM_manual_mode['Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(whole_day_CGM_manual_mode, 70, 150)/288).mean()
whole_day_CGM_manual_mode['Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_manual_mode, 0, 70)/288).mean()
whole_day_CGM_manual_mode['Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_manual_mode, 0, 54)/288).mean()
whole_day_CGM_manual_mode = whole_day_CGM_manual_mode.reset_index(drop=True)
whole_day_CGM_manual_mode.dropna()
#print(whole_day_CGM_manual_mode.iloc[0,:])

whole_day_CGM_manual_mode_final = whole_day_CGM_manual_mode.filter(items = header)
daytime_CGM_manual_mode_final = daytime_CGM_manual_mode.filter(items = header)
overnight_CGM_manual_mode_final = overnight_CGM_manual_mode.filter(items = header)
#print ("whole day final")
#print (whole_day_CGM_manual_mode_final.iloc[0,:])
#print ("daytime final")
#print (daytime_CGM_manual_mode_final.iloc[0,:])
#print ("overnight final")
#print (overnight_CGM_manual_mode_final.iloc[0,:])
overnight_CGM_manual_mode_final2 = overnight_CGM_manual_mode_final.drop(overnight_CGM_manual_mode_final.index[1:])
daytime_CGM_manual_mode_final2 = daytime_CGM_manual_mode_final.drop(daytime_CGM_manual_mode_final.index[1:])
whole_day_CGM_manual_mode_final2 = whole_day_CGM_manual_mode_final.drop(whole_day_CGM_manual_mode_final.index[1:])
#print(overnight_CGM_manual_mode_final2)
#print(daytime_CGM_manual_mode_final2)
#print (whole_day_CGM_manual_mode_final2)
#overnight_CGM_manual_mode_final2.to_csv('overnight.csv', index = True, header = False)
#daytime_CGM_manual_mode_final2.to_csv('daytime.csv', index = True, header = False)
#whole_day_CGM_manual_mode_final2.to_csv('wholeday.csv', index = True, header = False)
final_metrics_manual = pandas.concat([overnight_CGM_manual_mode_final2, daytime_CGM_manual_mode_final2, whole_day_CGM_manual_mode_final2], axis = 1)
#final_metrics_manual.to_csv('final_metrics_manual.csv', index = False, header = False)

##final_metrics_manual = pandas.concat([overnight_CGM_manual_mode, daytime_CGM_manual_mode, whole_day_CGM_manual_mode], axis = 1)

##final_metrics2_manual = final_metrics_manual[header]
#print (final_metrics2)
##final_metrics2_manual = final_metrics2_manual.drop(final_metrics2_manual.index[1:])
#print ("second print")
#print (final_metrics2)
##final_metrics2_manual.to_csv('Result_manual.csv', index = False, header = False)

################################
## Computes metrics in auto MODE
################################
df_auto_mode = df.loc[df['Date and Time'] >= start_auto_mode].copy()
#daytime_CGM_auto_mode = df_auto_mode[(df_auto_mode['Time'] >= '6:00:00') & (df_auto_mode['Time'] <= '23:59:59')].copy()
#overnight_CGM_auto_mode = df_auto_mode[(df_auto_mode['Time'] >= '00:00:00') & (df_auto_mode['Time'] <= '5:59:59')].copy()
#whole_day_CGM_auto_mode = df_auto_mode[(df_auto_mode['Time'] >= '00:00:00') & (df_auto_mode['Time'] <= '23:59:59')].copy()
#daytime_CGM_auto_mode = df_auto_mode.query("`Time` >= '06:00:00' and `Time` <= '23:59:59'")
#overnight_CGM_auto_mode = df_auto_mode.query("`Time` >= '00:00:00' and `Time` <= '06:00:00'")
#whole_day_CGM_auto_mode = df_auto_mode.query("`Time` >= '00:00:00' and `Time` <= '23:59:59'")
df_auto_mode['Time'] = pandas.to_datetime(df_auto_mode['Time']).dt.time
daytime_CGM_auto_mode = df_auto_mode[df_auto_mode['Time'].between(time(6, 0), time(23, 59))].copy()
#print("daytime")
#print(daytime_CGM_auto_mode)
overnight_CGM_auto_mode = df_auto_mode[df_auto_mode['Time'].between(time(0, 0), time(6, 0))].copy()

#print("overnight")
#print(overnight_CGM_auto_mode)
whole_day_CGM_auto_mode = df_auto_mode[df_auto_mode['Time'].between(time(0, 0), time(23, 59))].copy()

#print("whole day")
#print(whole_day_CGM_auto_mode)
#quit()

overnight_CGM_auto_mode['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_auto_mode, 180, 1000)/288)
#print(overnight_CGM)
overnight_CGM_auto_mode['Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_auto_mode, 250, 1000)/288).mean()
overnight_CGM_auto_mode['Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(overnight_CGM_auto_mode, 70, 180)/288).mean()
overnight_CGM_auto_mode['Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(overnight_CGM_auto_mode, 70, 150)/288).mean()
overnight_CGM_auto_mode['Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_auto_mode, 0, 70)/288).mean()
overnight_CGM_auto_mode['Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(overnight_CGM_auto_mode, 0, 54)/288).mean()
overnight_CGM_auto_mode.dropna()
overnight_CGM_auto_mode = overnight_CGM_auto_mode.reset_index(drop=True)
#overnight_CGM_auto_mode.to_csv('overnight_CGM_auto_mode.csv', index = True, header = False)
#print(overnight_CGM_auto_mode.iloc[0,:])

daytime_CGM_auto_mode['Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_auto_mode, 180, 1000)/288).mean()
daytime_CGM_auto_mode['Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_auto_mode, 250, 1000)/288).mean()
daytime_CGM_auto_mode['Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(daytime_CGM_auto_mode, 70, 180)/288).mean()
daytime_CGM_auto_mode['Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(daytime_CGM_auto_mode, 70, 150)/288).mean()
daytime_CGM_auto_mode['Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_auto_mode, 0, 70)/288).mean()
daytime_CGM_auto_mode['Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(daytime_CGM_auto_mode, 0, 54)/288).mean()
daytime_CGM_auto_mode = daytime_CGM_auto_mode.reset_index(drop=True)
daytime_CGM_auto_mode.dropna()
#print(daytime_CGM_auto_mode.iloc[0,:])

whole_day_CGM_auto_mode['Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_auto_mode, 180, 1000)/288).mean()
whole_day_CGM_auto_mode['Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_auto_mode, 250, 1000)/288).mean()
whole_day_CGM_auto_mode['Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (count_CGM_in_range(whole_day_CGM_auto_mode, 70, 180)/288).mean()
whole_day_CGM_auto_mode['Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (count_CGM_in_range(whole_day_CGM_auto_mode, 70, 150)/288).mean()
whole_day_CGM_auto_mode['Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_auto_mode, 0, 70)/288).mean()
whole_day_CGM_auto_mode['Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (count_CGM_in_range_strict_inequality(whole_day_CGM_auto_mode, 0, 54)/288).mean()
whole_day_CGM_auto_mode = whole_day_CGM_auto_mode.reset_index(drop=True)
whole_day_CGM_auto_mode.dropna()
#print(whole_day_CGM_auto_mode.iloc[0,:])

whole_day_CGM_auto_mode_final = whole_day_CGM_auto_mode.filter(items = header)
daytime_CGM_auto_mode_final = daytime_CGM_auto_mode.filter(items = header)
overnight_CGM_auto_mode_final = overnight_CGM_auto_mode.filter(items = header)
#print ("whole day final")
#print (whole_day_CGM_auto_mode_final.iloc[0,:])
#print ("daytime final")
#print (daytime_CGM_auto_mode_final.iloc[0,:])
#print ("overnight final")
#print (overnight_CGM_auto_mode_final.iloc[0,:])
overnight_CGM_auto_mode_final2 = overnight_CGM_auto_mode_final.drop(overnight_CGM_auto_mode_final.index[1:])
daytime_CGM_auto_mode_final2 = daytime_CGM_auto_mode_final.drop(daytime_CGM_auto_mode_final.index[1:])
whole_day_CGM_auto_mode_final2 = whole_day_CGM_auto_mode_final.drop(whole_day_CGM_auto_mode_final.index[1:])
#print(overnight_CGM_auto_mode_final2)
#print(daytime_CGM_auto_mode_final2)
#print (whole_day_CGM_auto_mode_final2)
#overnight_CGM_auto_mode_final2.to_csv('overnight.csv', index = True, header = False)
#daytime_CGM_auto_mode_final2.to_csv('daytime.csv', index = True, header = False)
#whole_day_CGM_auto_mode_final2.to_csv('wholeday.csv', index = True, header = False)
final_metrics_auto = pandas.concat([overnight_CGM_auto_mode_final2, daytime_CGM_auto_mode_final2, whole_day_CGM_auto_mode_final2], axis = 1)
final_metrics = pandas.concat([final_metrics_manual, final_metrics_auto], axis = 0) * 100
#final_metrics_auto.to_csv('final_metrics_auto.csv', index = False, header = False)
#quit()
#print (final_metrics2)
#final_metrics2_auto = final_metrics2_auto.drop(final_metrics2_auto.index[1:])
#print ("second print")
#print (final_metrics2)
final_metrics.to_csv('Result.csv', index = False, header = False)
