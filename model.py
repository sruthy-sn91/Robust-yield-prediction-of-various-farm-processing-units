import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# # train-data
train_data = pd.read_csv(r'train_data_manipulated.csv')

print('Shape of training dataset',train_data.shape)
print('\n')
# print('Shape of testing dataset',test_data.shape)

print('DataTypes \n', train_data.dtypes)
print('Null \n', train_data.isnull().sum())

train_data['date'] = pd.to_datetime(train_data['date'])
train_data['farm_id'] = train_data['farm_id'].astype('string')
print('DataTypes after cleaning \n', train_data.dtypes)

print('No. of rows before \n', train_data['farm_id'].count())
train_data = train_data.drop_duplicates(keep='first')
print('No. of rows after duplicates are removed \n',
      train_data['farm_id'].count())


# farm-data
farm_data = pd.read_csv(r'farm_data-1646897931981.csv')

print('Shape of farming dataset', farm_data.shape)

farm_data.drop('operations_commencing_year', axis=1, inplace=True)
print('DataTypes \n', farm_data.dtypes)
print('Null \n', farm_data.isnull().sum())


a=pd.pivot_table(farm_data,index=['farming_company'],values=['num_processing_plants'],
                 aggfunc={'num_processing_plants':'median'})
a.rename(columns={'num_processing_plants':'median_val'},inplace=True)

farm_data['num_processing_plants'].fillna('no', inplace=True)
farm_data=pd.merge(farm_data,a,on='farming_company')
farm_data["new"]=np.where(farm_data['num_processing_plants']=='no',farm_data['median_val'],farm_data['num_processing_plants'])
print('a \n',a,farm_data)
farm_data.drop('median_val',axis=1,inplace=True)
print('Fill null values\n', farm_data.isnull().sum())
# print('Duplicate value: ',farm_data.duplicated(subset=None,keep='first').sum())
farm_data.head()
farm_data.drop('num_processing_plants',axis=1,inplace=True)
farm_data.head()


# # weather-data
weather_data = pd.read_csv(r'train_weather_manipulated.csv')

print('Shape of farming dataset \n', weather_data.shape)
#print('Shape of testing dataset',test_data.shape)

weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
weather_data.rename(columns={'timestamp':'date'},inplace=True)
print('\n')
print('Weather Data Null Values')
print(weather_data.isnull().sum())

#Median Value for Temp_obs
temp_obs_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['temp_obs'],
                 aggfunc={'temp_obs':'median'})
temp_obs_median.rename(columns={'temp_obs':'median_val'},inplace=True)

weather_data['temp_obs'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,temp_obs_median,on='deidentified_location')
weather_data["temp_obs_new"]=np.where(weather_data['temp_obs']=='no',weather_data['median_val'],weather_data['temp_obs'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('temp_obs',axis=1,inplace=True)
#print(weather_data['temp_obs'].unique())

#Median Value for cloudiness
cloudiness_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['cloudiness'],
                 aggfunc={'cloudiness':'median'})
cloudiness_median.rename(columns={'cloudiness':'median_val'},inplace=True)

weather_data['cloudiness'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,cloudiness_median,on='deidentified_location')
weather_data["cloudiness_new"]=np.where(weather_data['cloudiness']=='no',weather_data['median_val'],weather_data['cloudiness'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.head()
weather_data.drop('cloudiness',axis=1,inplace=True)
#print(weather_data['cloudiness'].unique())


#Median Value for wind_direction            
wind_direction_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['wind_direction'],
                 aggfunc={'wind_direction':'median'})
wind_direction_median.rename(columns={'wind_direction':'median_val'},inplace=True)

weather_data['wind_direction'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,wind_direction_median,on='deidentified_location')
weather_data["wind_direction_new"]=np.where(weather_data['wind_direction']=='no',weather_data['median_val'],weather_data['wind_direction'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('wind_direction',axis=1,inplace=True)


#Median Value for dew_temp
dew_temp_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['dew_temp'],
                 aggfunc={'dew_temp':'median'})
dew_temp_median.rename(columns={'dew_temp':'median_val'},inplace=True)

weather_data['dew_temp'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,dew_temp_median,on='deidentified_location')
weather_data["dew_temp_new"]=np.where(weather_data['dew_temp']=='no',weather_data['median_val'],weather_data['dew_temp'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('dew_temp',axis=1,inplace=True)


#Median Value for pressure_sea_level
pressure_sea_level_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['pressure_sea_level'],
                 aggfunc={'pressure_sea_level':'median'})
pressure_sea_level_median.rename(columns={'pressure_sea_level':'median_val'},inplace=True)

weather_data['pressure_sea_level'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,pressure_sea_level_median,on='deidentified_location')
weather_data["pressure_sea_level_new"]=np.where(weather_data['pressure_sea_level']=='no',weather_data['median_val'],weather_data['pressure_sea_level'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('pressure_sea_level',axis=1,inplace=True)



#Median Value for precipitation
precipitation_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['precipitation'],
                 aggfunc={'precipitation':'median'})
precipitation_median.rename(columns={'precipitation':'median_val'},inplace=True)

weather_data['precipitation'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,precipitation_median,on='deidentified_location')
weather_data["precipitation_new"]=np.where(weather_data['precipitation']=='no',weather_data['median_val'],weather_data['precipitation'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('precipitation',axis=1,inplace=True)


#Median Value for wind_speed
wind_speed_median=pd.pivot_table(weather_data,index=['deidentified_location'],values=['wind_speed'],
                 aggfunc={'wind_speed':'median'})
wind_speed_median.rename(columns={'wind_speed':'median_val'},inplace=True)

weather_data['wind_speed'].fillna('no', inplace=True)
weather_data=pd.merge(weather_data,wind_speed_median,on='deidentified_location')
weather_data["wind_speed_new"]=np.where(weather_data['wind_speed']=='no',weather_data['median_val'],weather_data['wind_speed'])
weather_data.drop('median_val',axis=1,inplace=True)
weather_data.drop('wind_speed',axis=1,inplace=True)


#Weather_Data Column Drops
weather_data.drop('precipitation_new',axis=1,inplace=True)
weather_data.drop('cloudiness_new',axis=1,inplace=True)

print('Weather Data Null Values')
print(weather_data.isnull().sum())

print('\n')
print(weather_data.columns)
weather_data.dtypes

#Type Conversion

weather_data['deidentified_location'] = weather_data['deidentified_location'].astype('string')
weather_data['temp_obs_new'] = weather_data['temp_obs_new'].astype('float')
weather_data['wind_direction_new'] = weather_data['wind_direction_new'].astype('float')
weather_data['dew_temp_new'] = weather_data['dew_temp_new'].astype('float')
weather_data['pressure_sea_level_new'] = weather_data['pressure_sea_level_new'].astype('float')
weather_data['wind_speed_new'] = weather_data['wind_speed_new'].astype('float')
weather_data['date'] = weather_data['date'].astype('string')


weather_data.dtypes


#Merging train_data and farm_data on 'farm_id' column
train_data_merged = pd.merge(train_data,farm_data, on = 'farm_id')
train_data_merged['deidentified_location'] = train_data_merged['deidentified_location'].astype('string')
# print(train_data_merged.head())
# print(weather_data.head())
print(train_data_merged.dtypes)
print(weather_data.dtypes)
train_data_merged['date'] = train_data_merged['date'].astype('string')


weather_data["check"]=weather_data["deidentified_location"]+weather_data["date"]

# Concat trained_Data with weather_data
pivot=pd.pivot_table(weather_data,values=['temp_obs_new', 'wind_direction_new','dew_temp_new', 'pressure_sea_level_new', 'wind_speed_new'],
                     index=['check'],aggfunc={'temp_obs_new':'median', 'wind_direction_new':'median','dew_temp_new':'median', 'pressure_sea_level_new':'median', 'wind_speed_new':'median'}).reset_index()

train_data_merged.head()
train_data_merged["check"] = train_data_merged["deidentified_location"]+train_data_merged["date"]

Final_data= pd.merge(train_data_merged,weather_data,on="check",how="left")
Final_data.head()
Final_data.dtypes


Final_data['date_x'] = Final_data['date_x'].astype('datetime64[ns]')
Final_data['NewDateFormat'] = Final_data['date_x'].dt.strftime('%m/%Y')
# Final_data.rename(columns={'date_x':'month-year'},inplace=True)
Final_data.head()
Final_data.dtypes

# a = pd.pivot_table(Final_data,values=['wind_speed_new'],index=['NewDateFormat'],aggfunc={'wind_speed':''})
# a.to_excel(r"C:\Users\parab\OneDrive - HERE Global B.V-\Shraddha\NMIMS\Semester 7\Module 3 - BootCamp\Supply Chain Management Case Study\example.xls")

Final_data = Final_data[((Final_data['NewDateFormat']=='01/2016')|(Final_data['NewDateFormat']=='02/2016'))|((Final_data['NewDateFormat']=='04/2016')|(Final_data['NewDateFormat']=='05/2016'))|((Final_data['NewDateFormat']=='07/2016')|(Final_data['NewDateFormat']=='08/2016'))]

Final_data.head()

print(Final_data['NewDateFormat'].unique())
print(Final_data.shape)




#Correlation Plot on the numeric data
plt.figure(figsize=(22,10))
data = Final_data

df1 = pd.DataFrame(data)

corrMatrix = Final_data.corr(method = 'pearson',min_periods = 1)
sns.heatmap(corrMatrix, annot=True)
plt.show()

#Model Building 

#Necessary Import
from sklearn.model_selection import train_test_split

Final_data.head()
Final_data.drop('date_x', axis=1, inplace=True)
Final_data.drop('date_y', axis=1, inplace=True)
Final_data.drop('farm_id', axis=1, inplace=True)
Final_data.drop('check', axis=1, inplace=True)
Final_data.drop('NewDateFormat', axis=1, inplace=True)
Final_data.drop('deidentified_location_y', axis=1, inplace=True)

Final_data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Final_data.ingredient_type = le.fit_transform(Final_data.ingredient_type)
Final_data.farming_company = le.fit_transform(Final_data.farming_company)
Final_data.deidentified_location_x = le.fit_transform(Final_data.deidentified_location_x)
Final_data.new = le.fit_transform(Final_data.new)
Final_data['ingredient_type'] = Final_data['ingredient_type'].astype('category')
# Final_data['num_processing_plants'] = Final_data['num_processing_plants'].astype('float')
Final_data['farming_company'] = Final_data['farming_company'].astype('category')
Final_data['deidentified_location_x'] = Final_data['deidentified_location_x'].astype('category')
Final_data['new'] = Final_data['new'].astype('category')


Final_data.dtypes

Final_data.head()
Final_data = Final_data.sample(frac=0.00002)

y = Final_data['yield']
X = Final_data.loc[:, Final_data.columns != 'yield']
X.fillna(0,inplace=True)

X_train, X_test, y_train, y_test =   train_test_split(X, y, test_size=0.25,random_state = 123)


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

# Final_data.ingredient_type = le.fit_transform(Final_data.ingredient_type)
# Final_data.farming_company = le.fit_transform(Final_data.farming_company)
# Final_data.deidentified_location_x = le.fit_transform(Final_data.deidentified_location_x)
# Final_data.new = le.fit_transform(Final_data.new)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# First create the base model to tune
rf = RandomForestRegressor()
rf.fit(X_train, y_train)



train_pred1 = rf.predict(X_train)
test_pred1 = rf.predict(X_test)

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))


#export the model
pickle.dump(rf, open('model.pkl','wb'))
#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))


Final_data.head()
Final_data.dtypes


# Final_data.to_csv(r"C:\Users\parab\OneDrive - HERE Global B.V-\Shraddha\NMIMS\Semester 7\Module 3 - BootCamp\Supply Chain Management Case Study\filtereddata.csv")