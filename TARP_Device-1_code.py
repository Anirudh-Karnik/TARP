#Technical Answers to Real-World Problems (ECE1901)
#Project
#ML-based airfare prediction
#Under the guidance of Prof. Dr. Poonkuzhali R
#by:
#   Anirudh Karnik  (19BEC0353)
#   Arvind N        (19BEC0564)

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import pickle
import paho.mqtt.client as mqtt
#import graphviz


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#for dirname,_, filenames in os.walk(r'C:\hp\Users '):
#                                    print(os.path.join(dirname,filename))

##################################################################################################################################Analyzing and preparing the data for training
traindata=pd.read_excel(r"C:\Users\Anirudh Karnik\Desktop\TARP Project\Data_Train.xlsx",engine='openpyxl')
traindata.drop_duplicates(keep='first',inplace=True)
print(traindata["Additional_Info"].value_counts())
traindata["Additional_Info"] = traindata["Additional_Info"].replace({'No Info': 'No info'})
traindata['Duration']=  traindata['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

traindata["Journey_day"] = traindata['Date_of_Journey'].str.split('/').str[0].astype(int)
traindata["Journey_month"] = traindata['Date_of_Journey'].str.split('/').str[1].astype(int)
traindata.drop(["Date_of_Journey"], axis = 1, inplace = True)

traindata["Dep_hour"] = pd.to_datetime(traindata["Dep_Time"]).dt.hour
traindata["Dep_min"] = pd.to_datetime(traindata["Dep_Time"]).dt.minute
traindata.drop(["Dep_Time"], axis = 1, inplace = True)
traindata["Arrival_hour"] = pd.to_datetime(traindata.Arrival_Time).dt.hour
traindata["Arrival_min"] = pd.to_datetime(traindata.Arrival_Time).dt.minute
traindata.drop(["Arrival_Time"], axis = 1, inplace = True)
traindata['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)
traindata["Airline"].value_counts()
traindata["Airline"].replace({'Multiple carriers Premium economy':'Other', 
                                                        'Jet Airways Business':'Other',
                                                        'Vistara Premium economy':'Other',
                                                        'Trujet':'Other'
                                                   },    
                                        inplace=True)
traindata["Additional_Info"].value_counts()
traindata["Additional_Info"].replace({'Change airports':'Other', 
                                                        'Business class':'Other',
                                                        '1 Short layover':'Other',
                                                        'Red-eye flight':'Other',
                                                        '2 Long layover':'Other',   
                                                   },    
                                        inplace=True)

traindata.head()
data = traindata.drop(["Price"], axis=1)
train_categorical_data = data.select_dtypes(exclude=['int64', 'float64','int32'])
train_numerical_data = data.select_dtypes(include=['int64', 'float32','int32'])
train_categorical_data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_categorical_data = train_categorical_data.apply(LabelEncoder().fit_transform)
train_categorical_data.head()
X = pd.concat([train_categorical_data, train_numerical_data], axis=1)
y=traindata['Price']
X.head()
y.head()
#data pre-processing complete

#################################################################################################################################################Done analyzing and preparing the data for training

##################################################################################################################################Analyzing and preparing the data for training
preddata=pd.read_excel(r"C:\Users\Anirudh Karnik\Desktop\TARP Project\Data_test_2 - Copy.xlsx",engine='openpyxl')
preddata.drop_duplicates(keep='first',inplace=True)
print(preddata["Additional_Info"].value_counts())
preddata["Additional_Info"] = preddata["Additional_Info"].replace({'No Info': 'No info'})
preddata['Duration']=  preddata['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

preddata["Journey_day"] = preddata['Date_of_Journey'].str.split('/').str[0].astype(int)
preddata["Journey_month"] = preddata['Date_of_Journey'].str.split('/').str[1].astype(int)
preddata.drop(["Date_of_Journey"], axis = 1, inplace = True)

preddata["Dep_hour"] = pd.to_datetime(preddata["Dep_Time"]).dt.hour
preddata["Dep_min"] = pd.to_datetime(preddata["Dep_Time"]).dt.minute
preddata.drop(["Dep_Time"], axis = 1, inplace = True)
preddata["Arrival_hour"] = pd.to_datetime(preddata.Arrival_Time).dt.hour
preddata["Arrival_min"] = pd.to_datetime(preddata.Arrival_Time).dt.minute
preddata.drop(["Arrival_Time"], axis = 1, inplace = True)
#preddata['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)
preddata["Airline"].value_counts()
preddata["Airline"].replace({'Multiple carriers Premium economy':'Other', 
                                                        'Jet Airways Business':'Other',
                                                        'Vistara Premium economy':'Other',
                                                        'Trujet':'Other'
                                                   },    
                                        inplace=True)
preddata["Additional_Info"].value_counts()
preddata["Additional_Info"].replace({'Change airports':'Other', 
                                                        'Business class':'Other',
                                                        '1 Short layover':'Other',
                                                        'Red-eye flight':'Other',
                                                        '2 Long layover':'Other',   
                                                   },    
                                        inplace=True)

preddata.head()
#data_toPredict = preddata.drop(["Price"], axis=1)
data_toPredict = preddata
train_categorical_datatopredict = data_toPredict.select_dtypes(exclude=['int64', 'float64','int32'])
train_numerical_datatopredict = data_toPredict.select_dtypes(include=['int64', 'float32','int32'])
train_categorical_datatopredict.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_categorical_datatopredict = train_categorical_datatopredict.apply(LabelEncoder().fit_transform)
train_categorical_datatopredict.head()
X_pred = pd.concat([train_categorical_datatopredict, train_numerical_datatopredict], axis=1)
#y_pred=preddata['Price']
X_pred.head()
#y_pred.head()
#################################################################################################################################################Done analyzing and preparing the data for training


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,shuffle=False)
print(X_train)
print("The size of training input is", X_train.shape)
print("The size of training output is", y_train.shape)
print(50 *'*')
print("The size of testing input is", X_test.shape)
print("The size of testing output is", y_test.shape)
depth  =list(range(3,30))
param_grid =dict(max_depth =depth)

#desicion tree
tree =GridSearchCV(DecisionTreeRegressor(min_samples_leaf=2),param_grid,cv =10)
reg=tree.fit(X_train,y_train)
y_train_pred =tree.predict(X_train)
y_test_pred =tree.predict(X_test) 
print("The train results for Train data: ")
print(y_train_pred[0])
print("Train Results for Decision Tree Regressor Model:")
print(50 * '-')
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
pickle.dump(tree,open('modeldecision.pkl','wb'))

#random forest
#tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
#random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 20, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)
#random_regressor.fit(X_train, y_train)
#y_train_pred = random_regressor.predict(X_train)
#y_test_pred = random_regressor.predict(X_test)
#print("The train results for Train data: ")
#print(y_train_pred[0])
#print("Train Results for Random Forest Regressor Model:")
#print(50 * '-')
#print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
#print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
#print("R-squared: ", r2_score(y_train.values, y_train_pred))
#pickle.dump(random_regressor,open('modelrf.pkl','wb'))

#XGBoost
#tuned_params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}
#model = RandomizedSearchCV(XGBRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)
#model.fit(X_train, y_train)
#y_train_pred = model.predict(X_train)
#y_test_pred = model.predict(X_test)
#print("The train results for Train data: ")
#print(y_train_pred[0])
#print("Train Results for XGBoost Regressor Model:")
#print(50 * '-')
#print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
#print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
#print("R-squared: ", r2_score(y_train.values, y_train_pred))
#pickle.dump(model,open('modelXGB.pkl','wb'))

#visualizing the tree
#dot_data = tree.export_graphviz(tree, out_file=None, 
                                  
#                                class_names=X.head(),
#                                filled=True)
#graph = graphviz.Source(dot_data, format="png") 
#graph



pd.set_option('display.max_columns', None)

airline = int(input("Enter Airline [0 - Air Asia, 1 - AirIndia, 2 - GoAir, 3 - IndiGo, 4 - Jet Airways, 5 - Other, 6 - Jet Airways Business, 7 - SpiceJet, 8 - Vistara]: "))
origin = int(input("Enter Origin [0 - Bangalore, 1 - Chennai, 2 - Delhi, 3 - Kolkata, 4 - Mumbai]: "))
destination = int(input("Enter Destination [0 - Bangalore, 1 - Cochin, 2 - Delhi, 3 - Hyderabad, 4 - Kolkata, 5 - New Delhi]: "))
day = int(input("Enter Date (day): "))
month = int(input("Enter Date (month): "))
year = int(input("Enter Date (year): "))
route = 0
duration = 0
dephr=0
depmin=0
arrhr=0
arrmin=0
mqtt_topic2=""

if(origin == 0 and destination == 2):
    route = 18
    duration = 170
    dephr = 19
    depmin = 20
    arrhr = 22
    arrmin = 10
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: DEL 19:20"

elif(origin == 0 and destination == 5):
    route = 18
    duration = 170
    dephr = 19
    depmin = 20
    arrhr = 22
    arrmin = 10
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: DEL 19:20"
    
elif(origin == 1 and destination == 4):
    route = 127
    duration = 135
    dephr = 18
    depmin = 30
    arrhr = 20
    arrmin = 45
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: CCU 18:30"

elif(origin == 2 and destination == 1):
    route = 106
    duration = 195
    dephr = 8
    depmin = 30
    arrhr = 11
    arrmin = 45
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: COK 08:30"

elif(origin == 3 and destination == 0):
    route = 64
    duration = 145
    dephr = 11
    depmin = 20
    arrhr = 13
    arrmin = 45
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: BLR 11:20"

elif(origin == 4 and destination == 3):
    route = 48
    duration = 90
    dephr = 1
    depmin = 0
    arrhr = 3
    arrmin = 30
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "Flt: HYD 01:00"

else:
    print("No flights");
    X_pred.loc[0] = [airline,origin,destination,route,3,duration,day,month,dephr,depmin,arrhr,arrmin]
    mqtt_topic2 = "No flt scheduled"


a=tree.predict(X_pred)
#a=random_regressor.predict(X_pred)
#a=model.predict(X_pred)
print("The predicted airfare is: ", end = "")
print(a[0])


###################################################################### MQTT ###################################################################


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("test.mosquitto.org", 1883, 60)

client.publish("tarp/flight", payload=mqtt_topic2, qos=0, retain=False)
print("Flight details published over MQTT")
