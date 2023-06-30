#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries and loading the datasets

# In[1]:


import pandas as pd
import numpy as np

accident_data_17=pd.read_csv(r"D:\Accident Prediction\Publication 2023\Dataset\Raw CSV\DMP-2017-all.csv")
accident_data_18=pd.read_csv(r"D:\Accident Prediction\Publication 2023\Dataset\Raw CSV\DMP-2018-all.csv")
accident_data_19=pd.read_csv(r"D:\Accident Prediction\Publication 2023\Dataset\Raw CSV\DMP-2019-all.csv")
accident_data_20=pd.read_csv(r"D:\Accident Prediction\Publication 2023\Dataset\Raw CSV\DMP-2020-all.csv")

accident_data = pd.concat([accident_data_17, accident_data_18, accident_data_19, accident_data_20], ignore_index=True)

pd.set_option("display.max_columns", None)

accident_data.head() # Viewing the first 5 rows


# In[2]:


# viewing the shape of the data
accident_data.shape


# # Dropping the uncecessary columns

# In[3]:


columns_to_drop = [
    "Month","Date in month","Year","Passenger vehicle code","Passenger sex","Passenger age","Passenger injury",
    "Position in vehicle","Passenger action","Pedestrian vehicle code",
    "Pedestrian sex","Pedestrian age","Pedestrian injury",
    "Pedestrian location","Pedestrian action","contributory factors 1",
    "contributory factors 2","contributory factors 3",
    "Accident description","Location description","RUM code prior",
    "RUM code chosen","RUM code subsequent","Vehicle damage",
    "District of license issue","License num","Driver sex","Driver age",
    "Driver injury","XY map code","X coordinate","Y coordinate",
    "Route number","Kilometer post","x100 meters","Node map code",
    "Node 1","Node 2","District of registration","Vehicle registration",
    "Fitness certificate","Day of week","Accident ID",
    "Accident count","Report number","Unnamed: 0","FIR number","Thana","District",
    "No of vehicle","No of driver casualties","No of passenger casualties",
    "No of pedestrian casualties"]

accident_data.drop(columns_to_drop, axis=1, inplace=True)

accident_data.head()


# # Changing the Time into four quarters of a day

# In[4]:


accident_data["Accident severity"] = accident_data["Accident severity"].replace({"S": "Simple Injury", "F": "Fetal Injury", "G": "Grievious", "M": "Motor Collision"})
accident_data["Time"] = accident_data["Time"].astype(str)
accident_data["Time"] = accident_data["Time"].str.slice(0, 2) + ":" + accident_data["Time"].str.slice(2)
def change_time(time):
    if ":" in time:
        hour, minute = time.split(":")
        if hour.strip().isdigit(): 
            if int(hour) <= 6:
                return "Night"
            elif int(hour) > 6 and int(hour) <= 12:
                return "Day"
            elif int(hour) > 12 and int(hour) <= 18:
                return "Day"
            else:
                return "Night"
        else:
            return

accident_data["Time"] = accident_data["Time"].astype(str)
accident_data["Time"] = accident_data["Time"].apply(change_time)

accident_data.head()


# # Preparing the raw dataset by decoding the numeric values

# In[5]:


accident_data["Junction type"] = accident_data["Junction type"].replace({1.0: "No Junction","1": "No Junction", 2.0:"Junction", "2":"Junction", 3.0:"Junction", "3":"Junction", 4.0: "Junction", "4": "Junction", 5.0: "Junction", "5": "Junction", 6.0: "Junction", "6": "Junction", 7.0:"Junction","7":"Junction"})

accident_data["Traffic control"] = accident_data["Traffic control"].replace({ 1.0:"No control", 2.0:"control", 3.0:"control", 4.0:"control", 5.0:"control", 6.0:"control", 7.0: "control",8.0:"control"})

accident_data["Traffic control"] = accident_data["Traffic control"].replace({ "1":"No control", "2":"control", "3":"control", "4":"control", "5":"control", "6":"control", "7": "control","8":"control"})

accident_data["Collision type"] = accident_data["Collision type"].replace({1.0 : "Head On", 2.0 : "Not Head On", 3.0 : "Not Head On", 4.0 : "Not Head On", 5.0 : "Not Head On", 6.0 : "Not Head On", 7.0 : "Not Head On", 8.0 : "Not Head On", 9.0 : "Not Head On", 10.0 : "Not Head On", 11.0 : "Not Head On"})

accident_data["Movement"] = accident_data["Movement"].replace({1.0: "1-way Street", 2.0: "2-way Street"})

accident_data["Movement"] = accident_data["Movement"].replace({"1": "1-way Street", "2": "2-way Street"})

accident_data["Weather"] = accident_data["Weather"].replace({ 1.0:"Fair", 2.0:"Bad", 3.0:"Bad", 4.0:"Bad"})

accident_data["Light"] = accident_data["Light"].replace({ 1.0: "Daylight", 2.0:"No Daylight", 3.0:"No Daylight", 4.0: "No Daylight"})

accident_data["Road geometry"] = accident_data["Road geometry"].replace({1.0:"Straight+Flat",2.0:"Not Straight+Flat", 3.0:"Not Straight+Flat", 4.0:"Not Straight+Flat", 5.0:"Not Straight+Flat"})

accident_data["Surface type"] = accident_data["Surface type"].replace({1.0:"Sealed", 2.0:"Unsealed", 3.0:"Unsealed"})

accident_data["Surface quality"] = accident_data["Surface quality"].replace({1.0:"Good Quality",2.0:"Poor Quality",3.0:"Poor Quality"})

accident_data["Road class"] = accident_data["Road class"].replace({1.0:"National",2.0:"Regional", 3.0:"Regional",4.0:"Regional",5.0:"Regional"})

accident_data["Road feature"] = accident_data["Road feature"].replace({1.0:"None",2.0:"Feature Present",3.0:"Feature Present",4.0:"Feature Present",5.0:"Feature Present"})

accident_data["Road feature"] = accident_data["Road feature"].replace({"1":"None","2":"Feature Present","3":"Feature Present","4":"Feature Present","5":"Feature Present"})

accident_data["Location type"] = accident_data["Location type"].replace({"1":"Urban Road","2":"Rural Area"})

accident_data["Location type"] = accident_data["Location type"].replace({1.0:"Urban Road",2.0:"Rural Area"})

accident_data["Vehicle loading"] = accident_data["Vehicle loading"].replace({1.0:"Legal",2.0:"Unsafe"})

accident_data["Vehicle loading"] = accident_data["Vehicle loading"].replace({"1":"Legal","2":"Unsafe"})

accident_data["Vehicle defects"] = accident_data["Vehicle defects"].replace({1.0:"No Defects",2.0:"Defects",3.0:"Defects", 4.0:"Defects",5.0:"Defects",6.0:"Defects",7.0:"Defects"})

accident_data["Vehicle defects"] = accident_data["Vehicle defects"].replace({"1":"No Defects","2":"Defects","3":"Defects", "4":"Defects","5":"Defects","6":"Defects","7":"Defects"})

accident_data["Alcohol"] = accident_data["Alcohol"].replace({1.0:"Suspected", 2.0:"Not Suspected"})

accident_data["Alcohol"] = accident_data["Alcohol"].replace({"1":"Suspected", "2":"Not Suspected"})

accident_data["Seat belt"] = accident_data["Seat belt"].replace({1.0:"Yes", 2.0:"No"})

accident_data["Seat belt"] = accident_data["Seat belt"].replace({"1":"Yes", "2":"No"})

accident_data["Vehicle type"] = accident_data["Vehicle type"].replace({1.0:"Others",2.0:"Others",3.0:"Others",4.0:"Others",5.0:"Others",6.0:"Others", 7.0:"Bus+Truck", 8.0:"Bus+Truck", 9.0:"Bus+Truck",10.0:"Others",11.0:"Others",12.0:"Others",13.0:"Bus+Truck",14.0:"Bus+Truck",15.0:"Bus+Truck",16.0:"Bus+Truck", 17.0:"Bus+Truck", 18.0:"Others",19.0:"Others"})

accident_data["Vehicle type"] = accident_data["Vehicle type"].replace({"01":"Others","02":"Others","03":"Others","04":"Others","05":"Others","06":"Others", "07":"Bus+Truck", "08":"Bus+Truck", "09":"Bus+Truck","10":"Others","11":"Others","12":"Others","13":"Bus+Truck","14":"Bus+Truck","15":"Bus+Truck","16":"Bus+Truck", "17":"Bus+Truck", "18":"Others","19":"Others"})

accident_data["Vehicle maneuver"]= accident_data["Vehicle maneuver"].replace({1.0:"Not Overtaking",2.0:"Not Overtaking",3.0:"Not Overtaking",4.0:"Not Overtaking", 5.0:"Overtaking",6.0:"Not Overtaking",7.0:"Not Overtaking",8.0:"Not Overtaking",9.0:"Not Overtaking",10.0:"Not Overtaking",11.0:"Not Overtaking"})

accident_data["Vehicle maneuver"]= accident_data["Vehicle maneuver"].replace({"01":"Not Overtaking","02":"Not Overtaking","03":"Not Overtaking","04":"Not Overtaking", "05":"Overtaking","06":"Not Overtaking","07":"Not Overtaking","08":"Not Overtaking","09":"Not Overtaking","10":"Not Overtaking","11":"Not Overtaking"})

accident_data["Surface condition"] = accident_data["Surface condition"].replace({1.0:"Dry",2.0:"Wet",3.0:"Wet",4.0:"Wet",5.0:"Wet"})

accident_data["Divider"] = accident_data["Divider"].replace({1.0:"Yes",2.0:"No"})

pd.set_option("display.max_rows", None)
accident_data.head()


# # Checking the missing values

# In[6]:


accident_data.isnull().sum()


# In[7]:


accident_data["Accident severity"].unique()


# # Fill missing values by mode and Replacing the anomalous Values 

# In[8]:


accident_data.replace(["?", " ?"], float("nan"), inplace= True)
accident_data.fillna(accident_data.mode().iloc[0], inplace= True)
accident_data["Accident severity"] = accident_data["Accident severity"].replace("Simple Injury", "Non Fatal Injury")
accident_data["Accident severity"] = accident_data["Accident severity"].replace("Grievious", "Non Fatal Injury")
accident_data["Accident severity"] = accident_data["Accident severity"].replace("Motor Collision", "Non Fatal Injury")
accident_data["Accident severity"] = accident_data["Accident severity"].replace("Fetal Injury", "Fatal Injury")



accident_data.head()


# # now perform EDA to check the different classes of data

# In[9]:


# Assuming accident_data is a DataFrame
for column in accident_data.columns:
    unique_classes, counts = np.unique(accident_data[column], return_counts=True)
    print(f"Column '{column}' has the following unique classes:")
    for unique_class, count in zip(unique_classes, counts):
        print(f"Class '{unique_class}': {count} occurrences")
    print("\n")


# In[10]:


accident_data.isnull().sum()


# In[11]:


accident_data["Accident severity"].value_counts()


# # Saving the prepared raw dataset

# In[12]:


# accident_data.to_csv(r"D:\Accident Prediction\Publication 2023\Preparing the raw dataset\final_accident_data.csv", index=False)

