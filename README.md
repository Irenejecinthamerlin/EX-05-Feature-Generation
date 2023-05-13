# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
## Data.csv
```
import pandas as pd
df=pd.read_csv("data.csv")
df

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()
df1=df.copy()

df1["City"] = oe.fit_transform(df1[["City"]])
df1["bin_1"] = oe.fit_transform(df1[["bin_1"]])
df1["Ord_1"] = oe.fit_transform(df1[["Ord_1"]])
df1["Ord_2"] = oe.fit_transform(df1[["Ord_2"]])
df1["bin_2"] = oe.fit_transform(df1[["bin_2"]])

df2=df.copy()
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
## Encoding.csv
```
import pandas as pd
qf=pd.read_csv("encoding.csv")
qf

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()

qf1=qf.copy()


qf1["bin_1"] = oe.fit_transform(qf1[["bin_1"]])
qf1["nom_0"] = oe.fit_transform(qf1[["nom_0"]])
qf1["ord_2"] = oe.fit_transform(qf1[["ord_2"]])
qf1["bin_2"] = oe.fit_transform(qf1[["bin_2"]])
```

#feature scaling
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
qf0=pd.DataFrame(sc.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf0   

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
qf2=pd.DataFrame(sc1.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
qf3=pd.DataFrame(sc2.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
qf4=pd.DataFrame(sc3.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf4
```
## Titanic_dataset.csv
import pandas as pd
rf=pd.read_csv("titanic.csv")
rf

#removing unwanted data
rf.drop("Name",axis=1,inplace=True)
rf.drop("Ticket",axis=1,inplace=True)
rf.drop("Cabin",axis=1,inplace=True)  

rf["Age"]=rf["Age"].fillna(rf["Age"].median())
rf["Embarked"]=rf["Embarked"].fillna(rf["Embarked"].mode()[0])

rf.isnull().sum()

rf1=rf.copy()

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
oe=OrdinalEncoder()

e1=OrdinalEncoder(categories=[embark])
rf1['Embarked'] = e1.fit_transform(rf[['Embarked']])
rf1['Sex'] = oe.fit_transform(rf[['Sex']])
rf1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
rf0=pd.DataFrame(sc.fit_transform(rf1),columns=['PassengerId', 'Survived', 'Pclass', 'Sex','Age','SibSp','Parch','Fare','Embarked'])
rf0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
rf3=pd.DataFrame(sc1.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
rf4=pd.DataFrame(sc2.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
rf5=pd.DataFrame(sc3.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf5
```
# OUPUT
## Data.csv:
## Initial dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232380635-9b170286-e8e4-43ce-8df6-f3715202cd8e.png)
## Encoded dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232380721-af195910-8b6e-40d6-ab95-6835b8511fe8.png)
## Data scaling using MinMaxScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232380780-da31e3f1-9310-402e-a7e3-d88251b98e03.png)
## Data scaling using MaxAbsScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232380828-25ea354a-9683-4db6-b5f3-9ce1f5ea541b.png)
## Data scaling using RobustScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232380907-27a05fae-8d20-415d-aa79-fae9259733b3.png)
# Encoding.csv:
## Initial dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232381009-c226c605-66ff-4dc0-a66e-c9c4aea3e59e.png)
## Encoded dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232381086-43f139da-487e-4b15-b705-7e1e1d18f9c1.png)
## Data scaling using MinMaxScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232385345-afd8f8c2-c826-4c39-8973-825012e48cd5.png)
## Data scaling using StandardScalar:
### ![image](https://user-images.githubusercontent.com/128350225/232387314-fd592861-ecaa-4365-8fb1-8ec4900dfb70.png)
## Data scaling using MaxAbsScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232387450-0b70ab92-4a9d-44b3-a422-44bdf13b18ae.png)
## Data scaling using RobustScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232387640-baa09bdc-4eb7-41d6-9466-268f683ec6cf.png)
# Titanic_dataset.csv:
## Initial dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232388056-a8c40d89-dc90-48a3-b195-f3bfa10b28f1.png)
## isnull.sum()
### ![image](https://user-images.githubusercontent.com/128350225/232388160-9b28bebb-e9ab-441a-8fe1-75f8806e1f6b.png)
## Encoded dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232388285-9e6262ec-a9be-46dd-8449-3df0c198d5f4.png)
## Data scaling using MinMaxScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232388363-7ec62e11-8cea-4478-959a-d547abe12c4a.png)
## Data scaling using StandardScalar:
### ![image](https://user-images.githubusercontent.com/128350225/232388487-cf6cc218-5705-4bba-9636-5a451f871891.png)
## Data scaling using MaxAbsScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232388625-3ae32129-fa1a-4e8a-b979-2f1b7ea575cc.png)
## Data scaling using RobustScaler:
### ![image](https://user-images.githubusercontent.com/128350225/232388725-65dcabb1-aa2e-4f86-a1dd-4083e597969b.png)
# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
