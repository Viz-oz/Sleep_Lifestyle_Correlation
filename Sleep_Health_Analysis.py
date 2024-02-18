#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The project explores the relationship between sleep disorder and Lifestyle (features) presented in the dataset 
# Moreover, the project aims to understand and predict sleep disroder with the available features. 


#Importing necessary libraries 

import pandas as pd #Data processing

import numpy as np #linear algebra

import matplotlib.pyplot as plt #visualization

import seaborn as sns #visualization

import warnings 
warnings.filterwarnings("ignore") #filter out warning messsages

from sklearn.preprocessing import LabelEncoder #encoding categorical labels

from sklearn.preprocessing import StandardScaler #ensure features have similiar scale

from sklearn.model_selection import train_test_split #features engineering

from lazypredict.Supervised import LazyClassifier #automation


from sklearn.linear_model import LogisticRegression #classification

from sklearn.ensemble import RandomForestClassifier #randomforest classification


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #evaluate the prediction

from IPython.display import FileLink #To download the prediction csv


# In[4]:


#Importing the dataset 

file_path = "D:\\VV_Arbeit\\Data_Analytics_Portfolio\\Projects\\Completed_Projects\\Health_sleep_pattern\\Sleep_health_and_lifestyle_dataset.csv"
data = pd.read_csv(file_path)


# In[5]:


#Explorartory Data Analysis (EDA)

data.sample(5,random_state=42)


# In[5]:


data.head(6)


# In[6]:


data.info()


# In[7]:


data.describe()


# In[7]:


#Checking for missing values 

data.isnull().sum()


# In[9]:


#There are no missing values in the dataset 


# In[8]:


#Replacing the Sleep Disorder column from Float dtype into int64 dtype for easy analysis and intepretation. 

mapping = {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 1}

# Replace categorical values in the 'Sleep Disorder' column with numerical values
data['Sleep Disorder'] = data['Sleep Disorder'].replace(mapping).astype('int64')


# In[9]:


data.tail(3) 


# In[15]:


#Succesfully converted Float into Int64 for coloumn Sleep Disorder. Whereas, 0 = No Sleep Disorder and 1 = Sleep Disorder Exists


# In[10]:


#Plotting the dataset in Pie to chart to determine the frequency of Sleep Disorders 

labels =data['Sleep Disorder'].value_counts(sort = True).index
sizes = data['Sleep Disorder'].value_counts(sort = True)

colors = ["lightGreen","red"]
explode = (0.05,0) 
 
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)

plt.title('Percentage of Sleep Disorder in the dataset')

# Adding legend
plt.legend(labels=['No Sleep Disorder', 'Sleep Disorder'])

plt.show()


# In[11]:


#It can be understood that within the dataset, 41% of the people face sleep disorder 


# In[18]:


#As, 41% of the people face sleep disorder, let us dive into the determining factors of sleep disorder 


# In[12]:


#Data Visualization for features exploration 

#Graph_1- Relationship between Occupation and Sleep Disorder 

sns.catplot(y="Occupation", hue="Sleep Disorder", kind="count",
            palette="Blues", edgecolor=".6",
            data=data)


# In[21]:


#Graph_2- Relationship between Gender and Sleep Disorder 

sns.catplot(y="Gender", hue="Sleep Disorder", kind="count",
            palette="Blues", edgecolor=".6",
            data=data)


# In[22]:


#Graph_3- Relationship between BMI Category and Sleep Disorder 

sns.catplot(y="BMI Category", hue="Sleep Disorder", kind="count",
            palette="Blues", edgecolor=".6",
            data=data)


# In[ ]:


#Assumptions based on the EDA 

#1 - Out of the 11 occupations listed, Nurses, teachers and Sales Professionals face sleep disorder the most 

#2 - Compared with Males, Females tend to suffer some sort of sleep disored the most 

#3- BMI is a strong determinor of sleep quality. Overweight people have high probability of sleep disorder


# In[25]:


#Plotting heatmap to identify features 

#Dropping Id-column as it is irrelevant 
data_no_id = data.drop(columns=['Person ID'])

fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = data_no_id.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
plt.title("\nCorrelation Heatmap\n", fontsize=16, weight="bold")
plt.show()


# In[36]:


# Defining features & Preprocessing the data for train/test split 
#x = features
#y = target
x=data_no_id.drop("Sleep Disorder",axis=1)
y=data_no_id['Sleep Disorder']

x.tail()


# In[37]:


y.tail()


# In[38]:


# Separate categorical and numerical columns
categorical_cols = x.select_dtypes(include=['object']).columns
numerical_cols = x.select_dtypes(include=['int64']).columns

# Apply LabelEncoder to categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    x[col] = label_encoder.fit_transform(x[col])
    # Save label encoders
label_encoders = {col: label_encoder for col in categorical_cols}


x.tail() #to view the result after encoding


# In[39]:


print(x.dtypes)


# In[40]:


print(x.isnull().sum())


# In[41]:


print(x.shape)
print(y.shape)


# In[42]:


#splitting data into train and test 


x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)

print(x_tr.shape)
print(x_tst.shape)
print(y_tr.shape)
print(y_tst.shape)


# In[43]:


# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(x_tr, y_tr)


# In[44]:


# Make predictions on the test set
y_pred = model.predict(x_tst)


# In[45]:


# Create a DataFrame with 'Actual' and 'Predicted' columns
predictions_df = pd.DataFrame({
    'Actual': y_tst,      # Actual labels
    'Predicted': y_pred   # Predicted labels
})

# Save the DataFrame to a CSV file
csv_file_path = 'predictions.csv'
predictions_df.to_csv(csv_file_path, index=False,header=['Actual', 'Predicted'])

# Display a download link
FileLink(csv_file_path)


# In[46]:


# Evaluate model performance
accuracy = accuracy_score(y_tst, y_pred)
classification_report = classification_report(y_tst, y_pred)
conf_matrix = confusion_matrix(y_tst, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report)
print("\nConfusion Matrix:\n", conf_matrix)


# In[ ]:




