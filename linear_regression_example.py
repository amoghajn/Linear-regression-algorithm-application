# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 


st.title("Health Care Application using Machine Learning Algorithm ")
from PIL import Image
image1 = Image.open(r"C:\Users\HP\Desktop\health-insurance.jpg.crdownload")

st.image(image1)
df=pd.read_csv(r"C:\Users\HP\Downloads\insurance.csv")
#checking the data

def form_callback():
    #st.write(st.session_state.my_slider)
    st.write(st.session_state.my_checkbox)
    
with st.sidebar:
     #raw_data= st.selectbox("Dataset analysis",('raw data' ,'countplots related to EDA','boxplots related to EDA','data describing' ,'Heatmap'))

     show1 = st.checkbox('make predictions')
if show1:
     
    st.title(" Health insurance prediction using linear Regression")
    age1=st.number_input("enter the age of the person")
    sex1=st.selectbox('select the gender', ('Male','Female'))
    bmi1=st.number_input("enter the bmi of the person")
    smoker1=st.selectbox('smoker', ('Yes','No'))
    region1=st.selectbox('select the region',('southeast','southwest','northeast','northwest'))
    children1=st.number_input("enter the number of children in range of 1 to 5")
def ploting_features_count(feat):
    
     sns.set()
     fig=plt.figure(figsize=(6,6))
     sns.countplot(feat)
     st.title("the countplot for feature vs charges")
     st.pyplot(fig)



def ploting_features_box(feat,target):
    
     sns.set()
     fig=plt.figure(figsize=(6,6))
     sns.boxplot(feat,target)
     st.title("the boxplot for feature vs charges",)
     st.pyplot(fig)

#checking the information 
with st.sidebar:
     raw_data= st.selectbox("Dataset analysis",('raw data' ,'countplots related to EDA','boxplots related to EDA','data describing' ,'Heatmap'))

     show = st.checkbox('show data analysis')

if show:
    
    
    if raw_data=='raw data':
        st.write("showing the raw dataset")
        st.dataframe(df.head())
    elif raw_data=='countplots related to EDA':
        st.write("you selected ", raw_data)
    #st.dataframe(df.summary())     
        ploting_features_count(df['sex']) 
        ploting_features_count(df['smoker']) 
        ploting_features_count(df['region'])     
    elif raw_data=='boxplots related to EDA':
        ploting_features_box(df['smoker'],df['charges'])
        ploting_features_box(df['children'],df['charges'])
        ploting_features_box(df['region'],df['charges'])
    elif raw_data=='data describing':
        st.write("describing the dataset")
        st.dataframe(df.describe())
   # converting the categorical features into numerical 
   
   # encoding sex column
df.replace({'sex':{'male':0,'female':1}}, inplace=True)

 # encoding 'smoker' column
df.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

    


if raw_data=='Heatmap':
    st.write("showing the heatmap , feature correlation ")
    sns.set()
    fig=plt.figure(figsize=(6,6))
    sns.heatmap(df.corr(),annot=True) 
    st.pyplot(fig)
   
# to seperate the Depedendent feature and independent features by dropping dependent feature "charges"
X=df.drop(columns='charges',axis=1)
Y=df['charges']  

#Train test split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#print("the shape X is ;",X.shape)
#print("the shape X_train is :",X_train.shape)
#print("the shape X_test is :",X_test.shape)

# loading the Linear Regression model
regressor = LinearRegression()

model=regressor.fit(X_train,Y_train)
training_data_prediction =regressor.predict(X_train)
#print(training_data_prediction)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale for training data: ', r2_train)
# prediction on test data
test_data_prediction =regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale for testing data: ', r2_test)

regressor2= LinearRegression()

#print(X_train)
X_train1=X_train.drop(columns=['region','sex','children'],axis=1)
#df1=df1.drop(columns=['region','sex'],axis=1)
X_test_1=X_test.drop(columns=['region','sex','children'],axis=1)
model2=regressor2.fit(X_train1,Y_train)
training_data_prediction1 =regressor2.predict(X_train1)
r2_train_2 = metrics.r2_score(Y_train, training_data_prediction1)
print('R squared vale for training data: ', r2_train_2)
# prediction on test data
test_data_prediction1 =regressor2.predict(X_test_1)
r2_test1 = metrics.r2_score(Y_test, test_data_prediction1)
print('R squared vale for testing data: ', r2_test1)

Adj_r2 = 1-(1-r2_train_2)*(1338-1)/(1338-3-1)
print("adjusted r2 for training data is ",Adj_r2)

#new_data={'age'=age1,'bmi'=bmi1,'smoker'=smoker1}

def prediction(age1,sex1,bmi1,children1,smoker1,region1):   
 
    # Pre-processing user input    
    if sex1 == "Male":
        sex1 = 0
    else:
        sex1 = 1
 
    if smoker1 == "yes":
        smoker1 = 0
    else:
        smoker1 = 1
 
    if region1 == "southeast":
        region1 = 0
    elif region1=='southwest':
        region1 = 1 
    elif region1=='northeast':
        region1=2
    else:
        region1=3
        
    predict1=regressor.predict([[age1,sex1,bmi1,children1,smoker1,region1]])
    
    return predict1



#st.write("the predicted charges for the insurance is {}",new_charges)
#print(new_charges)
#new_charges=prediction(age1,sex1,bmi1,children1,smoker1,region1)
if show1:
   if st.button("Predict"): 
       new_charges=prediction(age1,sex1,bmi1,children1,smoker1,region1)
        #result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
       st.success('Your charges are   {}'.format(new_charges))
       print(new_charges)
    
#print(format(new_charges))