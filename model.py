# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:08:13 2021

@author: rohan
"""
import streamlit as st

import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

def app():
    # Sidebar - Specify parameter settings
    with st.sidebar.header('Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

    st.markdown("<h1 style='text-align: center; color: grey;'>The Machine Learning Algorithm Comparison App</h1>", unsafe_allow_html=True)
    
           
    raw_data = pd.read_csv('AB_NYC_2019.csv')
    
    for char in raw_data.name.astype(str):
        if char in " ?.!/;:":
            raw_data['name']=raw_data['name'].replace(char,'')
            
    selected_data = raw_data[['id','host_id', 'neighbourhood_group', 'neighbourhood', 
              'latitude', 'longitude', 'room_type','minimum_nights','number_of_reviews','availability_365','calculated_host_listings_count','price']]
    
    st.subheader('1. Dataset')
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(selected_data)
    
    ## This is to display the info about the data
    stats = []
    for col in selected_data.columns:
        stats.append((col, selected_data[col].nunique(), selected_data[col].isnull().sum() * 100 / selected_data.shape[0], selected_data[col].value_counts(normalize=True, dropna=False).values[0] * 100, selected_data[col].dtype))
    
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=False)
    
    st.markdown('**1.2. Displaying information about the data**')
    st.write(stats_df)
    
    st.markdown('**1.3. Dataset dimension**')
    st.write('X')
    st.info(selected_data.shape)
    st.write('Y')
    st.info(selected_data['price'].shape)
    
    st.markdown('**1.3. Variable details**:')
    st.write('X variables')
    
    st.info(list(selected_data.columns))
    st.write('Y variable')
    st.info(selected_data['price'].name)
    
    #Encode the input Variables
    def Encode(airbnb):
        for column in selected_data.columns[selected_data.columns.isin(['neighbourhood_group','neighbourhood', 'room_type'])]:
            selected_data[column] = selected_data[column].factorize()[0]
        return airbnb

    selected_data_en = Encode(selected_data.copy())
    
    ## plotting the correaltion heatmp
    st.markdown('**1.4 Correlation Heatmap**')
    corr = selected_data_en[['neighbourhood_group', 'neighbourhood','room_type', 'minimum_nights', 'number_of_reviews', 'availability_365', 'calculated_host_listings_count']].corr(method='kendall')
    plt.figure(figsize=(15,8))
    sns.heatmap(corr, annot=True)
    st.pyplot()
    
    ## sampling the data
    selected_data_en = selected_data_en.sample(frac =.10)
    ###Building the model
    X = selected_data_en[['neighbourhood_group', 'neighbourhood','room_type', 'minimum_nights', 'number_of_reviews', 'availability_365', 'calculated_host_listings_count']]
    X = X.iloc[:,:-1]
    Y = selected_data_en.iloc[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    
    st.subheader('2. Table of Model Performance')

    st.write(predictions_test)
    


    st.subheader('3. Plot of Model Performance (Test set)')
    with st.markdown('**R-squared**'):
        # Tall
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
        ax1.set(xlim=(0, 1))
   
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
   

    with st.markdown('**RMSE (capped at 100)**'):
        # Tall
        predictions_test["RMSE"] = [100 if i > 100 else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
  
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
  

    with st.markdown('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
 
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
   


    
    