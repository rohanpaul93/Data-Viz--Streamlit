# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:10:42 2021

@author: rohan
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from multiapp import MultiApp
import EDA, model

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, 'C:/Users/rohan/Desktop/GWU/spring 21/Sem 1/Data Visualization/Final_Project')
#import EDA,model


st.set_page_config(layout='wide')
app = MultiApp()


######################
# Page Title
######################


st.markdown("<h1 style='text-align: center; color: grey;'>NYC Airbnb Analysis Web App</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey;'>This app shows exploratory and predictive analysis of the data set</h1>", unsafe_allow_html=True)

# About
expander_bar = st.beta_expander("About")
expander_bar.markdown('''
* **Python libraries:** numpy, pandas, matplotlib, streamlit, pydeck, lazypredict, sklearn, seaborn, plotly.express, altair, wordcloud, folium, streamlit_folium
* **Data source:**  [NYC airbnb data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).
* **References:**   
    * (https://medium.com/@lilyng15/data-analysis-on-airbnb-new-york-city-60bb85560a01)
    * (https://www.kaggle.com/richieone13/airbnb-new-york-eda-and-predictive-modelling)
    * (https://www.kaggle.com/saket7788/airbnb-nyc-regression-model)
    * (https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)
    * (https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction)
                     ''' )

image = Image.open('background.jpg')
st.image(image,use_column_width=True)
# Add all your application here

app.add_app("EDA", EDA.app)
app.add_app("Model", model.app)
# The main app
app.run()


