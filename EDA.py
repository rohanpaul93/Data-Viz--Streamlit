# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:22:19 2021

@author: rohan
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image
import pandas as pd
import pydeck as pdk
import plotly.express as px
import altair as alt
import seaborn as sns
#sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud,STOPWORDS
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


def app():
    
    raw_data = pd.read_csv('AB_NYC_2019.csv')
    
    for char in raw_data.name.astype(str):
        if char in " ?.!/;:":
            raw_data['name']=raw_data['name'].replace(char,'')
            
    selected_data = raw_data[['id','host_id', 'neighbourhood_group', 'neighbourhood', 
              'latitude', 'longitude', 'room_type','price','minimum_nights','number_of_reviews']]
    
      
  
    col1,col2=st.beta_columns((1,1))
    
    col1.markdown('Map of NYC showing areas which have the most expensive listings')
    col1.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
    latitude=40.730610,
    longitude=-73.935242,
    zoom=8,
    pitch=30, 
    bearing=-27.36,
    pickable=False,
    ),
    layers=
    pdk.Layer(
    'HexagonLayer',
    data=selected_data,
    get_position='[longitude, latitude]',
    radius=200,
    auto_highlight=True,
    elevation_scale=4,
    elevation_range=[0, 1000],
    pickable=False,
    extruded=True,
    coverage=1,
    ),
    ))
    ##############################################################
    col2.markdown('Neighbhourhood groups vs avg price per night')
    
    fig1 = sns.barplot(x='neighbourhood_group', y='price', data=selected_data.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False).reset_index(),
    palette="Blues_d")
    
    sns.set(font_scale = 1.5)
    fig1.set_xlabel("District",fontsize=10)
    fig1.set_ylabel("Price ($)",fontsize=10)
    col2.pyplot()
    
    ##############################################################
    
    st.markdown("This shows the Relationship between Room type and neighbourhood")
                
    room_types_data = selected_data.groupby(['neighbourhood_group', 'room_type']).size().reset_index(name='Quantity')
    room_types_data = room_types_data.rename(columns={'neighbourhood_group': 'District', 'room_type':'Room Type'})
    room_types_data['Percentage'] = room_types_data.groupby(['District'])['Quantity'].apply(lambda x:100 * x / float(x.sum()))
  
   # sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize':(15,15)})
    #fig2 = sns.catplot(y='Percentage', x='District', hue="Room Type", data=room_types_data, height=6, kind="bar", palette="muted", ci=95,aspect=3);
    
    fig2 = px.bar(room_types_data, x="District", y="Percentage",
             color='Room Type', barmode='group')
 #   ,height=600,width=1500)
    
    #fig2.set(ylim=(0, 100))
    
    #for ax in fig2.axes.flat:
        #ax.yaxis.set_major_formatter(PercentFormatter(100))
    
    #fig2.update_layout(bgcolor='#0E1117')
    fig2.update_yaxes(showgrid=False)
    fig2.update_xaxes(showgrid=False)
    #fig2.layout.plot_bgcolor='0E1117'
    fig2.update_layout(plot_bgcolor= '#0E1117')
    
    
    st.plotly_chart(fig2,use_container_width=True)
    ###############################################################
   
    
    ######################################################################
    st.markdown("This shows the clusters of listings in New York City")
    Lat=40.80
    Long=-73.80

    locations = list(zip(raw_data.latitude, raw_data.longitude))

    map1 = folium.Map(location=[Lat,Long], zoom_start=11)
    FastMarkerCluster(data=locations).add_to(map1)
    folium_static(map1,width=1370)
    
    col3,col4=st.beta_columns((1,1))
    
    ######################################################################
    #TOP 10 HOSTS with theirs % contri of listings
    ### Ranking host ids according to most number of listings
    col3.markdown('Top 10 hosts ranked by their listings')
    Ids =     {
    219517861:1,
    107434423:2,
    30283594:3,
    137358866:4,
    12243051:5,
    16098958:6,
    61391963: 7,
    22541573:8,
    200380610:9,
    7503643:10
    }
    
    raw_data['host_id_new'] = raw_data['host_id'].map(Ids)
    
    
    grouped_Data = raw_data['host_id_new'].value_counts().reset_index().head(10)
    grouped_Data.columns =['host_id_new','listing_counts']
    fig3 = px.pie(grouped_Data, values='listing_counts', names='host_id_new')
    col3.plotly_chart(fig3)
    
    ##################################################################
    #SunBurst
    col4.markdown('Relation between Super hosts vs neighbourhood vs Number of reviews')
    gk = raw_data.groupby('host_id')[['number_of_reviews']].sum().reset_index()
    gk =  gk.loc[gk['host_id'].isin([219517861,107434423,30283594,137358866,12243051,16098958,61391963,22541573,200380610,7503643])]   
    
    gk['host_id_new'] = gk['host_id'].map(Ids)
    merged_Data =grouped_Data.merge(raw_data[['neighbourhood','number_of_reviews','host_id_new']],how='left',on='host_id_new')
    
    merged_Data = merged_Data.groupby(['host_id_new','neighbourhood'])[['number_of_reviews']].sum().reset_index()
    merged_Data['number_of_reviews'] = np.where(merged_Data['number_of_reviews']==0,1,merged_Data['number_of_reviews'])
    
    fig4 = px.sunburst(merged_Data, path=['host_id_new', 'neighbourhood'], values='number_of_reviews', color='neighbourhood',color_continuous_scale='sunsetdark',hover_name='number_of_reviews')
    col4.plotly_chart(fig4)
    
    ######################################################################
    col4,col5=st.beta_columns((1,1))
    # create the bins
    col4.markdown("Listing Price distribution")
    counts, bins = np.histogram(raw_data.price, bins=range(0, 200, 5))
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig5 = px.bar(x=bins, y=counts, labels={'x':'price_per_night', 'y':'count'})
    fig5.update_yaxes(showgrid=False)
    fig5.update_xaxes(showgrid=False)
    #fig2.layout.plot_bgcolor='0E1117'
    fig5.update_layout(plot_bgcolor= '#0E1117')
    col4.plotly_chart(fig5)
    #######################################################################
    
    col5.markdown("Minimum nights vs Price ")
    raw_data['minimum_nights'] = np.where(raw_data['minimum_nights']>30,30,raw_data['minimum_nights'])
    raw_data['price'] = np.where(raw_data['price']>1000,1000,raw_data['price'])
    fig6 = px.scatter(raw_data, x="minimum_nights", y="price")
    fig6.update_yaxes(showgrid=False)
    fig6.update_xaxes(showgrid=False)
    #fig2.layout.plot_bgcolor='0E1117'
    fig6.update_layout(plot_bgcolor= '#0E1117')
    col5.plotly_chart(fig6)
    
    #####################################################################
    st.markdown('')
    
    def make_wordcloud(words):

        text = ""
        for word in words.astype(str):
            text = text  + word
    
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords,colormap="plasma",width=1950, height=1090,max_font_size=200, max_words=500, background_color="black").generate(text)
        #plt.figure(figsize=(3,3))
        plt.imshow(wordcloud, interpolation="gaussian")
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot()
        
       
    make_wordcloud(raw_data['name'])   
    