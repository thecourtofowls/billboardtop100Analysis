#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Get Billboard Data
import bs4
import requests 
from time import sleep
import pandas as pd


# In[2]:


#Billboard Data
#Year 1970 to 1991 from billboard
bill_data_all_1 = pd.DataFrame(columns = ['date','rank','songs'])
start_date = 1970
while start_date < 1991:
    page = requests.get('https://www.billboard.com/charts/year-end/'+str(start_date)+'/hot-100-songs')
    soup = bs4.BeautifulSoup(page.content,'html.parser')
    rank = [x.get_text().strip('\n') for x in soup.find_all(class_ = 'ye-chart-item__rank')]
    songs = [x.get_text().strip('\n') for x in soup.find_all(class_='ye-chart-item__title')]
    artist = [x.get_text().strip('\n') for x in soup.find_all(class_="ye-chart-item__artist")]
    d = {'date':str(start_date),'rank': rank, 'songs': songs, 'artist':artist}
    bill_data_all_1 = bill_data_all_1.append(pd.DataFrame(d), ignore_index=True)
    sleep(0)
    start_date += 1

#Year 1991 to 2005 from wikipedia
bill_data_all_2 = pd.DataFrame(columns = ['date', 'rank', 'songs', 'artist'])
start_date = 1991
while start_date < 2006:
    page = requests.get('https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'+str(start_date))
    soup = bs4.BeautifulSoup(page.content,'html.parser')
    data = pd.DataFrame(columns = ['date', 'rank','songs','artist'])
    table = soup.find('table')
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if cols != []:
            d = {'date': str(start_date), 'rank': int(cols[0]), 'songs': cols[1],'artist': cols[2]}
            bill_data_all_2  = bill_data_all_2.append(pd.DataFrame(d, index=[0]), ignore_index=True)
        else:
            pass
    sleep(0.0001)
    start_date += 1


#Year 2006 to 2020 from billboard
bill_data_all_3 = pd.DataFrame(columns = ['date','rank','songs'])
start_date = 2006
while start_date < 2021:
    sleep(0.5)
    page = requests.get('https://www.billboard.com/charts/year-end/'+str(start_date)+'/hot-100-songs')
    soup = bs4.BeautifulSoup(page.content,'html.parser')
    rank = [x.get_text().strip('\n') for x in soup.find_all(class_ = 'ye-chart-item__rank')]
    songs = [x.get_text().strip('\n') for x in soup.find_all(class_='ye-chart-item__title')]
    artist = [x.get_text().strip('\n') for x in soup.find_all(class_="ye-chart-item__artist")]
    d = {'date':str(start_date),'rank': rank, 'songs': songs, 'artist':artist}
    bill_data_all_3 = bill_data_all_3.append(pd.DataFrame(d), ignore_index=True)
    start_date += 1


# In[5]:


#Data Concat
bill_all_data = pd.concat([bill_data_all_1, bill_data_all_2, bill_data_all_3], axis=0)
bill_all_data.to_csv('bill_all_data.csv', index = False)
bill_all_data.shape


# In[ ]:





# # Billboard Data

# In[46]:


#Year 1970 to 2011 from wikipedia
bill_data_all_w1 = pd.DataFrame(columns = ['date', 'rank', 'songs', 'artist'])
start_date = 1970
while start_date < 2011:
    page = requests.get('https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_'+str(start_date))
    soup = bs4.BeautifulSoup(page.content,'html.parser')
    data = pd.DataFrame(columns = ['date', 'rank','songs','artist'])
    table = soup.find('table')
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if cols != []:
            d = {'date': str(start_date), 'rank': cols[0], 'songs': cols[1],'artist': cols[2]}
            bill_data_all_w1 = bill_data_all_w1.append(pd.DataFrame(d, index=[0]), ignore_index=True)
        else:
            pass
    sleep(0.0001)
    start_date += 1


# In[47]:


#Year 2012 to 2020 from billboard
bill_data_all_w2 = pd.DataFrame(columns = ['date','rank','songs'])
start_date = 2011
while start_date < 2021:
    sleep(0.5)
    page = requests.get('https://www.billboard.com/charts/year-end/'+str(start_date)+'/hot-100-songs')
    soup = bs4.BeautifulSoup(page.content,'html.parser')
    rank = [x.get_text().strip('\n') for x in soup.find_all(class_ = 'ye-chart-item__rank')]
    songs = [x.get_text().strip('\n') for x in soup.find_all(class_='ye-chart-item__title')]
    artist = [x.get_text().strip('\n') for x in soup.find_all(class_="ye-chart-item__artist")]
    d = {'date':str(start_date),'rank': rank, 'songs': songs, 'artist':artist}
    bill_data_all_w2 = bill_data_all_w2.append(pd.DataFrame(d), ignore_index=True)
    start_date += 1


# In[48]:


#Data Concat
bill_all_data_wb = pd.concat([bill_data_all_w1, bill_data_all_w2], axis=0)
bill_all_data_wb.to_csv('bill_all_data_wb.csv', index = False)
bill_all_data_wb.shape


# # Spotify
bill_all_data_wb = bill_all_data_wb[['date', 'artist', 'songs']]
bill_all_data_wb.sort_values(by = 'date', inplace=True)
all_data = bill_all_data_wb.copy()
#authentication
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
cid = 'Your Spotify Client ID'
secret = 'secret'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
all_data.head()

#get song artist ID for audio feature extraction
all_id = []
for i in range(len(all_data)):
    try:
        m = sp.search(
            q='artist:' + all_data.iloc[i,1] + ' track:' + all_data.iloc[i,2], 
            type='track')['tracks']['items'][0]['id']
        all_id.append(m)
        sleep(0.0005)
    except IndexError:
        m = 0
        all_id.append(m)

#new id column
all_data['id'] = all_id
all_data.head()
#filter out song-artist with missing id information
all_data = all_data.loc[all_data['id'] != 0] #Get only those with id
#Get audio features
all_data['features'] = all_data['id'].apply( lambda x: sp.audio_features(x))
all_data_2 = pd.concat([all_data, all_data["features"].apply(lambda x: pd.Series(x[0]))], axis=1).drop("features", axis=1)
all_data_2.to_csv("all_data.csv", index=False) #Write to csv file

import pandas as pd


# In[2]:


all_data_2 = pd.read_csv("all_data.csv")


# In[72]:


# Data Description 


# In[3]:


desc_data = all_data_2[['date','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy()


# In[4]:


desc_data.describe()


# In[75]:


bb = pd.DataFrame(data=desc_data.describe().T)
bb.to_csv("descriptionData.csv")


# In[ ]:





# # Result Selection

# In[5]:

#Corr Analysis
columns_num = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
               'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
new_data = all_data_2[columns_num].copy()
#libraris
from sklearn import preprocessing
scaler_var = preprocessing.MinMaxScaler()
new_data_norm = scaler_var.fit_transform(new_data)
new_data_norm = pd.DataFrame(new_data_norm)
new_data_norm.columns = new_data.columns
#corr
pd.DataFrame(new_data_norm).corr()
#plot
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")
# Compute the correlation matrix
corr = pd.DataFrame(new_data_norm).corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 11))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns_plot_corr = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .99})
ax.figure.savefig("audio_corr.png")
sns_plot_corr


# In[ ]:





# # Data Description

# In[9]:


viz_data = all_data_2[['date', 'danceability', 'energy','loudness', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy()
viz_data.head()
#Average
viz_data_mean = viz_data.groupby(['date']).mean()
viz_data_mean = viz_data_mean.reset_index()
viz_data_mean.head()
#plot
import pandas as pd 
import numpy as np 
import plotly_express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')
import psutil

#Average
viz_data_m2 = viz_data_mean[['date', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                             'instrumentalness', 'liveness', 'valence', 'tempo',  'duration_ms']].copy()
viz_data_m2 = viz_data_m2.set_index('date')
viz_data_m_stack = viz_data_m2.stack()
viz_data_m_stack = viz_data_m_stack.reset_index()
viz_data_m_stack = viz_data_m_stack.rename(columns={'level_1':'feature',0:'value'})
viz_data_m_stack.head()

fig = px.line(viz_data_m_stack[(viz_data_m_stack['feature'] != 'tempo') & (viz_data_m_stack['feature'] != 'loudness') & 
                                 (viz_data_m_stack['feature'] !=  'duration_ms')], 
                 x = 'date', y='value', color = 'feature',title='', log_y=False)
fig.update_layout(hovermode='closest', template='seaborn', width=1000, height = 600,
                  xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=True),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[16]:


fig = px.line(viz_data_m_stack[(viz_data_m_stack['feature'] == 'tempo')], 
                 x = 'date', y='value', color = 'feature',title='Tempo', log_y=False)
fig.update_layout(hovermode='closest', template='seaborn', width=1000, height = 600,
                  xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=True),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'),
                 showlegend = False)
fig.show()


# In[17]:

#LOudness
fig = px.line(viz_data_m_stack[(viz_data_m_stack['feature'] == 'loudness')], 
                 x = 'date', y='value', color = 'feature',title='Loudness', log_y=False)
fig.update_layout(hovermode='closest', template='seaborn', width=1000, height = 600,
                  xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=True),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'),
                 showlegend = False)
fig.show()


# In[18]:

#Duration
fig = px.line(viz_data_m_stack[(viz_data_m_stack['feature'] ==  'duration_ms')], 
                 x = 'date', y='value', color = 'feature',title='Duration (ms)', log_y=False)
fig.update_layout(hovermode='closest', template='seaborn', width=1000, height = 600,
                  xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=True),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'),
                 showlegend = False)
fig.show()


#Hypothesis Test
import pandas as pd
stat_data = pd.read_csv(r'all_data.csv')
stat_data = stat_data[['date', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy()
#stat_data = stat_data.loc[stat_data['date'] != 2020]
stat_data['date'].unique()
#Average
stat_data_mean = stat_data.groupby(['date']).mean()
stat_data_mean = stat_data_mean.reset_index()
#set index
stat_data_mean['date'] = stat_data_mean['date'].apply(lambda x:str(x) + '-'+'12' + '-' + '31' )
stat_data_mean['date'] = pd.to_datetime(stat_data_mean['date'])
stat_data_mean = stat_data_mean.set_index('date')
stat_data_mean.head()
# In[8]:
import numpy as np
import pandas as pd
import pyhomogeneity as hg
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
#empty dataframe
stat_data_hyp = pd.DataFrame(columns = ['feature','h', 'change_point', 'p_val', 'U', 'avg'])
#pettitt test
for i in stat_data_mean.columns:
    m = hg.pettitt_test(stat_data_mean[[i]], alpha=0.05)
    d = {'feature': i ,'h': m.h, 'change_point': m.cp, 'p_val': m.p, 'U': m.U, 'avg': m.avg}
    stat_data_hyp = stat_data_hyp.append(pd.DataFrame(d), ignore_index=True)


stat_data_hyp['avgs'] = ['mu_before','mu_after'] * (len(stat_data_mean.columns))
stat_data_hyp
#Reshape data
stat_data_hyp = stat_data_hyp.pivot_table(
    index = ['feature','h', 'change_point', 'p_val', 'U'], 
    columns = ['avgs'], values = 'avg' ).reset_index()

stat_data_hyp
stat_data_hyp.to_csv("hyp.csv")
