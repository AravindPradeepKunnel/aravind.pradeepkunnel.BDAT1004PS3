#!/usr/bin/env python
# coding: utf-8

# # Question1 

# ### OCCUPATION

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")


# In[4]:


import requests

url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

for file in files:
    print(f"Downloading {file}...")
    response = requests.get(url + file, stream=True)
    with open(file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"{file} downloaded.")

print("Download complete.")


# In[8]:


#Assigning it to variable called users
users = pd.read_csv(r"C:\Users\Aravind\Downloads\user.csv")


# In[9]:


users.head()


# In[10]:


#Step 4 Discover what is the mean age per occupation
age = users.groupby(['occupation'])['age'].mean()
print(age)


# In[11]:


#Step 5 For each occupation, calculate the minimum and maximum ages

users[users['gender']=='M'].groupby('occupation')['gender'].value_counts().sort_values(ascending = False)
#users[users['gender']=='M'].groupby('occupation')['gender'].count().sort_values(ascending=False)


# In[12]:


#Step 6 For each occupation, calculate the minimum and maximum ages

df = users.groupby(['occupation']).agg({'age':['min', 'max']})
print(df)
print('\n\n')

min_age = users.groupby(['occupation'])['age'].min()
max_age = users.groupby(['occupation'])['age'].max()
df1 = pd.merge(min_age, max_age, how='inner',on='occupation')
print(df1)


# In[13]:


#Step 7 For each combination of occupation and sex, calculate the mean age
users.groupby(['occupation','gender'])['age'].mean()


# In[14]:


#Step 8 What is the number of columns in the dataset?
import math
gender = users.groupby(['occupation','gender'])['gender'].count()
percentage = gender.groupby(level=0).apply(lambda x: round(100 * x / x.sum(),2))
percentage


# # Question 2

# ### Euro teams

# In[15]:


#Step 1
import pandas as pd
import numpy as np


# In[21]:


#Step 2 and 3
euro12 = pd.read_csv(r"C:\Users\Aravind\Downloads\euro_stats.csv")


# In[22]:


euro12.head()


# In[23]:


#Step 4 What is the number of columns in the dataset?
pd.DataFrame(euro12['Goals'])


# In[24]:


#Step 5 What is the number of columns in the dataset?
teams = euro12['Team'].count()
print("Total number of teams participated in Euro12 are: ", teams)


# In[25]:


#Step 6 What is the number of columns in the dataset?
print("Total number of columns in euro stats are:", euro12.shape[1])


# In[26]:


#Step 7 View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline


# In[27]:


#Step 8  Sort the teams by Red Cards, then to Yellow Cards
discipline.sort_values(["Red Cards", "Yellow Cards"])


# In[28]:


# Step 9. Calculate the mean Yellow Cards given per Team
discipline.groupby('Team')['Yellow Cards'].mean()


# In[29]:


# Step 10. Filter teams that scored more than 6 goals 
euro12[euro12['Goals'] > 6]


# In[30]:


# Step 11. Select the teams that start with G
euro12[euro12['Team'].str[0].isin(['G'])]

#euro12[euro12.Team.str.startswith('G')]["Team"]


# In[31]:


#Step 12. Select the first 7 columns
euro12.iloc[0:7]


# In[32]:


# Step 13. Select all columns except the last 3
euro12.iloc[:, :32]


# In[33]:


# Step 14. Present only the Shooting Accuracy from England, Italy and Russia
teams = euro12[(euro12.Team == "England") | (euro12.Team == "Italy") | (euro12.Team == "Russia")]
accuracy = teams[["Team","Shooting Accuracy"]]
accuracy


# # Question 3

# ### Housing

# In[34]:


# Step 1. Import the necessary libraries
import numpy as np
import pandas as pd


# In[35]:


# Step 2. Create 3 differents Series, each of length 100, as follows:
# The first a random number from 1 to 4
# The second a random number from 1 to 3
# The third a random number from 10,000 to 30,000

series1 = pd.Series(np.random.randint(1,4, size=100))
series2 =  pd.Series(np.random.randint(1,3, size=100))
series3 =  pd.Series(np.random.randint(10000,30000, size=100))

print(series1, '\n\n', series2, '\n\n', series3)


# In[36]:


# Step 3. Create a DataFrame by joinning the Series by column
dataFrame1 = pd.concat([series1, series2, series3], axis = 1)
dataFrame1


# In[37]:


# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
df1 = dataFrame1.set_axis(['bedrs', 'bathrs', 'price_sqr_meter'], axis=1, inplace=False)
df1


# In[38]:


# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'
bigcolumn = pd.concat([series1,series2,series3],axis=0)
column = bigcolumn.to_frame()
column


# In[39]:


#Step 6. Ops it seems it is going only until index 99. Is it true?


# Yes, we the index at end is 99 because the pythin indexing starts from 0. SO for 100 number, the indecing is 99. (0-100)

# In[40]:


# Step 7. Reindex the DataFrame so it goes from 0 to 299
bigcolumn.index = range(300)
bigcolumn


# # Question 4

# ### Wind Statistics

# In[146]:


# Step 1. Import the necessary libraries
# Step 2. Import the dataset from the attached file wind.txt
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.
import pandas as pd
import numpy as np
from io import StringIO

data = pd.read_csv(r"C:\Users\Aravind\Downloads\wind.csv")


# In[147]:


data.head()


# In[148]:


import datetime

date = data.apply(lambda x: datetime.date(int(x['Yr']+1900), int(x['Mo']), int(x['Dy'])),axis=1)

# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.
# Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns]

date = date.astype('datetime64[ns]')
data = data.drop(columns=['Yr', 'Mo', 'Dy'])
data.insert(0, 'Date', date)
data.set_index(['Date'], inplace = True)
data.head()


# In[151]:


# Step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below.

null = data.isnull().sum()
print("Number of Null Values are:\n", null)

print('\n\n')


# In[152]:


#Step 7. Compute how many non-missing values there are in total.
notnull = data.notnull().sum()
print("Number of non null values are:\n", notnull)


# In[154]:


data = data.dropna()
data


# In[156]:


# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
data.mean().mean()


# In[157]:


#Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations 
#of the windspeeds at each location over all the days. A different set of numbers for each location.

#loc_stats = data.iloc[:, 1:]
#loc_stats = pd.concat([locs.min(), locs.max(), locs.mean(), locs.std()], axis = 1)
#loc_stats = loc_stats.rename({0: "Minimum", 1 : "Maximum", 2 : "Mean", 3 : "Standard Deviation"}, axis = 1)

loc_stats = data.describe().T
loc_stats


# In[158]:


# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the
# windspeeds across all the locations at each day.

day_stats = pd.DataFrame()

day_stats['Min values']=data.min(axis=1)
day_stats['Max values']=data.max(axis=1)
day_stats['Average']=data.mean(axis=1)
day_stats['Standard Deviation']=data.std(axis=1)
day_stats


# In[159]:


#Step 11. Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January.

WindSpeedInJan = data[data.index.month == 1]
WindSpeedInJan.mean()


# In[160]:


# Step 12. Downsample the record to a yearly frequency for each location
data.resample('Y').mean()


# In[161]:


# Step 13. Downsample the record to a monthly frequency for each location.
data.resample('m').mean()


# In[162]:


# Step 14. Downsample the record to a weekly frequency for each location.

data.resample('W').mean()


# In[163]:


# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations 
# for each week (assume that the first week starts on January 2 1961) for the first 52 weeks

weeks = data.resample('W').agg(['min','max','mean','std'])
weeks.head(5).T


# ## Question 5

# In[64]:


import pandas as pd
import numpy as np

chipo = pd.read_csv(r"C:\Users\Aravind\Downloads\chipotle.csv")


# In[65]:


chipo


# In[66]:


# Step 4. See the first 10 entries
chipo.head(10)


# In[67]:


# Step 5. What is the number of observations in the dataset?
chipo.size


# In[68]:


# Step 6. What is the number of columns in the dataset?
chipo.shape


# In[69]:


chipo.columns.size


# In[70]:


chipo.info()


# In[71]:


# Step 7. Print the name of all the columns

for col in chipo.columns:
    print(col)


# In[72]:


chipo.columns


# In[73]:


# Step 8. How is the dataset indexed?
chipo.index


# In[74]:


# Step 9. Which was the most-ordered item?
# Step 10. For the most-ordered item, how many items were ordered
chipo.item_name.value_counts()


# ### Chicken Bowl is the most ordered item and order count is 726

# In[75]:


# Step 11. What was the most ordered item in the choice_description column?
choice_description = chipo.choice_description.value_counts()
choice_description[:1]


# In[76]:


# Step 12. How many items were orderd in total?
chipo.quantity.sum()


# ## Step 13. 
# ####        • Turn the item price into a float 
# ####        • Check the item price type 
# ####        • Create a lambda function and change the type of item price 
# ####        • Check the item price type 

# In[77]:


print(chipo.dtypes)


# In[78]:


chipo["item_price"] = chipo["item_price"].apply(lambda x: float(x[1:]))
chipo["item_price"].dtypes


# In[79]:


# Step 14. How much was the revenue for the period in the dataset?
chipo['total_revenue'] = chipo["quantity"] * chipo["item_price"]
total_revenue = chipo.total_revenue.sum()
print("Total Revenue is $", total_revenue)


# In[80]:


# Step 15. How many orders were made in the period?
len(chipo['order_id'].unique())


# In[81]:


# Step 16. What is the average revenue amount per order?
chipo.groupby("order_id")["total_revenue"].mean()


# In[82]:


# Step 17. How many different items are sold?
print("Total unique items: ", len(chipo["item_name"].unique()))


# ### Question 6
# 

# Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. Label both lines and show the legend

# In[83]:


import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt  
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


marriage_divorces = pd.read_csv(r"C:\Users\Aravind\Downloads\us-marriages-divorces.csv")


# In[85]:


marriage_divorces


# In[86]:


marriage_divorces.plot.line(x = "Year", y = ["Marriages_per_1000", "Divorces_per_1000"],figsize = (10,8))
plt.title("Number of Marriages and Divorces per capita in the U.S between 1867 and 2014")
plt.xlabel("Year")
plt.ylabel("Marriage/Divorce")
plt.show()


# ## Question 7

# Create a vertical bar chart comparing the number of marriages and divorces per capita in the U.S. between 1900, 1950, and 2000.

# In[87]:


us_data = marriage_divorces[(marriage_divorces.Year == 1900) | (marriage_divorces.Year == 1950) | (marriage_divorces.Year == 2000)]
us_data = us_data.drop(columns = ['Marriages', 'Divorces', 'Population'])
us_data = us_data.set_index('Year')

# sns.barplot(data = us_data, x = 'Year', y = ["Marriages_per_1000", "Divorces_per_1000"])
us_data.plot.bar(figsize = (8, 5))

plt.xlabel('Years')
plt.ylabel('Number of Marriage & Divorces')
plt.title('US Marriages & Divorces Data')


# ### Question 8

# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort the actors by their kill count and label each bar with the corresponding actor's name.

# In[88]:


import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt  
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


actors = pd.read_csv(r"C:\Users\Aravind\Downloads\actor_kill_counts.csv")


# In[90]:


actors


# In[91]:


new = actors.sort_values(by = "Count", ascending = False)
plt.figure(figsize = (7,6))
sns.barplot(data = new, y = "Actor", x = "Count", orient = 'h')
plt.title("Kill count of Hollywood Actors")


# # Question 9

# Create a pie chart showing the fraction of all Roman Emperors that were assassinated. Make sure that the pie chart is an even circle, labels the categories, and shows the percentage breakdown of the categories.

# In[92]:


roman_emporers = pd.read_csv(r"C:\Users\Aravind\Downloads\roman-emperor-reigns.csv")


# In[93]:


roman_emporers


# In[94]:


assasinated = roman_emporers[roman_emporers['Cause_of_Death'] == "Assassinated"]
plt.figure(figsize =(11, 7))

colors = sns.color_palette('pastel')
plt.pie([len(assasinated),len(roman_emporers) - len(assasinated)], labels = ['Assassinated', 'Deaths by other cause'],
        startangle = 90, colors = colors, autopct='%.0f%%')
plt.show()


# # Question 10

# Create a scatter plot showing the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009.

# In[95]:


df = pd.read_csv(r"C:\Users\Aravind\Downloads\arcade-revenue-vs-cs-doctorates.csv")


# In[96]:


df


# In[97]:


plt.figure(figsize = (8,6))
sns.scatterplot(data=df, x='Total Arcade Revenue (billions)', y='Computer Science Doctorates Awarded (US)', 
                hue='Year').set_title('Total Arcade Revenue and Computer Science Doctorates Awarded (US) between 2000 and 2009.\n')


# In[ ]:




