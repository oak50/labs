
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import requests # Page requests

from bs4 import BeautifulSoup as soup # HTML parser
import re # Regular expressions
import time # Time delays
import random # Random numbers


#header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'} 
#url = 'https://charlottesville.craigslist.org/search/cta?purveyor=owner#search=1~gallery~0~0' 
#raw = requests.get(url,headers=header) # Get page
     
# QUESTION 1
# using craigslist page with vintage toys
# Would collect info about the types of boats on the market. EDA analysis could include
# finding the most popular buzzwords/ boat types sold, the distribution of prices, and the distribution of years
# of origin for boats sold

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'} 
url = 'https://charlottesville.craigslist.org/search/boo#search=1~gallery~0~0' 
raw = requests.get(url,headers=header) # Get page

# QUESTION 2
# use beautiful soup to extract data

from bs4 import BeautifulSoup as soup # HTML parser
bsObj = soup(raw.content,'html.parser') # Parse the html
listings = bsObj.find_all(class_="cl-static-search-result") # Find all listings

import re # Regular expressions

brands = ['yamaha', "honda", "mariah", "trophy"] # collecting text data about boat types

bsObj = soup(raw.content,'html.parser') # Parse the html
listings = bsObj.find_all(class_="cl-static-search-result") # Find all listings

data = [] # We'll save our listings in this object
for k in range( len(listings) ):
    title = listings[k].find('div',class_='title').get_text().lower()
    price = listings[k].find('div',class_='price').get_text()
    link = listings[k].find(href=True)['href']
    # Get brand from the title string:
    words = title.split()
    hits = [word for word in words if word in brands] # Find brands in the title
    if len(hits) == 0:
        brand = 'missing'
    else:
        brand = hits[0]
    # Get years from title string:
    regex_search = re.search(r'20[0-9][0-9]|19[0-9][0-9]', title ) # Find year references
    if regex_search is None: # If no hits, record year as missing value
        year = np.nan 
    else: # If hits, record year as first match
        year = regex_search.group(0)
    #
    data.append({'title':title,'price':price,'year':year,'link':link,'brand':brand})

# QUESTION 3

## Wrangle the data
df = pd.DataFrame.from_dict(data)
df['price'] = df['price'].str.replace('$','', regex=False)
df['price'] = df['price'].str.replace(',','', regex=False)
df['price'] = pd.to_numeric(df['price'],errors='coerce')
df['year'] = pd.to_numeric(df['year'],errors='coerce')
df['age'] = 2025-df['year']
print(df.shape)
df.to_csv('craigslist_cville_boats.csv') # Save data in case of a disaster
df.head()

# EDA for price and age:

# create histogram for distribution of product price
print(df['price'].describe())
df['price'].hist(grid=False)
plt.show()
plt.savefig( "price_hist.png")

# create histogram for distribution of product year
import matplotlib.pyplot as plt
print(df['year'].describe())  # Display descriptive statistics
df['year'].hist(bins=10, grid=False)
plt.show()
plt.savefig("year_hist.png")

# Price by brand chart:
print("price by brand: \n", df.loc[:,['price','brand']].groupby('brand').describe() )

# Year by brand chart:
print("year by brand: \n", df.loc[:,['year','brand']].groupby('brand').describe() )

# scatterplot: year and price by brand
plt.figure(figsize=(12, 6)) 
ax = sns.scatterplot(data=df, x='year', y='price', hue='brand')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlim(0, 50)  # Set x-axis range from 0 to 20,000
plt.show()
plt.savefig("year_price_boatscatter")

# There are very few points in this graph, with only a few brands represented. However, it appears that 
# there is a negative correlation between the age and price of boats, regardless of brand.

df['log_price'] = np.log(df['price']) # apply log function to normalize the distribution / reduce impact of outliers
df['log_year'] = np.log(df['year'])

ax = sns.scatterplot(data=df, x='log_year', y='log_price',hue='brand')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()
plt.savefig("year_price_boatscatter_log") # applying the log function had a minimal effect on the scatterplot

print("Covariance of price and year: \n", df.loc[:,['log_price','log_year']].cov())

# the covariance between log_age and log_price is -0.584. This indicates that as age moves positively, 
# price moves negatively

print("Correlation between price and year: \n", df.loc[:,['log_price','log_year']].corr())
# the correlation between price and age is -0.892. This indicates that for a 1% increase in boat age corresponds to an 
# average decrease in price of 0.89%.

# create jointplot of age and price
sns.jointplot(data=df, x='log_year', y='log_price',kind='hex')
plt.show()
plt.savefig("year_price_joint")

# the hexogonal bins on this plot are very large and few, which indicates underfitting. It's difficult 
# to make any assumptions from this graph.

# create kde plot for price by brand
sns.kdeplot(data=df,x='price',hue='brand')
plt.xlim(0, 10)  # Change x-axis scale
plt.ylim(0, 15)
plt.show()
plt.savefig("kde_price_brand")

# create dictionary for brand / price pairs and print it out:
for index, row in df.iterrows():
    print(f"Brand: {row['brand']}, Price: {row['price']}")
# you can see here a large proportion of missing values for brand. Should refine scraping technique to get more accurate info.

# look at descriptive statistics tables for boats with missing brand name & compare to boats that do have a listed brand name:
missing_brand_df = df[df['brand'] == 'missing'] # Get descriptive statistics for filtered items
stats = missing_brand_df.describe() # Print the descriptive statistics
print( "descriptive statistics for boats in dataset which are missing a brand name:" )
print(stats)

non_missing_brand_df = df[df['brand'] != 'missing'] # Get descriptive statistics for filtered items
stats = non_missing_brand_df.describe() # Print the descriptive statistics
print( "descriptive statistics for boats in data not missing brand name:" )
print(stats)

median_missing = missing_brand_df['price'].median()
median_non_missing = non_missing_brand_df['price'].median()

# Calculate the percentage difference
percentage_difference = ((median_non_missing - median_missing) / median_missing) * 100

# Print the results
print(f"Median price for items with missing brand: {median_missing}")
print(f"Median price for items without missing brand: {median_non_missing}")
print(f"Percentage difference: {percentage_difference:.2f}%")

# from this, we see that boats in the dataset which are missing a brand name have a median price of $999
# conversely, boats in the dataset which have brand name included have a median price of $10500
# Boats with brand names listed are priced 951% higher than brands without brands names, on average.
# this could possibly indicate that including brand name in Craiglist descriptions typically raises price
# However, potential biases include the fact that smaller boats (such as kayaks) are more likely to be listed without a brand name,
# which drives down the price median for non-brand boats.

# OVERALL FINDINGS:
# Overall, it was difficult to find consistent patterns in the data due to a lack of points in the dataset.
# However, the plots lead me to infer the following:
    
    # age and price are negatively correlated for boats on the marketplace (according to the scatterplot), which means that
    # people selling newer boats are more likely to price them high (this is also intutitive).

    # Trophy boats had the highest median value at $19000, followed by Honda at $14000. However, the lack of data
    # makes it difficult to infer anything from these statistics.

    # There were 8 boats with age info and 36 boats with price info, according to the tables on price and age. However, 
    # as shown in the brand/price list at the end of the section, many brand names are missing from the data.

    # Boats with a brand name listed were found to have a 951% higher median price than boats without a brand name.



# QUESTION 4:

import time # Time delays
import random # Random numbers

links = df['link']
data = []
for link in links: # about 3 minutes
    time.sleep(random.randint(1, 3)) # Random delays
    raw = requests.get(link,headers=header) # Get page
    bsObj = soup(raw.content,'html.parser') # Parse the html
    #
    try:
        year_post = bsObj.find(class_='attr important').find(class_ = 'valu year').get_text()
    except:
        year_post = np.nan
    #
    try:
        condition = bsObj.find(class_='attr condition').find(href=True).get_text()
    except:
        condition = 'missing'
    #
    try:
        cylinders = bsObj.find(class_='attr auto_cylinders').find(class_ = 'valu').get_text()
        cylinders = cylinders.replace('\n','')
    except:
        cylinders = 'missing'
    #
    try:
        drivetrain = bsObj.find(class_='attr auto_drivetrain').find(href=True).get_text()
    except:
        drivetrain = 'missing'
    #
    try:
        fuel = bsObj.find(class_='attr auto_fuel_type').find(href = True).get_text()
    except:
        fuel = 'missing'
    #
    try:
        miles = bsObj.find(class_='attr auto_miles').find(class_ = 'valu').get_text()
    except:
        miles = np.nan
    #
    try:
        color = bsObj.find(class_='attr auto_paint').find(href=True).get_text()
    except:
        color='missing'
    #
    try:
        title = bsObj.find(class_='attr auto_title_status').find(href=True).get_text()
    except:
        title='missing'
    #
    try:
        transmission = bsObj.find(class_='attr auto_transmission').find(href=True).get_text()
    except:
        transmission = 'missing'
    #
    try:
        bodytype = bsObj.find(class_='attr auto_bodytype').find(href=True).get_text()
    except:
        bodytype = 'missing'
    #
    text = bsObj.find(id='postingbody').get_text()
    text = text.replace('\n','')
    text = text.replace('QR Code Link to This Post','')
    record = {'title':title,
              'name':bodytype,
              'text':text,}
    data.append(record)

    new_df = pd.DataFrame.from_dict(data)
new_df.head()

df = pd.concat([df,new_df],axis=1) # combine data frames
print( df.head() )
print(df.columns.tolist())

df['age'] = pd.to_numeric(df['age'],errors='coerce')

df['year'] = pd.to_numeric(df['year'],errors='coerce')
df.to_csv('craiglist_cville_boats_long.csv')

ax = sns.scatterplot(data=df, x='age', y='price',hue='brand')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()
plt.savefig( "age_price_brand_new.png" )