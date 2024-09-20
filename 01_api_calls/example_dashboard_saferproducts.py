import pandas as pd
import streamlit as st
import urllib.request
import json

"""
## Dashboard Template
 
This simple dashboard provides tables from the saferproducts.gov API.

- `remedy` is how consumers were compensated for the recall
- `mnf_country` is the place where the product originated

We focus on products in which the word "gas" appeared in the Recall Title.
"""

url = 'https://www.saferproducts.gov/RestWebServices/Recall'
#query = '?format=json&ProductType=Phone' #29
#query = '?format=json&ProductType=Grill' #70
#query = '?format=json&ProductType=Exercise' # 91
query = '?format=json&RecallTitle=Gas' # 216

response = urllib.request.urlopen(url+query)
response_bytes = response.read()
data = json.loads(response_bytes)
response.close() 

df = pd.DataFrame.from_dict(data)
print(df.shape)

df.head()

temp = df['RemedyOptions']
clean_values = []
for i in range(len(temp)):
    if len(temp[i])==1 :
        clean_values.append( str(temp[i][0]['Option']) )
    elif len(temp[i])>1:
        countries = []
        for j in range(len(temp[i])):
            countries.append( temp[i][j]['Option'] )
        clean_values.append( str(countries) )
    else:
        clean_values.append('')
df['mnf_country'] = clean_values
st.write(df['mnf_country'].value_counts())

temp = df['RemedyOptions']
clean_values = []
for i in range(len(temp)):
    if len(temp[i])>0:
        values = []
        for j in range(len(temp[i])):
            values.append(temp[i][j]['Option'] )
        clean_values.append(values[0])
    else:
        clean_values.append('')
df['remedy'] = clean_values
st.write(df['remedy'].value_counts())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
xtab = pd.crosstab( df['remedy'],df['mnf_country'] )

# Create streamlit output:
st.write(xtab)