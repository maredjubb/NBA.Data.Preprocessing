import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# Preprocess the data to prepare it for use in a machine learning model that predicts the salaries of NBA players.

#Handle missing values, remove extraneous characters, and parse the features.
def clean_data(path):
    #Reading file (needs to be in csv)
    df = pd.read_csv(path)

    #Parse the b_day and draft_year features as datetime objects
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')

    #Replace the missing values in team feature with "No Team"
    df.fillna({'team':"No Team"}, inplace=True)

    #Take the height feature in meters, the height feature contains metric and customary units;
    df['height'] = df['height'].apply(lambda x: x.split('/')[1])

    #Take the weight feature in kg, the weight feature contains metric and customary units
    df['weight'] = df['weight'].apply(lambda x: x.split('/')[1].split('kg')[0])

    #Remove the extraneous $ character from the salary feature;
    df['salary'] = df['salary'].apply(lambda x: x.split('$')[1])

    #Parse the height, weight, and salary features as floats
    df['height'] = df['height'].astype(float)
    df['weight'] = df['weight'].astype(float)
    df['salary'] = df['salary'].astype(float)

    #Categorize the country feature as "USA" and "Not-USA"
    df['country'] = df['country'].apply(lambda x: "USA" if x == 'USA' else "Not-USA")

    #Replace the cells containing "Undrafted" in the draft_round feature with the string "0"
    df['draft_round'] = df['draft_round'].apply(lambda x: "0" if x == "Undrafted" else x)

    #Return the dataframe
    return df

#Create new numerical features out of the existing ones and deal with high cardinality.
def feature_data(df):

    #Get the unique values in the version column of the DataFrame you got from clean_data as a year
    unique_versions = df.version.unique()
    years = []
    #Extract the years from unique version
    for v in unique_versions:
        year_suffix = v[-2:]
        full_year = "20" + year_suffix
        years.append(full_year)

    #Parse as a datetime object
    date_time_years = pd.to_datetime(years, format='%Y')
    df['version'] = df['version'].apply(lambda x: date_time_years[0] if x == 'NBA2k20' else date_time_years[1])

    #Engineer the age feature by subtracting b_day column from version. Calculate the value as year
    df['age'] = df['version'].dt.year - df['b_day'].dt.year

    #Engineer the experience feature by subtracting draft_year column from version. Calculate the value as year
    df['experience'] = df['version'].dt.year - df['draft_year'].dt.year

    #Engineer the bmi (body mass index) feature from weight (w) and height (h) columns
    df['bmi'] = df['weight'] / (df['height']**2)

    #Drop the version, b_day, draft_year, weight, and height columns
    df = df.drop(columns = ['version', 'b_day', 'draft_year', 'weight', 'height'])

    #Remove the high cardinality features
    column_list = df.columns.tolist()
    columns_to_drop = [cols for cols in column_list if df[cols].nunique() >= 50 and cols != 'bmi' and cols != 'salary']
    df = df.drop(columns = columns_to_drop)

    return df

#Drop the multicollinear features by observing the correlation coefficients.
def multicol_data(df):
    #Create a correlation matrix with all the values I want to test
    correlation_df = df[['rating','age','experience','bmi']]
    correlation_df = correlation_df.corr()

    #Get the absolute values of the correlation matrix
    correlation_df.abs()

    #Convert the diagonal will Null values
    np.fill_diagonal(correlation_df.values, np.nan)

    #print correlation_df to view correlation values
    print(correlation_df)

    #Take the two features with high correlation (age and salary) and compare with target variable salary
    columns_to_correlate = ['age', 'experience', 'salary']
    salary_correlation_df = df[columns_to_correlate].corr()
    print(salary_correlation_df)

    #Drop the column with the lowest correlation with salary (age)
    df = df.drop(columns = 'age')

    return df

#Apply the transformation techniques to numerical and categorical features.
def transform_data(df):
    #Seperate the numerical and categorical features
    num_feat_df = df.select_dtypes('number')  # numerical features
    cat_feat_df = df.select_dtypes('object')  # categorical features

    num_feat_df = num_feat_df.drop('salary', axis = 1) # Drop salary axis as this won't be transformed

    #Transform numerical features using StandardScaler and convert back to dataframe
    scaler_std = StandardScaler()
    num_std_df = scaler_std.fit_transform(num_feat_df)
    num_std_df = pd.DataFrame(num_std_df)

    # Transform nominal categorical variables in the DataFrame using OneHotEncoder and convert back to dataframe
    encoder = OneHotEncoder(sparse_output=False)
    cat_std_df = encoder.fit_transform(cat_feat_df)
    cat_std_df = pd.DataFrame(cat_std_df)

    #Concatenate the transformed numerical and categorical features
    final_df = pd.concat([num_std_df, cat_std_df], axis=1)
    print(df.columns)
    target_variable = df['salary']

    # Return two objects: X, where all the features are stored, and y with the target variable.
    return final_df, target_variable

df_cleaned = clean_data(data_path)
df_featured = feature_data(df_cleaned)
df_multicol = multicol_data(df_featured)
X, y = transform_data(df_multicol)
