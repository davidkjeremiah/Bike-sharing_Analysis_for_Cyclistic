# Maximizing Annual Memberships for Cyclistic Bike-Share: A Data-Driven Approach

# Import the necessary libraries
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy import stats
from geopy.distance import geodesic

## Bikeshare Dataset list
# get list of bike_share .csv files in the 'bike_share' folder
bikeshare_list = os.listdir('bike_share') 

# print list of bike_share .csv files
print(bikeshare_list)

## Clean and prepare data
def clean_data(bikeshare_list):
    """
    Cleans the bikeshare data and saves the results to a new CSV file.
    """

    # specify the path to the folder containing raw bike_share files
    raw = 'bike_share'

    for index, df_path in enumerate(bikeshare_list):
        # read in data 
        try:
            df = pd.read_csv(os.path.join(raw, df_path))
            print(df.shape)
        except:
            print(f'Error reading {df_path}')
            break
            
        # remove duplicates    
        df.drop_duplicates(inplace=True)
        
        # fill missing values in start and end station columns with unknownwith 'unknown'
        df['start_station_name'].fillna('unknown', inplace=True)
        df['end_station_name'].fillna('unknown', inplace=True)
        
        # fill missing values in latitude and longitude columns with mean values
        df['start_lat'].fillna(df['start_lat'].mean(), inplace=True)
        df['start_lng'].fillna(df['start_lng'].mean(), inplace=True)
        df['end_lat'].fillna(df['end_lat'].mean(), inplace=True)
        df['end_lng'].fillna(df['end_lng'].mean(), inplace=True)
        
        # convert started_at and ended_at columns to datetime objects
        try:
            if (df['started_at'].dtypes != 'datetime64[ns]') & (df['ended_at'].dtypes != 'datetime64[ns]'):
                df['started_at'] = pd.to_datetime(df['started_at'], format='%Y-%m-%d %H:%M:%S')
                df['ended_at'] = pd.to_datetime(df['ended_at'], format='%Y-%m-%d %H:%M:%S')
            else:
                print(f'datetime columns in {df_path} are of correct datatype')
                continue
        except:
            print(f'Error converting datetime columns in {df_path}')
            break

        # specify the path to the folder
        folder_path = 'bike_share_clean'

        # create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # save the clean data to the specified folder
        df.to_csv(os.path.join(folder_path, df_path[:-4] + '-clean.csv'), index=False)

        # print success message
        print(f"{index}, {os.path.join(folder_path, df_path[:-4] + '-clean.csv')} is saved successfully!")

clean_data(bikeshare_list)


## Merge the dataset into one
# specify the path to the folder
folder_path = 'bike_share_clean'

# List all dataset in the 'bike_share_clean' folder for the twelve months
clean_data = os.listdir(folder_path)

# Load the first month dataset
df1 = pd.read_csv(os.path.join(folder_path, clean_data[0]))

# Load the second month dataset
df2 = pd.read_csv(os.path.join(folder_path, clean_data[1]))

# Load the third month dataset
df3 = pd.read_csv(os.path.join(folder_path, clean_data[2]))

# Load the fourth month dataset
df4 = pd.read_csv(os.path.join(folder_path, clean_data[3]))

# Load the fifth month dataset
df5 = pd.read_csv(os.path.join(folder_path, clean_data[4]))

# Load the sixth month dataset
df6 = pd.read_csv(os.path.join(folder_path, clean_data[5]))

# Load the seventh month dataset
df7 = pd.read_csv(os.path.join(folder_path, clean_data[6]))

# Load the eighth month dataset
df8 = pd.read_csv(os.path.join(folder_path, clean_data[7]))

# Load the nineth month dataset
df9 = pd.read_csv(os.path.join(folder_path, clean_data[8]))

# Load the tenth month dataset
df10 = pd.read_csv(os.path.join(folder_path, clean_data[9]))

# Load the eleventh month dataset
df11 = pd.read_csv(os.path.join(folder_path, clean_data[10]))

# Load the twelveth month dataset
df12 = pd.read_csv(os.path.join(folder_path, clean_data[11]))

# Unite the twelve datasets
df_union = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12])


## Validate that data is clean
### Check for missing values
# Check for missing values
missing = df_union.isnull().sum()
print(missing)

# Drop `start_station_id` and `end_station_id` columns
df_union.drop(['start_station_id', 'end_station_id'], axis=1, inplace=True)

### Check for duplicates
# Check for duplicates
duplicates = df_union.duplicated().sum()
print(duplicates)

### Check for correct datatype
# Inspect datatype of df_union
print(df_union.dtypes)

# convert started_at and ended_at columns to datetime objects
df_union['started_at'] = pd.to_datetime(df_union['started_at'], format='%Y-%m-%d %H:%M:%S')
df_union['ended_at'] = pd.to_datetime(df_union['ended_at'], format='%Y-%m-%d %H:%M:%S')

# Re-inspect datatype of df_union
print(df_union.dtypes)

### Check for data quality
# Check for date completeness (Jan - Dec 2021)
# for `started_at` and `ended_at`
completeness_s = df_union['started_at'].dt.month_name().unique()
completeness_e = df_union['ended_at'].dt.month_name().unique()

print(completeness_s)
print(completeness_e)


## Exploratory Data Analysis
### Q1. How do annual members and casual riders use Cyclistic bikes differently?

#### Duration of the ride and Stations:
"""We start by analyzing the `member_casual` column, comparing the usage patterns of annual members and casual riders by looking at the `started_at` and `ended_at` columns to see how long each group tends to take their rides, as well as the `start_station_name` and `end_station_name` columns to see which stations they tend to start and end their rides at."""

# Calculate the average ride duration for annual members and casual riders
annual_members_avg_duration = df_union[df_union['member_casual'] == 'member']['ended_at'].subtract(df_union[df_union['member_casual'] == 'member']['started_at']).mean()
casual_riders_avg_duration = df_union[df_union['member_casual'] == 'casual']['ended_at'].subtract(df_union[df_union['member_casual'] == 'casual']['started_at']).mean()

# Calculate the most popular start and end stations for annual members and casual riders
annual_members_start_stations = df_union[(df_union['member_casual'] == 'member') & (df_union['start_station_name'] != 'unknown')]['start_station_name'].value_counts().head(5)
annual_members_end_stations = df_union[(df_union['member_casual'] == 'member') & (df_union['end_station_name'] != 'unknown')]['end_station_name'].value_counts().head(5)
casual_riders_start_stations = df_union[(df_union['member_casual'] == 'casual') & (df_union['start_station_name'] != 'unknown')]['start_station_name'].value_counts().head(5)
casual_riders_end_stations = df_union[(df_union['member_casual'] == 'casual') & (df_union['end_station_name'] != 'unknown')]['end_station_name'].value_counts().head(5)

# Print the results
print(f"Annual members average ride duration: {annual_members_avg_duration}")
print(f"Casual riders average ride duration: {casual_riders_avg_duration}")
print(f"Annual members most popular start stations:\n{annual_members_start_stations}")

# Set the figure size
plt.figure(figsize=(10, 5))

# Plot the average ride duration for annual members and casual riders
sns.barplot(x=['Annual Members', 'Casual Riders'], y=[annual_members_avg_duration.total_seconds()/60, casual_riders_avg_duration.total_seconds()/60], palette=['#4C8BF5','#EA4335'])
plt.title('Average Ride Duration')
plt.xlabel('Member Type')
plt.ylabel('Duration (minutes)')
plt.show()

# Set the figure size
plt.figure(figsize=(10, 5))

# Plot the most popular start stations for annual members
sns.barplot(x=annual_members_start_stations.values, y=annual_members_start_stations.index, color='#4C8BF5')
plt.title('Most Popular Start Stations for Annual Members')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(10, 5))

# Plot the most popular start stations for casual riders
sns.barplot(x=casual_riders_start_stations.values, y=casual_riders_start_stations.index, color='#EA4335')
plt.title('Most Popular Start Stations for Casual Riders')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(10, 5))

# Plot the most popular end stations for annual members
sns.barplot(x=annual_members_end_stations.values, y=annual_members_end_stations.index, color='#4C8BF5')
plt.title('Most Popular End Stations for Annual Members')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(10, 5))

# Plot the most popular end stations for casual riders
sns.barplot(x=casual_riders_end_stations.values, y=casual_riders_end_stations.index, color='#EA4335')
plt.title('Most Popular End Stations for Casual Riders')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')

# Show the plot
plt.show()


#### Type of bike:
"""
Here we analyze the `rideable_type` column, to determine whether each ride was taken on an electric, classic or docked bike and compare the usage patterns of these three types of bikes among annual members and casual riders.

But first, let's confirm that these are the three types of bikes used by either Cyclistic annual members or casual riders from our dataset...
"""

# Types of bikes used
bike_type  = df_union['rideable_type'].unique()
print(bike_type)

# Create a pivot table to count the number of rides for each rideable type and membership status
pivot_table = df_union.pivot_table(index='rideable_type', columns='member_casual', values='ride_id', aggfunc='count')
print(pivot_table)

# Set figure
plt.figure(figsize=(8,6))

# Create a countplot
sns.countplot(x='rideable_type', hue='member_casual', palette=['#4C8BF5','#EA4335'], data=df_union)
plt.ylabel('Number of Rides')
plt.title('Number of Rides by Bike Type and Membership Status')

# Show the plot
plt.show()

# Calculate chi-squared statistic and p-value
chi2, p_value, _, _ = stats.chi2_contingency(pivot_table)
print(f"Chi-squared statistic: {chi2:.2f}, p-value: {p_value:.2f}")


#### Distance traveled:
"""
Now, we will explore how annual members and casual riders differ in terms of `distance traveled` using pandas and seaborn to visualize the data.
"""
# Convert the `lat` and `lng` to list and then to tuple
start = tuple(map(tuple, (df_union[['start_lat', 'start_lng']].values.tolist())))
end = tuple(map(tuple, (df_union[['end_lat', 'end_lng']].values.tolist())))

# Create an empty list to store the distances
distances = []

# Iterate over the start and end coordinates
for start_coord, end_coord in zip(start, end):
    # Calculate the geodesic distance between the start and end coordinates (i.e the distance traveled by each rider)
    distance = geodesic(start_coord, end_coord).km
    # Append the distance to the list
    distances.append(distance)

# Create a new column for `distance` traveled by each rider
df_union['distance'] = distances

# Create a subplot for annual members
plt.subplot(1, 2, 1)
sns.distplot(df_union[df_union['member_casual'] == 'annual']['distance'])
plt.xlabel('Distance Traveled (km)')
plt.ylabel('Density')
plt.title('Distance Traveled by Annual Members')

# Create a subplot for casual riders
plt.subplot(1, 2, 2)
sns.distplot(df_union[df_union['member_casual'] == 'casual']['distance'])
plt.xlabel('Distance Traveled (km)')
plt.ylabel('Density')
plt.title('Distance Traveled by Casual Riders')

plt.tight_layout()
plt.show()


#### Time of day:
"""
Here we analyze how annual members and casual riders use Cyclist's bike differently based on the time of day. 

We will also explore the average ride duration for annual members and casual riders at different time of day.
"""
# Extract the hour from the started_at column and create a new column for the hour
df_union['hour'] = df_union['started_at'].dt.hour

# Calculate the average number of rides per hour for annual members and casual riders
annual_members_rides_per_hour = df_union[df_union['member_casual'] == 'member']['hour'].value_counts().sort_index() / df_union[df_union['member_casual'] == 'member'].shape[0]
casual_riders_rides_per_hour = df_union[df_union['member_casual'] == 'casual']['hour'].value_counts().sort_index() / df_union[df_union['member_casual'] == 'casual'].shape[0]

# Plot the average number of rides per hour for annual members and casual riders
plt.plot(annual_members_rides_per_hour.index, annual_members_rides_per_hour.values, label='Annual Members', color='#4C8BF5')
plt.plot(casual_riders_rides_per_hour.index, casual_riders_rides_per_hour.values, label='Casual Riders', color='#EA4335')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Rides')
plt.title('Average Number of Rides per Hour')
plt.legend()
plt.show()

# Calculate duration and create duration column in dataframe
df_union['duration'] = (df_union['ended_at'] - df_union['started_at']) / np.timedelta64(1, 'm')

# Calculate the average ride duration for annual members and casual riders at different times of the day
annual_members_duration_by_hour = df_union[df_union['member_casual'] == 'member'].groupby('hour')['duration'].mean()
casual_riders_duration_by_hour = df_union[df_union['member_casual'] == 'casual'].groupby('hour')['duration'].mean()

# Plot the average ride duration by hour for annual members and casual riders
plt.plot(annual_members_duration_by_hour, label='Annual Members', color='#4C8BF5')
plt.plot(casual_riders_duration_by_hour, label='Casual Riders', color='#EA4335')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Ride Duration (minutes)')
plt.legend()
plt.show()


#### Day of the week
"""
Here we compare usage patterns of annual members and casual riders on different days of the week (e.g. weekdays vs. weekends)
"""
# Set figure
plt.figure(figsize=(10,6))

# Create a new column for the day of the week
df_union['day_of_week'] = df_union['started_at'].dt.day_name()

# Plot the distribution of riders by day of the week and rider type
sns.countplot(data=df_union, x='day_of_week', hue='member_casual', palette=['#4C8BF5','#EA4335'])

# Add labels and show the plot
plt.xlabel('Day of the Week')
plt.ylabel('Number of Rides')
plt.title('Rides by Day of the Week and rider type')
plt.show()


#### Season:
"""
Here, we compare the usage patterns of annual members and casual riders during different seasons (e.g. summer vs. winter)
"""
# Set figure
plt.figure(figsize=(10,6))

# Create a new column for the season based on the month of the 'started_at' column
df_union['season'] = pd.DatetimeIndex(df_union['started_at']).month
df_union['season'] = df_union['season'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')

# Plot the count of rides by season for annual members and casual riders
sns.countplot(x='season', hue='member_casual', data=df_union, hue_order=['member', 'casual'], order=['Winter', 'Spring', 'Summer', 'Fall'], palette=['#4C8BF5','#EA4335'])
plt.title('Number of Rides by Season')
plt.xlabel('Season')
plt.ylabel('Number of Rides')
plt.show()



## Q2: Why would casual riders buy Cyclistic annual memberships?
"""
Based on the insights above, it is possible that casual riders may consider purchasing an annual membership for the following reasons:
"""

#### Cost savings: 
"""
If a casual rider uses the bike-sharing service frequently, an annual membership may offer a lower per-ride cost compared to paying for each ride individually.
"""
# Test hypotheses
from scipy import stats

# Hypothesis 1: Riders who take longer rides are more likely to purchase annual memberships
# Extract the ride duration data for casual riders and annual members
casual_ride_duration = df_union[df_union['member_casual'] == 'casual']['duration'].dropna()
annual_ride_duration = df_union[df_union['member_casual'] == 'member']['duration'].dropna()

# Perform a t-test to compare the mean ride duration of casual riders and annual members
t_statistic, p_value = stats.ttest_ind(casual_ride_duration, annual_ride_duration)

# Print the t-statistic and p-value
print(t_statistic)
print(p_value)

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant difference in the mean ride duration between casual riders and annual members.")
else:
    print("There is no statistically significant difference in the mean ride duration between casual riders and annual members.")


#### Convenience: 
"""
An annual membership may allow a casual rider to easily access the bike-sharing service without having to constantly purchase individual rides or keep track of payment methods.
"""
# Hypothesis 2: Casual riders are more likely to purchase annual memberships for convinence purposes if they use the service more frequently
# Extract the number of rides data for casual riders and annual members
casual_rides = df_union[df_union['member_casual'] == 'casual']['ride_id'].count()
annual_rides = df_union[df_union['member_casual'] == 'member']['ride_id'].count()

# Perform a chi-squared test to compare the number of rides for casual riders and annual members
chi2_statistic, p_value, _, _ = stats.chi2_contingency([[casual_rides, annual_rides]])

# Print the chi-squared statistic and p-value
print(chi2_statistic)
print(p_value)

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant difference in the number of rides between casual riders and annual members.")
else:
    print("There is no statistically significant difference in the number of rides between casual riders and annual members.")


#### Access to electric bikes: 
"""
Annual members have access to electric bikes, which may be more appealing to casual riders who prefer electric bikes, or who use the bike-sharing service for longer rides.
"""
# Hypothesis 3: Access to electric bikes
# Extract the rideable type data for casual riders and annual members
casual_rideable_type = df_union[df_union['member_casual'] == 'casual']['rideable_type']
annual_rideable_type = df_union[df_union['member_casual'] == 'member']['rideable_type']

# Count the number of electric bike rides for each group
casual_electric_rides = casual_rideable_type[casual_rideable_type == 'electric_bike'].value_counts()
annual_electric_rides = annual_rideable_type[annual_rideable_type == 'electric_bike'].value_counts()

# Create a contingency table from the count data
contingency_table = pd.DataFrame({'casual': casual_electric_rides, 'annual': annual_electric_rides}).unstack()
print(contingency_table)

# Perform a Chi-squared test to compare the proportion of electric bike rides between the two groups
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the Chi-squared statistic and p-value
print(chi2)
print(p_value)

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant difference in the proportion of electric bike rides between casual riders and annual members.")
else:
    print("There is no statistically significant difference in the proportion of electric bike rides between casual riders and annual members.")


#### Flexibility

# Hypothesis 4: Annual members have more flexibility in terms of bike usage
# Create a contingency table to count the number of rides for each group
contingency_table = pd.DataFrame({'casual': casual_rideable_type.value_counts(), 'annual': annual_rideable_type.value_counts()}).unstack()
print(contingency_table)

# Perform a chi-squared test to compare the distribution of bike types used by casual riders and annual members
chi_squared_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

# Print the chi-squared statistic and p-value
print(chi_squared_stat)
print(p_value)

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant difference in the distribution of bike types used by casual riders and annual members.")
else:
    print("There is no statistically significant difference in the distribution of bike types used by casual riders and annual members.")


#### Ability to use the service for commuting: 
"""
Based on the insights above, annual members tend to use the bike-sharing service for commuting purposes, and a casual rider may see the value in purchasing an annual membership in order to use the service for their daily commute.
"""
# Hypothesis 5: Annual members use the bike-sharing service more for commuting purposes

# Extract the start time data for casual riders and annual members
casual_start_time = df_union[df_union['member_casual'] == 'casual']['started_at']
annual_start_time = df_union[df_union['member_casual'] == 'member']['started_at']

# Convert the start time data to hour of the day
casual_start_hour = casual_start_time.dt.hour
annual_start_hour = annual_start_time.dt.hour

# Reset the index of the two DataFrames
casual_start_hour = casual_start_hour.reset_index(drop=True)
annual_start_hour = annual_start_hour.reset_index(drop=True)

# Create a contingency table to count the number of rides for each group
contingency_table = pd.crosstab(casual_start_hour, annual_start_hour)
print(contingency_table)

# Perform a chi-squared test to compare the distribution of start times between the two groups
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the Chi-squared statistic and p-value
print(chi2)
print(p_value)

# Interpret the results
if p_value < 0.05:
    print("There is a statistically significant difference in the distribution of start times between casual riders and annual members.")
else:
    print("There is no statistically significant difference in the distribution of start times between casual riders and annual members.")


## Question 3: How can Cyclistic use digital media to influence casual riders to become members?
"""
This question is asking about the potential use of digital media (such as social media, email, or online ads) to persuade casual riders to purchase annual memberships with Cyclistic.
"""
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Create a new column for digital media usage
df_union['digital_media'] = np.where((df_union['start_lat'] > 41.8) & (df_union['start_lng'] < -87.6), 1, 0)

# Create the features and labels
X = df_union[['duration', 'rideable_type', 'digital_media']]
y = df_union['member_casual']

# Encode the 'rideable_type' column
le = LabelEncoder()
X['rideable_type'] = le.fit_transform(X['rideable_type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Calculate the AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.2f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='member')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()