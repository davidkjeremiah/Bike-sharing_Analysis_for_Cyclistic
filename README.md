# Project Title

Bike-Sharing Usage Analysis for Cyclistic: Understanding Casual Riders and Annual Members

## Background

The goal of this project is to understand the usage patterns of bike-sharing services by casual riders and annual members. The project uses data provided by Cyclistic, a fictional bike-sharing company, to perform statistical analysis and generate insights that can inform the company's marketing and sales strategies.

## Data

The data used in this project includes information on bike usage, such as `start` and `end stations`, ride duration (which can be siphoned from the `started_at` and `ended_at` features), and `bike` or `rideable type`. It also includes information on the riders, such as whether they are casual or annual members. The data used in this project covers a period of one year (2021) and contains over 1.1 million records.

## Methodology

The data was cleaned and wrangled to make it ready for analysis. This was followed by exploratory and statistical analysis of the data, involving the use of `chi-squared tests`, `t-tests`, and `machine learning models`. The data was visualized using various plots and charts to make it easy to understand.

## Result

The analysis revealed several key insights about the usage patterns of bike-sharing services by casual riders and annual members. It was found that annual members tend to use the service for more utilitarian trips such as commuting to work or school, running errands (like dropping off a package), or going to the store (e.g. to the grocery store to pick up a few items or to the library to borrow a book), while casual riders tend to take longer rides likely for tourism or recreational purposes - with more of such rides during the weekends (saturdays and sundays).

Additionally, it was found that casual riders who take longer rides may be more likely to purchase annual memberships. 

The result also indicate that there is a statistically significant difference in the distribution of start times between casual riders and annual members, which suggests that a casual rider may see the value in purchasing an annual membership in order to use the service for their daily commute.

## Recommendations

The insights generated from the analysis were used to make recommendations for Cyclistic's marketing and sales strategies. To influence casual riders to become members, the company may wish to consider the following `recommendations`:

1. `Target marketing efforts towards casual riders who take longer rides:` Based on the analysis of ride duration, it appears that casual riders who take longer rides may be more likely to purchase annual memberships. The marketing team could consider targeting these users with promotional offers or messaging that highlights the benefits of an annual membership, such as discounted rates or access to a larger network of bikes.

2. `Promote the use of the bike-sharing service for commuting:` The statistical analysis showed that there is a statistically significant difference in the distribution of start times between casual riders and annual members, which suggests that casual riders may see the value in purchasing an annual membership in order to use the service for their daily commute. The marketing team could consider promoting the bike-sharing service as a convenient and cost-effective way to commute to work or school.

3. `Leverage digital media to reach potential annual members:` The machine learning model showed that digital media usage is a strong predictor of annual membership status. The marketing team could consider using digital media platforms (such as social media, email marketing, or targeted online ads) to reach potential annual members and promote the benefits of an annual membership.

4. `Consider offering incentives or discounts for annual memberships:` The analysis did not find a statistically significant difference in the distribution of bike types used by casual riders and annual members, or in the convenience or flexibility of bike usage. The marketing team could consider offering incentives or discounts for annual memberships in order to increase the attractiveness of the annual membership option to casual riders.

5. `Use machine learning to personalize marketing efforts:` The machine learning model was able to predict annual membership status with an AUC score of 0.71, indicating that it is relatively effective at identifying potential annual members. The marketing team could consider using the model (or building a similar model) to personalize marketing efforts and target promotional messages to users who are more likely to purchase annual memberships.

## Installation

The code in this project was written in Python using libraries such as `pandas`, `numpy`, `matplotlib` and `scikit-learn`. To run the code, you will need to have these libraries installed on your machine. You can install them by running the following command in your command line:

`pip install -r requirements.txt`

You also need to have a dataset to run the code, you can download the dataset from this link: https://divvy-tripdata.s3.amazonaws.com/index.html for the year 2021, or you can use your own dataset as long as it follows the same structure and column names as the provided dataset.

Once you have the dataset and the necessary libraries installed, you can run the code by executing the `main.py` file. This will run the entire analysis, including data cleaning, feature engineering, hypothesis testing, and model building.

The results of the analysis, including the charts, tables, and statistical tests, can be found in the `results` folder. Additionally, the presentation of the project will be in the `PPT folder`, the model performance results and outputs are the `Model outputs` folder and the code for running the entire project can be found in the `src` folder.

## Note

The project is for educational purposes only and the data are not the actual data used by the bike-sharing company in consideration, and the insights and recommendations presented here are made based on the analysis of the data. 
