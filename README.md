# Dividend-Policy-Forecast
This is the github repository for the freelance project - Dividend Policy Forecast

# Success Metrics
In Progress
The understanding of the group is that the success metrics/KPI of this project should be set not only in terms of technical performance or model accuracies, but in business contexts as well. Considerations must be given to the overall economy, including and not limited to: budgets, scalability, etc.

# Project Objectives
The goal of this project is to build and optimize a machine learning model that would predict if a particular company would cut its dividend shares at a particular time. 

# Project Scope
All companies within S&P500 index.

# Data to be used
TBA

# Project Pipeline (Subject to Change)
![image](https://github.com/PeiqiAlan/Dividend-Policy-Forecast/assets/59779308/ca8a05c8-bc4d-48d9-bcf7-f562bb6a4fdb)

The project is still in very early stages and the team is actively researching on possible indicators, data features and APIs.

The first stage would be data mining and web scrapping. Data is expected to come from a variety of sources in heterogeneous forms: Strutured and Unstructured data, NoSQL, .xml formats, .csv and json. This would refer to the "Data Sources #" as you would observe on the left-most side of the image.

The second stage would be data extraction, cleaning, transformation processes, or an ETL process would suffice. The idea is that eventually the data need to be trained on machine learning models (quite possibly classical models such as regressions, clustering given the financial context of this project). Therefore such procedure is necessary.

Coming out of the second stage, the data would be stored in a database.

The third stage is the official data pipeline, which consists of data preprocessing, feature engineering and model training and tuning parts. Preprocessing would likely include EDA (exploratory data analysis) and feature selections to select the most relevant features, whethere statistically or financially given the qualitative nature of dividend policy decisions. Considering the highly human-oriented nature of dividend policy decisions, new features will be engineered based on financial and accounting principles. Then models will be trained and optimized to select the best performance possible. 

The conclusions&outputs will be given to the clients. A feedback loop is in place so that if the client wishes to predict the policy changes of new companies, relevant data will be inputed as well.

