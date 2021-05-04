# Disaster_Response_Pipeline
## Description
This project is part of the **Udacity Data Scientist Nanodegree**: Disaster Response Pipeline Project.
The main goal of this project is to build a model that can help emergency workers classify incoming messages and sort them into specific categories and they can easily analyze the messages to predict the disasters messages.

## Install 
 - NumPy
 - Pandas
 - Sklearn
 - Nltk
 - Json
 - Plotly
 - Flask
 - Sqlalchemy
 - Sys
 - Re
 - Pickle

## Files 
- process_data.py: This code extracts data from both CSV files: messages.csv and categories.csv and creates an SQLite database containing a merged and cleaned version of this data.

- train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

- run.py: contains the visualization code and the connection with the web page.

- disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.

- templates folder: This folder contains all of the files necessary to run and render the web app. 

## Instructions: 
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Acknowledgements
This app was completed as part of the **Udacity Data Scientist Nanodegree**. 
Code templates and data were provided by Udacity, the data was originally sourced by **Udacity** from [Figure Eight](https://www.figure-eight.com/)
