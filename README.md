# udacity-disasterpipeline

# Disaster Response Pipeline Project
This repository contains the ready-to-use code for the Disaster Response Project of the Udacity Data Science Nano-Degree.

## What is this project about ? 

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 

## Datasets
The project uses disaster_categories.csv and disaster_messages.csv, that can be found under data folder

## File Structure

Following is the description of files in this repository

- app
  - template
  - master.html  # main page of web app
  - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

## Web App Example
<img width="1440" alt="Screenshot 2021-04-07 at 21 43 48" src="https://user-images.githubusercontent.com/44292641/113917812-f774e100-97e1-11eb-80dd-a4cf0efd6d18.png">

### Instructions for Deployment:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Technologies Used

- Flask : Hosts the web app
- SQLAlchemy : Creates the SQLite database 
- Jupyter Notebook : Scripts for ML Pipeline was created by using this

