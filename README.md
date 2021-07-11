# Disaster Response Pipeline Project

## Summary of the project-
This project is aimed to classify real time messages sent by people stuck in unprecedented situations with the help of Machine Learning.
This project has 3 parts:
1.	Using ETL(Extract, Transform, Load) the dataset is read, cleaned, processed and then stored in SQLite Database.
2.	Splitting the data into training and test set and creating Machine Learning Pipeline with the help of which messaged could be classified in different categories.
3.	Displaying the results in a Flask Web App.

## Dependencies-
1.	The code should run with no issues using Python versions 3.*
2.	pandas
3.	numpy
4.	SciPy
5.	sklearn
6.	NLP library- NLTK
7.	To load the data into an SQLite database- SQLAlchemy engine
8.	Model Loading and Saving Library: Pickle
9.	Web App: Flask



## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Files in the repository:
1.	In the Data folder:  
-	disaster_categories.csv and disaster_messages.csv (Given dataset)
-	DisasterResponse.db: created database from transformed and cleaned data containing one table named “Messages_Categories”.
-	Process_data.py:  reads in the data, cleans and saves it in a SQL database

2.	In the Models folder:  
-	Train_classifier.py: Loads Data from dataset, tokenizes to parse text data to the model, use Pipeline API to setup feature vectors and classifier, evaluating learned model     on test set and Saving trained model to disk
-	Classifier.pkl: Saved trained classifier.


3.	App folder:
-	run.py: Flask app and the user interface used to predict results and display them.
-	templates: folder containing the html templates

## Screenshots
This is the front page with 2 visulizations

![Alt text](https://github.com/surabhi-dubey/Disaster_response_pipeline/blob/master/Screenshot1.PNG?raw=true "Screenshot1")
![Alt text](https://github.com/surabhi-dubey/Disaster_response_pipeline/blob/master/Visualisation2.PNG?raw=true "Visualisation2")

By inputting a word, you can check its category

![Alt text](https://github.com/surabhi-dubey/Disaster_response_pipeline/blob/master/Screenshot2.PNG?raw=true "Screenshot2")

### Classification report

![Alt text](https://github.com/surabhi-dubey/Disaster_response_pipeline/blob/master/classification1.PNG?raw=true "Screenshot3")

![Alt text](https://github.com/surabhi-dubey/Disaster_response_pipeline/blob/master/classification2.PNG?raw=true "Screenshot4")

## Authors

Surabhi Dubey

## Acknowledgements

Figure Eight for providing the datasets to train the model
