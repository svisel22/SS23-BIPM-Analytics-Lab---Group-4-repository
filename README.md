# Welcome to sentiment analaysis for Bayer04 Leverkusen news repository

## Introduction

### Our Mission
We aim to understand and tap into the emotions of football fans, attract sponsors by showcasing the true essence of the game, and provide valuable insights for talent scouting.

### Understanding Fan Emotions
We use sentiment analysis techniques to dive deep into public football news and discover what fans truly feel. By understanding their thoughts and desires, we help clubs form stronger connections with their supporters, building fan attraction and loyalty that goes beyond the pitch.
We use clustering techniques to understand different fan sentiments and identify common themes among them. This way, we can give Bayer04 more detailed insights into what their fans love and what they're concerned about.
 

## Content

### Folder structure
In this repository there are the folders archive, data_files, data_gathering_preprocessing, sentiment_analysis and text_clustering. 
Additionally there are the README.md file, which explains this repository, and the utils.py file, which inlcudes all functions. 

### archive
This folder includes old versions of the data and model files. This is not relevant for grading. Rather it documents development steps and experiments, which were stopped troughtout the process, for Bayer04 in case they want to move forward with this project.

### data_files
data_files includes all the csvs that are generated and used in the files of this repository and the dashboard:
* all_data_v3.csv is generated using the update_data file in data_gathering_preprocessing. It was last updated on the 30th of May.
* match_info_players.csv is a csv provided by Bayer04 to us. It is used to visualize the match dates and results in HEX.
This folder is structured with three subfolders: 
* data_clean includes all the files that were generated by the preprocessing files. It includes the folder labeled-data which contains the preprocessed data which was then manually labeled (see next paragraph). 
* data_clustering includes all the files that were generated troughout the clustering process #ACTION: Kevin
* data_sentiment contains the file that includes the results from the multilingual sentiment model for all three languages.

#### Labeled data
As the original data retrieved from the API and parsed hmtls wasn't labeled, this was done manually. 
For sentiment per language 30 datapoints were labeled. The goal was to have an as much balanced support as possible to make the evaluation comparable. This was achieved for sentiment for the German and Spanish data. The English data didn't include enough negativ datapoints, which is why only four datapoints could be labeled negative and therefore 13 were labeled as positive and 13 as neutral. 
For text clustering only the German data was labeled. Accidentally the topic Bundesliga-News was labled with one more datapoint than Europa-League and and Situation & Match Performance. Which leads to 31 labeled datapoints in total for text clustering.

### data_gathering_preprocessing
This folder contains the update_data file which is used to extract the relevant urls using the API and parsing the htmls using BeautifulSoup4. 
The three other files are used for preprocessing. As there are a number of preprocessing steps there is one file per language. Also there is additonal preprocessing for the text clustering that is exclusively performed in the German preprocessing file.

### sentiment_anlysis
There are two files in this folder: The file with the multingual bert model that is used for the dashboard in HEX and the file for the other sentiment models per language that were explored. Both files include the evaluation of the performance per model. 

### text_clustering
This folder includes the word_frequencies file that calculates absolute frequencies, as requested by Bayer04, per language and per player. 
The main subject of this folder is adressed in the text_clustering file, which uses kMeans and shows how the topics were named manually. It examines cluster dirstributions and prediction probabilities per cluster, tests the accuracy per cluster with the manually labeled data and creates wordclouds. This file also includes the LDA model and explains why it wasn't chosen for the dashboard in HEX. 

### Generated files (HEX)

**all_data_v3.csv**

This file contains the original data pulled from the urls, it is not yet cleaned. We use it in HEX to show the original data we gather form the beggining.

**data_sentiment_final.csv**

This file contains the cleaned and processed data, including a sentiment score and label. With information on players, languages, dates, sentiment, and other relevant details, it serves as the basis for dynamic dashboards in HEX. These dashboards enable users to explore the average sentiment per player, period of time, and language.

**word_frequencies.csv**

This file contains the absolute frequency of words per player and language. It is used in HEX to create dashboards that highlight the most frequent words used by each player and language.

**match_info_players.csv**

This file was provided by Bayer04 and contains match information, including the opposing team, number of goals scored, and the players who scored them. In HEX, we utilized this data in combination with sentiment information to establish a relationship between the match status and sentiment.

## Instalation?

## Usage

Files to run to get to final generated files

1. 

2. 

3.

4.

5. 

6.

7. 



## Final considerations



