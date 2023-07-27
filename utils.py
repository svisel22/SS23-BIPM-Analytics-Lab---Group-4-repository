from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import emoji
import re
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# Function which finds the lines where a players name is contained
def find_lines_with_player(dataframe, playerlist, n_lines = 0):
    """
    This function transform article data into just the lines where a playername appears. 
    
    Parameters:
        dataframe: DataFrame which contains the whole article data in the column "data"
        playerlist: list of all players for which the lines should be conducted
        n_lines: number of lines following the line where the name appears which also should be conducted
    
    Returns:
        DataFrame with the sentences where the playername appears as data column
    """
    
    # Create empty df 
    df_complete = pd.DataFrame()

    # Iterating over all players
    for player in playerlist:

        # Get players first and last_name to include them in later sentence checks
        player_first_name, player_last_name = player.split()

        # Just select player indiviual data
        df_player = dataframe[dataframe["player"] == player]
        df_player = df_player.reset_index(drop=True)

        # Iterate over all data for the player
        for i in range(len(df_player)):

            # Get the current record
            current_line = df_player['data'].iloc[i]

            # Split up the records in lines
            lines = current_line.split('\\n')

            # Create an empty string
            new_string = ''

            line_counter = 0
            # Iterate over all lines in the record
            for line in lines:

                # If the playername can be found in the line add the line to the string
                if line.find(player) != -1:
                    new_string = new_string + line + " "
                    if line_counter <= 0:
                        line_counter = line_counter + n_lines
            
                elif line.find(player_first_name) != -1:
                    new_string = new_string + line + " "
                    if line_counter <= 0:
                        line_counter = line_counter + n_lines
        
                elif line.find(player_last_name) != -1:
                    new_string = new_string + line + " "
                    if line_counter <= 0:
                        line_counter = line_counter + n_lines
            
                elif line_counter >= 0:
                    new_string = new_string + line + " "
                    line_counter = line_counter-1
        
            # Switch the previos record against the newly created string 
            df_player['data'].iloc[i] = new_string

        # Add the new data to the Dataframe and return
        df_complete = pd.concat([df_complete, df_player], axis=0)
        
    return df_complete


def remove_similar_rows(column_as_df, df, threshold=0.9):
    """
    This function deletes similiar data rows.
    
    Parameters:
        column_as_df: data out of a DataFrame where the similiar rows should be removed
        df: corresponding DataFrame
        threshold: integer with degree of row similiarity. default is deletion of rows which are 90% similiar.
    
    Returns:
        DataFrame with dropped rows.
    """
    # Compute similarity scores for each pair of rows
    similarity_scores = {}
    for i, row in column_as_df.iterrows():
        for j, other_row in column_as_df.iterrows():
            if i >= j:
                continue
            score = SequenceMatcher(None, row, other_row).ratio()
            if score >= threshold:
                similarity_scores[(i, j)] = score
    
    # Identify rows to remove
    rows_to_remove = []
    for (i, j), score in similarity_scores.items():
        if i not in rows_to_remove and j not in rows_to_remove:
            rows_to_remove.append(j if df.index[i] < df.index[j] else i)
    
    # Remove rows and return modified DataFrame
    return df.drop(rows_to_remove)



def remove_similar_rows_per_player(df, playerlist, threshold=0.9):
    """
    This function deletes similiar data rows for each player individually. 
    The procedure deletion of similiar articles needs to be done for each player individually, 
    because articles which include information for multiple players should then also be used for multiple players.
    
    Parameters:
        df: Dataframe which containes the columns: data and player as an input. The column data relates to news articles.
        playerlist: list of all players for which rows should be removed
        threshold: integer with degree of row similiarity. default is deletion of rows which are 90% similiar.
    
    Returns:
        DataFrame with dropped rows.
    """

    # Define empty df which will be returned in the end
    df_complete = pd.DataFrame()

    for player in playerlist:
        
        # Create the df for the player
        df_player = df[df["player"] == player]
        df_player = df_player.reset_index(drop=True)
        column_as_df = pd.DataFrame(df_player['data'])


        
        # Compute similarity scores for each pair of rows
        similarity_scores = {}
        for i, row in column_as_df.iterrows():
            for j, other_row in column_as_df.iterrows():
                if i >= j:
                    continue
                score = SequenceMatcher(None, row, other_row).ratio()
                if score >= threshold:
                    similarity_scores[(i, j)] = score
        
        # Identify rows to remove 
        rows_to_remove = []
        for (i, j), score in similarity_scores.items():
            if i not in rows_to_remove and j not in rows_to_remove:
                rows_to_remove.append(j if df_player.index[i] < df_player.index[j] else i)
        
        # Remove rows and concatenate df
        df_player = df_player.drop(rows_to_remove)
        df_complete = pd.concat([df_complete, df_player], axis=0)

        #Return modified DataFrame
    return df_complete


def del_patterns(df_line, pattern):
    """
    This function removes defined patterns from a list 
    
    Parameters:
        df_line: string which contains pattern
        pattern: pattern which should be delted
    
    Returns:
        string without stopwords
    """

    # Split up the records in lines
    lines = df_line.split('\\n')

    if len(lines)>1:

        # Create an empty string
        new_string = ''

        # Iterating over the lines 
        for line in lines:
        
        # Set deleting to False first
            deleting = False
            # Check if a pattern word is included in the line and set deleting to True if so 
            for word in pattern:
                if deleting == True:
                    break
                elif word in line:
                    deleting = True
                else:
                    deleting = False
            
            # If the setence should not be delete it add it to the string  
            if deleting == False:
                new_string = new_string + line + '\\n '

    else:
        new_string = df_line
    # Return the string 
    return new_string


def word_frequency_per_player(df, playerlist):
    """
    This function returns the word freqencies for each players data. 
    
    Parameters:
        df: DataFrame with the data in the column 'data'
        playerlist: list of players for which the word frequencies should be conducted
    
    Returns:
        DataFrame with all players and corresponding word frequencies
    """
     # Define empty df which will be returned in the end
    df_complete = pd.DataFrame()

    # Iterate over players
    for player in playerlist:

         # Create the df for the player
        df_player = df[df["player"] == player]
        df_player = df_player.dropna(subset=['data'])
        df_player = df_player.reset_index(drop=True)

        # Create a stemmed data corpus
        df_player['stemmed_data'] = df_player['data'].apply(stem_text)
        data_stem = df_player['data'].apply(stem_text)
        data = data_stem.tolist()

        # Create a corpus
        corpus_gen=[doc.split() for doc in data]

        # Store corpus Dicitonary
        id2word = Dictionary(corpus_gen)

        # Filter out rare and common words
        id2word.filter_extremes(no_below=5, no_above=0.95)

        # Display features and their frequencies
        df_frequencies = pd.DataFrame(columns=['Word', 'Frequency', 'player'])
    
        i = 0 
        for feature, frequency in id2word.cfs.items():

            # Append a new row to the DataFrame
            df_frequencies.loc[i]= [id2word[feature],frequency, player]
            i = i+1

        # Sort the frequencies descanding
        df_frequencies = df_frequencies.sort_values('Frequency', ascending=False)

        # Concatenate to df_complete
        df_complete = pd.concat([df_complete, df_frequencies], axis=0)

    # return df_complete
    return df_complete


def print_player_freq(df_freq):
    """
    This function prints out the 20 most common words for each players data. 
    
    Parameters:
        df_freq: DataFrame with the word frequncy data conducted with the "word_frequency_per_player" function
    
    Returns:
        DataFrame with the 20 most common words per player
    """
    # Create empty df
    df_freq_res = pd.DataFrame()
    
    for player in df_freq['player'].unique():
        # Delete rows we don't want to see
        # Define the strings to be deleted
        strings_to_delete = ['palacio', 'bayerleverkusen','andrich','leverkusen', 'bayer', 'au', 'fur', 'nicht', 'diabi','alonso', 'wurd', 'hatt', 'zwei', 'seit', 'acht',
                             'man', 'get', 'player','july', 's', 't', 'it',
                             'ma', 'do', 'tambien', 'tra', 'xa']

        # Delete rows containing the specified strings 
        df_freq = df_freq[~df_freq['Word'].isin(strings_to_delete)]

        # Select word frequencies for one Player
        df_player = df_freq[(df_freq["player"] == player)]
        # Print out the top 20 word frequencies 
        print('Top Words for ', player , '\n', df_player.head(20), '\n')

        # Add data to the DataFrame
        df_freq_res  = pd.concat([df_freq_res, df_player.head(20)], axis=0)

    
    return df_freq_res


def map_emoji_to_description(emoji_text, language): 
    """
    This function transforms emojis into text
    
    Parameters:
        emoji_text: string which contains emojis
        language: language in which the text is written and to which the emojis should be mapped
    
    Returns:
        string with text instead of emojis
    """
    emoji_description = emoji.demojize(emoji_text, language=language)
    return emoji_description


def translate_emojis(text, language):
    """
    This function transforms text which contains emojis into text without emojis 
    
    Parameters:
        text: string which contains emojis
        language: language in which the text is written and to which the emojis should be mapped
    
    Returns:
        string with text instead of emojis
    """
    return re.sub(r'[\U0001F000-\U0001F999]', lambda match: map_emoji_to_description(match.group(), language=language), text)


def remove_accents(text):
    """
    This function removes accents from text
    
    Parameters:
        text: string which contains accents
    
    Returns:
        string without accents
    """
    return unidecode(text)


def remove_stopwords_from_text(text, stopwords_list_per_language):
    """
    This function removes stopwords from text
    
    Parameters:
        text: string which contains stopwords
        stopwords_list_per_language: stopwords which should be deleted
    
    Returns:
        string without stopwords
    """
    return remove_stopwords(text, stopwords=stopwords_list_per_language)



def name_wordgroups(df):
    """
    This function matches first and surname of players to just last name 
    
    Parameters:
        df: DataFrame with the data in the column 'data'
    
    Returns:
        DataFrame with all player names as just last names
    """
    # create patterns which should be matched 
    # first lastname and firstname should both result in just lastname
    pattern_match2d = np.array([[r"\b(mitchel bakker|mitchel)\b", 'bakker'], 
                                [r"\b(xabi alonso|xabi)\b", 'alonso'], 
                                [r"\b(exequiel palacios|exequiel)\b", 'palacios'],
                                [r"\b(nadiem amiri|nadiem)\b", 'amiri'],
                                [r"\b(kerem demirbay|kerem)\b", 'demirbay'],
                                [r"\b(robert andrich|robert)\b", 'andrich'],
                                [r"\b(exequiel palacios|exequiel)\b", 'palacios'],
                                [r"\b(piero hincapie|piero)\b", 'hincapie'],
                                [r"\b(jeremie frimpong|jeremie)\b", 'frimpong'],
                                [r"\b(jonathan tah|jonathan)\b", 'tah'],
                                [r"\b(moussa diaby|moussa)\b", 'diaby'],
                                [r"\b(mykhaylo mudryk|mykhaylo)\b", 'mudryk'],
                                [r"\b(amine adli|amine)\b", 'adli'],
                                [r"\b(florian wirtz|florian)\b", 'wirtz'],
                                [r"\b(jose mourinho|jose)\b", 'mourinho'],     
                                #other wordgroups
                                [r"\b(europa league)\b", 'europaleague'],
                                [r"\b(champions league)\b", 'championsleague'],
                                [r"\b(bayer leverkusen|bayer|leverkusen|leverkusens)\b", 'bayerleverkusen']
                                ])

    # do the pattern matching for each player
    for pattern, player in pattern_match2d:
        df['data'] = df['data'].apply(lambda x: re.sub(pattern, str(player), str(x)))

    return df






'''
# Function to remove specific words from the string
def remove_words(text):
    pattern = r"\b(mitchel|bakker|exequiel|palacios|piero|hincapie|jeremie|frimpong|jonathan|tah|moussa|diaby|mykhaylo|mudryk)\b"
    return re.sub(pattern, "", text)

# Apply the function to the data column
df_stem['data'] = df_stem['data'].apply(lambda x: remove_words(str(x)))

df_stem
'''

# For Models_accuracy_per_langugage to evaluate performance
def evaluate_performance(df, sentiment_column, label_column):
    """
    This function evaluates the performance of a sentiment analysis model by calculating accuracy, generating a confusion matrix, and creating a classification report. It takes a DataFrame with true sentiment labels and predicted sentiment scores as input.

    Parameters:
        df: DataFrame
            A DataFrame containing true sentiment labels in the column specified by 'label_column' and predicted sentiment scores in the column specified by 'sentiment_column'.
        sentiment_column: str
            The name of the column in the DataFrame containing the predicted sentiment scores.
        label_column: str
            The name of the column in the DataFrame containing the true sentiment labels.

    Returns:
        tuple
            A tuple containing the following elements:
            - accuracy: float
                The accuracy of the sentiment analysis model.
            - unique_predicted: ndarray
                An array containing the unique predicted sentiment labels.
            - cm_df: DataFrame
                A DataFrame representing the confusion matrix for better visualization.
            - report: str
                The classification report containing precision, recall, F1-score, and support for each class.
    """

    # Calculate the accuracy
    accuracy = accuracy_score(df[label_column], df[sentiment_column])

    # Find unique predicted sentiment labels
    unique_predicted = df[sentiment_column].unique()

    # Assign true and predicted labels
    true_labels = df[label_column]
    predicted_labels = df[sentiment_column]

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Convert the confusion matrix to a DataFrame for better visualization
    labels = np.unique(np.concatenate((true_labels, predicted_labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)


    # Generate the classification report
    report = classification_report(true_labels, predicted_labels)

    return accuracy, unique_predicted, cm_df, report



# For Models_accuracy_per_langugage to transform scores into positiv, negativ,  neutral
def transform_scores(df, sentiment_column):
    """
    This function transforms sentiment scores into three-dimensional sentiment labels (positive/neutral/negative). It takes a DataFrame containing sentiment scores as input and returns a list of corresponding sentiment labels.

    Parameters:
        df: DataFrame
            A DataFrame containing sentiment scores, typically in a column named 'sentiment_bert'.
            
    Returns:
        list of str
            A list of sentiment labels ('positive', 'neutral', or 'negative') based on the input sentiment scores.
    """

    sentiment_3_labels = []
    for score in df['sentiment_bert']: 
        if score > 0.6:
            sentiment_label = "positiv"
        elif score < 0.4:
            sentiment_label = "negativ"
        else:
            sentiment_label = "neutral"
        sentiment_3_labels.append(sentiment_label)
    return sentiment_3_labels