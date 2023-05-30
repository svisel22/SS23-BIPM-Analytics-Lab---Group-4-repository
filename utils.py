from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import emoji
import re
from gensim.parsing.preprocessing import remove_stopwords
from unidecode import unidecode


# Function which finds the lines where a players name is contained
def find_lines_with_player(dataframe, playerlist, n_lines = 0):
    
    # create empty df 
    df_complete = pd.DataFrame()

    # iterating over all players
    for player in playerlist:

        # get players first and last_name to include them in later sentence checks
        player_first_name, player_last_name = player.split()

        # just select player indiviual data
        df_player = dataframe[dataframe["player"] == player]
        df_player = df_player.reset_index(drop=True)

        # iterate over all data for the player
        for i in range(len(df_player)):

            # get the current record
            current_line = df_player['data'].iloc[i]
            # split up the records in lines
            lines = current_line.split('\\n')
            # create an empty string
            new_string = ''

            line_counter = 0
            # iterate over all lines in the record
            for line in lines:
                # if the playername can be found in the line add the line to the string
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
        
            # switch the previos record against the newly created string 
            df_player['data'].iloc[i] = new_string

        # add the new data to the Dataframe and return
        df_complete = pd.concat([df_complete, df_player], axis=0)
        
    return df_complete

def remove_similar_rows(column_as_df, df, threshold=0.9):
    ''' The old Function of removing similiarities is deleting allduplicate articles'''

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
    '''The procedure of deleting similiar articles needs to be done by each player because if an article writes about 
    # e.g. two players we want to keep it for both of the players'''

    # define empty df which will be returned in the end
    df_complete = pd.DataFrame()

    for player in playerlist:
        
        # create the df for the player
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

        #return modified DataFrame
    return df_complete

def map_emoji_to_description(emoji_text, language): 
    emoji_description = emoji.demojize(emoji_text, language=language)
    return emoji_description

def translate_emojis(text, language):
    return re.sub(r'[\U0001F000-\U0001F999]', lambda match: map_emoji_to_description(match.group(), language=language), text)

def remove_accents(text):
    return unidecode(text)

def remove_stopwords_from_text(text, stopwords_list_per_language):
    return remove_stopwords(text, stopwords=stopwords_list_per_language)

def del_patterns(df_line, pattern):
    '''
    Function which takes an input and deletes defined text pattern 
    '''
    # split up the records in lines
    lines = df_line.split('\\n')
    '''
    lines = df_line.split('\n')  # Split at newline characters first

    split_lines = []
    for line in lines:
        split_lines.extend(line.split('\\t'))  # Split each line at tab characters

    '''

    if len(lines)>1:

        # create an empty string
        new_string = ''

        # iterating over the lines 
        for line in lines:
        
        # set deleting to False first
            deleting = False
            # check if a pattern word is included in the line and set deleting to True if so 
            for word in pattern:
                if deleting == True:
                    break
                elif word in line:
                    deleting = True
                else:
                    deleting = False
            
            # if the setence should not be delete it add it to the string  
            if deleting == False:
                new_string = new_string + line 

    else:
        new_string = df_line
    # return the string 
    return new_string