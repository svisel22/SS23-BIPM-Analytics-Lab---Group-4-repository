from difflib import SequenceMatcher
import numpy as np
import pandas as pd


def remove_similar_rows(column_as_df, df, threshold=0.9):
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


# Function which finds the lines where a players name is contained
def find_lines_with_player(dataframe, playerlist):
    
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

            # iterate over all lines in the record
            for line in lines:
                # if the playername can be found in the line add the line to the string
                if line.find(player) != -1:
                    new_string = new_string + line
                elif line.find(player_first_name) != -1:
                    new_string = new_string + line
                elif line.find(player_last_name) != -1:
                    new_string = new_string + line
            
            # switch the previos record against the newly created string 
            df_player['data'].iloc[i] = new_string

        # add the new data to the Dataframe and return
        df_complete = pd.concat([df_complete, df_player], axis=0)
        
    return df_complete

