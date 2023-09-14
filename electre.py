"""
ELECTRE solver

Args: 
- Tabled data for Alternatives x Criteria

Alternatives| Criteria 1| Criteria 2| Criteria 3|   
Weights     | w1         | w2         | w3      |
Paris       | x11        | x12        | x13     |
Copenhagen  | x21        | x22        | x23     |


- Weights for Criteria
- Thresholds for Criteria
"""


import numpy as np
import pandas as pd

# Generate concordance matrix
def concordance_matrix(df, weights, thresholds):
    """
    Script to take a dataframe with rows as alternatives
    and columns as criteri.

    Index is alternative names
    Columns are criteria names

    Args:
    df: dataframe
    weights: list of weights for each criteria

    Returns:
    concordance matrix
    """

    # Create empty matrix
    con_mat = np.zeros((len(df), len(df)))

    # For each alternative we compare with all other alternatives 
    # If the alternative is better than the other alternative for each criteria
    # The value added to the concordance matrix is sum of outperforming criterias over sum(weights)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                # For each criteria
                sum_crit = 0
                for k in range(len(df.columns)):
                    if df.iloc[i, k] >= df.iloc[j, k]:
                        sum_crit += weights[k]
                con_mat[i, j] = round(sum_crit/sum(weights),2) if sum_crit/sum(weights) >= thresholds[k] else 0

    return con_mat

def discordance_matrix(df, thresholds):
    """
    Script to take a dataframe with rows as alternatives
    and columns as criteri.

    Index is alternative names
    Columns are criteria names

    Args:
    df: dataframe
    weights: list of weights for each criteria
    thresholds: list of thresholds for each criteria

    Returns:
    concordance matrix
    """

    # Create empty matrix
    disc_mat = np.zeros((len(df), len(df)))
    # We apply the function:
    # D(a,b) = 1 if z(b) - z(a) > threshold for at least one criteria
    # D(a,b) = 0 otherwise
    for i in range(len(df)):
        for j in range(len(df)):
            # For each criteria
            for k in range(len(df.columns)):
                if df.iloc[j, k] - df.iloc[i, k] > thresholds[k]:
                    disc_mat[i, j] = 1
                    break

    return disc_mat

def selected_alternatives(con_df, disc_df, alts_names):
    """
    Script to take both the concordance and discordance matrices
    and return the list of selected alternatives based on the
    ELECTRE method and the established thresholds.

    Args:
    con_df: concordance matrix
    disc_df: discordance matrix

    Returns:
    list of selected alternatives
    """

    # Map discordance matrix to concordance matrix
    # If discordance is 1, concordance is 0

    # Create empty matrix
    final_mat = con_df * disc_df

    # Set cols and rows names
    final_mat = pd.DataFrame(final_mat, columns=alts_names, index=alts_names)

    return final_mat


weights = [6, 4, 3, 10, 8, 6]
con_thresholds  = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
disc_thresholds = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
alts_names = ['Paris', 'Brussels', 'Amsterdam', 'Berlin', 'Warsaw', 'Milan', 'London']
crit_names = ['AoS', 'Accs', 'QoL', 'BPPub', 'BPPriv', 'EoS']
# Rows definition
col_dat = [[3, 4, 4, 4, 5, 3], 
           [2, 3, 3, 5, 3, 3], 
           [2, 5, 3, 4, 5, 4], 
           [4, 2, 2, 3, 4, 4], 
           [4, 2, 2, 3, 3, 2],
           [4, 2, 5, 3, 4, 4], 
           [3, 5, 4, 3, 4, 5]]

# Create dataframe
df = pd.DataFrame(col_dat, columns=crit_names, index=alts_names)
df.index.name = 'Alternatives'
df.columns.name = 'Criteria'

print("\n" + "="*30)
print("Dataframe")
print("="*30)
print(df)

# Create concordance matrix
con_mat = concordance_matrix(df, weights, con_thresholds)
print("\n" + "="*30)
print("Concordance matrix")
print("="*30)
print(con_mat)

# Create discordance matrix
disc_mat = discordance_matrix(df, disc_thresholds)
print("\n" + "="*30)
print("Discordance matrix")
print("="*30)
print(disc_mat)

# Get selected alternatives
selected_alts = selected_alternatives(con_mat, disc_mat, alts_names)
print("\n" + "="*30)
print(selected_alts)


