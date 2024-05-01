###### Step 1: Import Python Libraries

# Data processing
import pandas as pd
import numpy as np
###### Step 2: Import Data

df = pd.read_csv('/Users/bibekdhakal/DJANGO/recomend/suggest/product/europe-motorbikes-zenrows.csv')
df = df.drop(columns=['link'])

# Optional: truncating the data for fast testing; remove this for actual use
df = df.head(2000)

# Fill NaN values with a default value (e.g., empty string) in columns that are being used for recommendation
df['fuel'] = df['fuel'].fillna('')
df['gear'] = df['gear'].fillna('')
df['version'] = df['version'].fillna('')
df['make_model'] = df['make_model'].fillna('')

# Sample interactions data (user_id, item_id, timestamp)

data = [
    (1, 101, '2024-04-28T12:00:00Z'),
    (1, 102, '2024-04-28T12:15:00Z'),
    (1, 101, '2024-04-28T12:30:00Z'),
    (1, 105, '2024-04-28T13:00:00Z'),
    (2, 101, '2024-04-28T14:30:00Z'),
    (2, 103, '2024-04-28T14:45:00Z'),
    (2, 102, '2024-04-28T15:00:00Z'),
    (2, 105, '2024-04-28T15:30:00Z'),
    (3, 105, '2024-04-28T12:00:00Z'),
    (3, 102, '2024-04-28T13:15:00Z'),
    (1, 101, '2024-04-29T10:00:00Z'),
    (1, 104, '2024-04-29T10:30:00Z'),
    (1, 102, '2024-04-29T11:00:00Z'),
    (1, 105, '2024-04-29T11:30:00Z'),
    (2, 104, '2024-04-29T14:00:00Z'),
    (2, 103, '2024-04-29T14:30:00Z'),
    (2, 101, '2024-04-29T15:00:00Z'),
    (2, 102, '2024-04-29T15:30:00Z'),
    (3, 105, '2024-04-29T12:00:00Z'),
    (3, 103, '2024-04-29T13:15:00Z'),
    (1, 106, '2024-04-30T10:00:00Z'),
    (1, 107, '2024-04-30T10:30:00Z'),
    (1, 108, '2024-04-30T11:00:00Z'),
    (1, 109, '2024-04-30T11:30:00Z'),
    (2, 106, '2024-04-30T14:00:00Z'),
    (2, 107, '2024-04-30T14:30:00Z'),
    (2, 108, '2024-04-30T15:00:00Z'),
    (2, 109, '2024-04-30T15:30:00Z'),
    (3, 106, '2024-04-30T12:00:00Z'),
    (3, 107, '2024-04-30T13:15:00Z'),
    (1, 110, '2024-05-01T10:00:00Z'),
    (1, 111, '2024-05-01T10:30:00Z'),
    (1, 112, '2024-05-01T11:00:00Z'),
    (1, 113, '2024-05-01T11:30:00Z'),
    (2, 110, '2024-05-01T14:00:00Z'),
    (2, 111, '2024-05-01T14:30:00Z'),
    (2, 112, '2024-05-01T15:00:00Z'),
    (2, 113, '2024-05-01T15:30:00Z'),
    (3, 110, '2024-05-01T12:00:00Z'),
    (3, 111, '2024-05-01T13:15:00Z'),
    (1, 114, '2024-05-02T10:00:00Z'),
    (1, 115, '2024-05-02T10:30:00Z'),
    (1, 116, '2024-05-02T11:00:00Z'),
    (1, 117, '2024-05-02T11:30:00Z'),
    (2, 114, '2024-05-02T14:00:00Z'),
    (2, 115, '2024-05-02T14:30:00Z'),
    (2, 116, '2024-05-02T15:00:00Z'),
    (2, 117, '2024-05-02T15:30:00Z'),
    (3, 114, '2024-05-02T12:00:00Z'),
    (3, 115, '2024-05-02T13:15:00Z'),
    (1, 118, '2024-05-03T10:00:00Z'),
    (1, 119, '2024-05-03T10:30:00Z'),
    (1, 120, '2024-05-03T11:00:00Z'),
    (1, 121, '2024-05-03T11:30:00Z'),
    (2, 118, '2024-05-03T14:00:00Z'),
    (2, 119, '2024-05-03T14:30:00Z'),
    (2, 120, '2024-05-03T15:00:00Z'),
    (2, 121, '2024-05-03T15:30:00Z'),
    (3, 118, '2024-05-03T12:00:00Z'),
    (3, 119, '2024-05-03T13:15:00Z'),
    (1, 122, '2024-05-04T10:00:00Z'),
    (1, 123, '2024-05-04T10:30:00Z'),
    (1, 124, '2024-05-04T11:00:00Z'),
    (1, 125, '2024-05-04T11:30:00Z'),
    (2, 122, '2024-05-04T14:00:00Z'),
    (2, 123, '2024-05-04T14:30:00Z'),
    (2, 124, '2024-05-04T15:00:00Z'),
    (2, 125, '2024-05-04T15:30:00Z'),
    (3, 122, '2024-05-04T12:00:00Z'),
    (3, 123, '2024-05-04T13:15:00Z'),
    (1, 126, '2024-05-05T10:00:00Z'),
    (1, 127, '2024-05-05T10:30:00Z'),
    (1, 128, '2024-05-05T11:00:00Z'),
    (1, 129, '2024-05-05T11:30:00Z'),
    (2, 126, '2024-05-05T14:00:00Z'),
    (2, 127, '2024-05-05T14:30:00Z'),
    (2, 128, '2024-05-05T15:00:00Z'),
    (2, 129, '2024-05-05T15:30:00Z'),
    (3, 126, '2024-05-05T12:00:00Z'),
    (3, 127, '2024-05-05T13:15:00Z'),
    (1, 130, '2024-05-06T10:00:00Z'),
    (1, 131, '2024-05-06T10:30:00Z'),
    (1, 132, '2024-05-06T11:00:00Z'),
    (1, 133, '2024-05-06T11:30:00Z'),
    (2, 130, '2024-05-06T14:00:00Z'),
    (2, 131, '2024-05-06T14:30:00Z'),
    (2, 132, '2024-05-06T15:00:00Z'),
    (2, 133, '2024-05-06T15:30:00Z'),
    (3, 130, '2024-05-06T12:00:00Z'),
    (3, 131, '2024-05-06T13:15:00Z'),
    (1, 134, '2024-05-07T10:00:00Z'),
    (1, 135, '2024-05-07T10:30:00Z'),
    (1, 136, '2024-05-07T11:00:00Z'),
    (1, 137, '2024-05-07T11:30:00Z'),
    (2, 134, '2024-05-07T14:00:00Z'),
    (2, 135, '2024-05-07T14:30:00Z'),
    (2, 136, '2024-05-07T15:00:00Z'),
    (2, 137, '2024-05-07T15:30:00Z'),
    (3, 134, '2024-05-07T12:00:00Z'),
    (3, 135, '2024-05-07T13:15:00Z'),
    (1, 138, '2024-05-08T10:00:00Z'),
    (1, 139, '2024-05-08T10:30:00Z'),
    (1, 140, '2024-05-08T11:00:00Z'),
    (1, 141, '2024-05-08T11:30:00Z'),
    (2, 138, '2024-05-08T14:00:00Z'),
    (2, 139, '2024-05-08T14:30:00Z'),
    (2, 140, '2024-05-08T15:00:00Z'),
    (2, 141, '2024-05-08T15:30:00Z'),
    (3, 138, '2024-05-08T12:00:00Z'),
    (3, 139, '2024-05-08T13:15:00Z'),
    (1, 142, '2024-05-09T10:00:00Z'),
    (1, 143, '2024-05-09T10:30:00Z'),
    (1, 144, '2024-05-09T11:00:00Z'),
    (1, 145, '2024-05-09T11:30:00Z'),
    (2, 142, '2024-05-09T14:00:00Z'),
    (2, 143, '2024-05-09T14:30:00Z'),
    (2, 144, '2024-05-09T15:00:00Z'),
    (2, 145, '2024-05-09T15:30:00Z'),
    (3, 142, '2024-05-09T12:00:00Z'),
    (3, 143, '2024-05-09T13:15:00Z'),
    (1, 146, '2024-05-10T10:00:00Z'),
    (1, 147, '2024-05-10T10:30:00Z'),
    (1, 148, '2024-05-10T11:00:00Z'),
    (1, 149, '2024-05-10T11:30:00Z'),
    (2, 146, '2024-05-10T14:00:00Z'),
    (2, 147, '2024-05-10T14:30:00Z'),
    (2, 148, '2024-05-10T15:00:00Z'),
    (2, 149, '2024-05-10T15:30:00Z'),
    (3, 146, '2024-05-10T12:00:00Z'),
    (3, 147, '2024-05-10T13:15:00Z'),
    (1, 150, '2024-05-11T10:00:00Z'),
    (1, 151, '2024-05-11T10:30:00Z'),
    (1, 152, '2024-05-11T11:00:00Z'),
    (1, 153, '2024-05-11T11:30:00Z'),
    (2, 150, '2024-05-11T14:00:00Z'),
    (2, 151, '2024-05-11T14:30:00Z'),
    (2, 152, '2024-05-11T15:00:00Z'),
    (2, 153, '2024-05-11T15:30:00Z'),
    (3, 150, '2024-05-11T12:00:00Z'),
    (3, 151, '2024-05-11T13:15:00Z'),
]
# Convert data to a pandas DataFrame
interactions_df = pd.DataFrame(data, columns=['user_id', 'item_id', 'timestamp'])

# Convert the 'timestamp' column to pandas datetime objects
interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])

def calculate_decay_rate(desired_weight_after_interval, interval_in_days=30):
    """
    Calculate the decay rate for exponential decay function.
    
    Args:
    desired_weight_after_one_month (float): The desired weight after one month (e.g., 0.01).
    one_month_days (int): The number of days considered as one month (default is 30).
    
    Returns:
    float: The calculated decay rate.
    """
    # Calculate the time difference in seconds for one month
    total_seconds = interval_in_days * 24 * 60 * 60  # Convert days to seconds
    
    # Calculate the decay rate using the exponential decay formula
    decay_rate = -np.log(desired_weight_after_interval) / total_seconds
    
    return decay_rate


###### Step 4: Create User-Moterbike Matrix

# Define the decay rate for the exponential decay function
decay_rate = calculate_decay_rate(0.01,10)

# Calculate the maximum timestamp for each user_id
max_timestamps = interactions_df.groupby('user_id')['timestamp'].max().reset_index()
max_timestamps.columns = ['user_id', 'max_timestamp']

# Merge the maximum timestamps back to the original DataFrame
interactions_df = interactions_df.merge(max_timestamps, on='user_id')

# Calculate the time difference from the maximum timestamp for each user_id
interactions_df['time_diff'] = (interactions_df['max_timestamp'] - interactions_df['timestamp']).dt.total_seconds()

# Calculate the weight using an exponential decay function
interactions_df['weight'] = np.exp(-decay_rate * interactions_df['time_diff'])

# Create a pivot table with weighted sums for each user_id and item_id
user_item_matrix = interactions_df.pivot_table(index='user_id', columns='item_id', values='weight', aggfunc='sum',fill_value=0)
user_item_matrix.head()

###### Step 5: Data Normalization
# Normalize user-item matrix
matrix_norm = user_item_matrix.subtract(user_item_matrix.mean(axis=1), axis = 0)
matrix_norm.head()

###### Step 6: Calculate Similarity Score

# Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.corr()
item_similarity.head()

def item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations =3):
    import operator
    # Movies that the target user has not watched    
    picked_userid_uninteracted = pd.DataFrame(user_item_matrix.T[picked_userid]==0).reset_index()
    picked_userid_uninteracted = picked_userid_uninteracted[picked_userid_uninteracted[1]==True]['item_id'].values.tolist()
    # items that the target user has interacted
    picked_userid_interacted = pd.DataFrame(matrix_norm.T[picked_userid].dropna(axis=0, how='all')\
                                .sort_values(ascending=False))\
                                .reset_index()\
                                .rename(columns={1:'rating'})
                
    # Dictionary to save the uninteracted items and predicted rating pair
    rating_prediction ={} 
    # Loop through uninteracted items       
    for picked_item in picked_userid_uninteracted:
        # Calculate the similarity score of the picked item with other movies
        picked_item_similarity_score = item_similarity[[picked_item]].reset_index().rename(columns={picked_item:'similarity_score'})
        # Rank the similarities between the picked user interacted item and the picked uninteracted item.
        picked_userid_interacted_similarity = pd.merge(left=picked_userid_interacted, 
                                                    right=picked_item_similarity_score, 
                                                    on='item_id', 
                                                    how='inner')\
                                            .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
        # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
        predicted_rating = round(np.average(picked_userid_interacted_similarity['rating'], 
                                            weights=picked_userid_interacted_similarity['similarity_score']), 6)
        # Save the predicted rating in the dictionary
        rating_prediction[picked_item] = predicted_rating
    # Return the top recommended movies
    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
    
