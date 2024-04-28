#Import all the required packages
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Load the .csv file into a Pandas dataframe
df = pd.read_csv('/Users/bibekdhakal/DJANGO/recomend/suggest/product/europe-motorbikes-zenrows.csv')
df = df.drop(columns=['link'])

# truncating the data for fast testing remove this for actual use
df = df.head(2000)

# the numeric column contain NaN value is not a problem as it is handled in the program logic using penalties

# Replace NaN values in the 'fuel' column with a default value (e.g., empty string)
df['fuel'] = df['fuel'].fillna('')

# Replace NaN values in the 'gear' column with a default value (e.g., empty string)
df['gear'] = df['gear'].fillna('')

# Replace NaN values in the 'version' column with a default value (e.g., empty string)
df['version'] = df['version'].fillna('')


# Create a separate DataFrame for text-based data
text_df = pd.DataFrame()

# Concatenate all text columns into a single text column in the new DataFrame
text_df['text'] = df['make_model'] + " " + df['offer_type'] + " " + df['version'] + " " + df['date'] + " " + df['fuel'] + " " + df['gear']


# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_matrix = vectorizer.fit_transform(text_df['text'])



def recommend_system(df, preferences, num_recommendations=5):
    """
    Combines numeric similarity and text-based similarity to provide recommendations.

    Parameters:
        df: DataFrame - The dataset containing all entries.
        preferences: dict - An object with properties like price, mileage, power, make_model, date, fuel, gear, offer_type, and version.
        num_recommendations: int - The number of recommendations to retrieve (default: 5).

    Returns:
        A list of tuples containing the index and combined similarity score of the top N recommendations.
    """
    # Define a dictionary of preferred values for numeric columns
    preferred_values = {
        'price': preferences.get('price', None),
        'power': preferences.get('power', None),
        'mileage': preferences.get('mileage', None)
    }
    # Calculate penalties for missing values in numeric columns
    penalties = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Calculate the mean, max, and min of the column
            col_mean = df[col].mean()
            col_max = df[col].max()
            col_min = df[col].min()

            # Calculate squared distances from mean to max and mean to min
            max_distance = (col_max - col_mean) ** 2
            min_distance = (col_mean - col_min) ** 2

            # Calculate the penalty as the maximum of the squared distances
            penalties[col] = max(max_distance, min_distance)
    # Calculate numeric similarity scores
    numeric_similarity_scores = []

    # Iterate through the dataset
    for idx, row in df.iterrows():
        # Calculate distance based on numeric columns
        distance = 0
        count = 0

        for col, preferred_value in preferred_values.items():
            if preferred_value is not None and col in df.columns:
                # Check for NaN in the row
                if pd.isna(row[col]):
                    # Apply the penalty for missing value from the penalties dictionary
                    distance += penalties[col]
                else:
                    # Calculate squared difference and sum it up
                    distance += (row[col] - preferred_value) ** 2
                    count += 1

        if count > 0:
            # Calculate Euclidean distance and convert it to a similarity score
            distance = np.sqrt(distance)
            numeric_similarity_score = 1 / (1 + distance)
        else:
            # Default similarity score when no preferences are provided
            numeric_similarity_score = 0

        # Append the index and similarity score as a tuple
        numeric_similarity_scores.append((idx, numeric_similarity_score))

    # Sort entries based on numeric similarity scores in descending order
    numeric_similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Combine text-based similarity (cosine similarity)
    # Concatenate text columns into a single text column
    df['text'] = df['make_model'] + " " + df['offer_type'] + " " + df['version'] + " " + df['date'] + " " + df['fuel'] + " " + df['gear']

    # Transform the query text (from preferences) using the same TfidfVectorizer used for the dataset
    query_text = f"{preferences.get('make_model', '')} {preferences.get('offer_type', '')} {preferences.get('version', '')} {preferences.get('date', '')} {preferences.get('fuel', '')} {preferences.get('gear', '')}"
    query_vector = vectorizer.transform([query_text])

    # Calculate cosine similarity between the query vector and the TF-IDF matrix of the dataset
    query_similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Combine numeric and text similarities
    combined_similarity_scores = []

    # Iterate through the dataset and combine similarities
    for idx in range(len(df)):
        # Combine similarity scores using a simple weighted average (50% weight to each type of similarity)
        combined_score = (numeric_similarity_scores[idx][1] + query_similarity_scores[idx]) / 2

        # Append the index and combined similarity score as a tuple
        combined_similarity_scores.append((idx, combined_score))

    # Sort entries based on combined similarity scores in descending order
    combined_similarity_scores.sort(key=lambda x: x[1], reverse=True)


    # Get the top N recommendations
    top_recommendations = combined_similarity_scores[:num_recommendations]



    # Return the top N recommendations as a list of tuples containing the index and combined similarity score
    return top_recommendations



