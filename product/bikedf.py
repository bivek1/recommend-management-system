    #Import all the required packages
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def getDG():
    # Load the .csv file into a Pandas dataframe
    df = pd.read_csv('europe-motorbikes-zenrows.csv')
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

    return df