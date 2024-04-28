from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .recomend import recommend_system
from .models import LogReport

df = None

def runNow():
    global df
    file_path = '/Users/bibekdhakal/DJANGO/recomend/suggest/product/europe-motorbikes-zenrows.csv'  # Update the path to your file
    # try:
    #     with open(file_path, 'r') as file:
    #         file_content = file.read()
    # except FileNotFoundError:
    #     file_content = "File not found"
    # Load the .csv file into a Pandas dataframe
    df = pd.read_csv(file_path)
    df = df.drop(columns=['link'])
    df.reset_index(inplace=True)

    # Rename the default index column to 'id'
    df.rename(columns={'index': 'id'}, inplace=True)

    # Print the DataFrame to verify the changes
    print(df)

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

def homepage(request):
    runNow()
    global df
    
    dist = {}
    if request.method == 'POST':
        print(request.POST['search'])
        # Define preferences as a dictionary
        preferences = {
            'price': int(request.POST['price']),  # Preferred price
            'mileage': int(request.POST['milage']),  # Preferred mileage
            'power':int(request.POST['power']) ,  # Preferred power
            'make_model': request.POST['search'],  # Preferred make and model
            'date': '',  # Preferred date
            'fuel': '',  # Preferred fuel type
            'gear': '',  # Preferred gear type
            'offer_type': '',  # Preferred offer type
            'version': ''  # Preferred version
        }

        # Define the number of recommendations you want
        num_recommendations = 10

        # Call the recommend_system function
        recommendations = recommend_system(df, preferences, num_recommendations)
        print(recommendations)
        # Output the recommendations
        print("Top recommendations based on numeric and text-based similarity:")
        # Create a list of indices from the recommendations
        recommended_indices = [idx for idx, score in recommendations]

        # Use the list of indices to filter the original DataFrame
        filtered_df = df.loc[recommended_indices]

        # Display the filtered DataFrame with the recommended entries
        print(filtered_df)
        df = pd.DataFrame(filtered_df)

        # Convert DataFrame to HTML table
        html_table = df.to_html(index=False)

        # Print HTML table
        print(html_table)
        
        

        def create_row_html(row):
        # Create HTML for link
            link_html = f'<a href="http://127.0.0.1:8000/details/{row["id"]}">Details</a>'
            # Create HTML for table row
            row_html = '<tr>'
            for value in row[:-1]:  # Exclude the last column which is the link
                row_html += f'<td>{value}</td>'
            row_html += f'<td>{link_html}</td>'
            row_html += '</tr>'
            return row_html

            
        html_table = '<table border="1">'
        html_table += '<thead><tr>'
        html_table += ''.join([f'<th>{column}</th>' for column in df.columns])
        html_table += '</tr></thead>'
        html_table += '<tbody>'
        html_table += ''.join(df.apply(create_row_html, axis=1))
        html_table += '</tbody></table>'
        
        dist = {
            'data':html_table
        }
    return render(request, "index.html", dist)



def productView(request, id):
    global df
    import random
    nom = LogReport.objects.all()
    rand = random.randint(nom.count(), 20000)
    request.session['user_id'] = nom.count() + rand
    LogReport.objects.create(name  = rand, device = "Chrome", product = id, session = request.session['user_id'] )
    specific_data = df.loc[df['id'] == id]

    filterdLog = LogReport.objects.filter(session = request.session['user_id'])
    for i in filterdLog:
        print(i)
    print(specific_data)
    print("/---------LOG REPORT")
    print(nom)

    dist = {
        'product' : specific_data.to_html(index=False)
    }
    return render(request, "productDetails.html", dist)

def filterBike(request):
    return render(request, "bike.html")



def filterClothes(request):
    return render(request, "cloth.html")


def filterBook(request):
    return render(request, "book.html")