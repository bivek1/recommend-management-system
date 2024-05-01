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
import matplotlib.pyplot as plt
import matplotlib
from django.contrib.auth import login, logout, authenticate
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from .test import item_based_rec
import datetime
from django.contrib import messages
import operator

matplotlib.use('Agg')
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
    similarity_data = []
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


        def create_similarity_pie_chart(row):
            print("This is row")
            print(row)
            # Prepare labels for the pie chart
            labels = [f'Recommendation {i+1}' for i in range(len(row))]

            # Create a figure and axis for the chart
            fig, ax = plt.subplots()

            # Plot a pie chart
            ax.pie(row['id'], labels=labels, autopct='%1.1f%%', startangle=90)

            # Add a title
            ax.set_title('Similarity Scores of Recommendations')

            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            temp_file = f'/Users/bibekdhakal/DJANGO/recomend/suggest/media/piechart.png'
            plt.savefig(temp_file)
            
            # Close the figure to release resources
            plt.close(fig)

            return temp_file

            # # Return the figure object
            # return fig

        # Example usage:
        # similarity_scores = [0.2, 0.4, 0.6, 0.8, 1.0]  # Replace with your actual similarity scores
        # fig = create_similarity_pie_chart(similarity_scores)
       
       

        def create_similarity_chart(row):
            # Create a figure and axis for the chart
            fig, ax = plt.subplots()
            print(row)

            # Plot similarity data (replace this with your actual data)
            # similarity_data = [0.2, 0.4, 0.6, 0.8, 1.0]  # Example data
            ax.plot(similarity_data)

            # Customize the chart (labels, title, etc.)
            ax.set_xlabel('Similarity')
            ax.set_ylabel('Index')
            ax.set_title('Similarity Chart')

            # Save the chart to a temporary file
            temp_file = f'/Users/bibekdhakal/DJANGO/recomend/suggest/media/{row["id"]}_similarity_chart.png'
            plt.savefig(temp_file)
            
            # Close the figure to release resources
            plt.close(fig)

            # Return the path to the saved chart
            return temp_file

        # Define the number of recommendations you want
        num_recommendations = 10

        # Call the recommend_system function
        recommendations = recommend_system(df, preferences, num_recommendations)
        print(recommendations)

        
        # Output the recommendations
        print("Top recommendations based on numeric and text-based similarity:")
        # Create a list of indices from the recommendations
        recommended_indices = [idx for idx, score in recommendations]
        # recommended_indices = [idx for idx, score in recommendations]
        scores = [score for idx, score in recommendations]

        # Zip recommended indices with scores
        indices_and_scores = zip(recommended_indices, scores)

        # Append to similarity_data
        similarity_data = []
        for idx, score in indices_and_scores:
            similarity_data.append(score)

        # Use the list of indices to filter the original DataFrame
        filtered_df = df.loc[recommended_indices]

        # Display the filtered DataFrame with the recommended entries
        print(filtered_df)
        df_new = pd.DataFrame(filtered_df)
        create_similarity_pie_chart(df_new)

        # Convert DataFrame to HTML table
        html_table = df_new.to_html(index=False)

        # Print HTML table
        print(html_table)

    
        def create_row_html(row):
            # Create HTML for link
            link_html = f'<a href="http://127.0.0.1:8000/details/{row["id"]}">Details</a>'
            # Create HTML for image
            similarity_chart_path = create_similarity_chart(row)
            file_name = similarity_chart_path.split("media/")[-1]
            image_html = f'<a href="http://127.0.0.1:8000/media/{file_name}" target="_blank"><img src="http://127.0.0.1:8000/media/{file_name}" alt="Similarity Chart"></a>'

            # Create HTML for table row
            row_html = '<tr>'
            for value in row:
                row_html += f'<td>{value}</td>'
            # Add details link and image to the row
            row_html += f'<td>{link_html}</td>'
            row_html += f'<td>{image_html}</td>'
            row_html += '</tr>'
            return row_html

        html_table = '<table border="1">'
        html_table += '<thead><tr>'
        html_table += ''.join([f'<th>{column}</th>' for column in df_new.columns])
        html_table += '<th>Details</th>'
        html_table += '<th>Similarity Chart</th>'
        html_table += '</tr></thead>'
        html_table += '<tbody>'
        html_table += ''.join(df_new.apply(create_row_html, axis=1))
        html_table += '</tbody></table>'

            
        dist = {
            'data':html_table
        }
        print(similarity_data)
        print("tuhshshahshdasdd similar")
    return render(request, "index.html", dist)



def productView(request, id):
    global df
   
    
    
    LogReport.objects.create( user = request.user, product = id)
    specific_data = df.loc[df['id'] == id]
    # Get recommendations
    
    # Load user log data
    logs = LogReport.objects.all()

    # Create a DataFrame from user logs
    data = []
    for log in logs:
        data.append({'user_id': log.user_id, 'product_id': log.product, 'timestamp': log.dateTime})
    # interactions_df = pd.DataFrame(data)

    # print(data)
    converted_data = []
    for item in data:
        user_id = item['user_id']
        product_id = int(item['product_id'])  # Convert product ID to integer
        timestamp = item['timestamp'].strftime('%Y-%m-%dT%H:%M:%SZ')  # Format timestamp as string
        converted_data.append((user_id, product_id, timestamp))
    print(converted_data)
    
    # Convert data to a pandas DataFrame
    interactions_df = pd.DataFrame(converted_data, columns=['user_id', 'product_id', 'timestamp'])

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
    user_item_matrix = interactions_df.pivot_table(index='user_id', columns='product_id', values='weight', aggfunc='sum',fill_value=0)
    user_item_matrix.head()
    # print('This is user item matrix')
    # print(user_item_matrix)

    ###### Step 5: Data Normalization
    # Normalize user-item matrix
    matrix_norm = user_item_matrix.subtract(user_item_matrix.mean(axis=1), axis = 0)
    matrix_norm.head()

    ###### Step 6: Calculate Similarity Score

    # Item similarity matrix using Pearson correlation
    item_similarity = matrix_norm.corr()
    item_similarity.head()
   
    def item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations =3):
   
        # Movies that the target user has not watched    
        picked_userid_uninteracted = pd.DataFrame(user_item_matrix.T[request.user.id]==0).reset_index()
        picked_userid_uninteracted = picked_userid_uninteracted[picked_userid_uninteracted[picked_userid] == True]['product_id'].values.tolist()
        
        picked_userid_interacted = pd.DataFrame(matrix_norm.T[request.user.id].dropna(axis=0, how='all')\
                                    .sort_values(ascending=False))\
                                    .reset_index()\
                                    .rename(columns={request.user.id:'rating'})
        # print("Unteractedddddd.d.d.d.d.d.dd.")
        # print(picked_userid_uninteracted)
        
        # print(picked_userid_interacted)
                    
        # Dictionary to save the uninteracted items and predicted rating pair
        rating_prediction ={} 
        # Loop through uninteracted items       
        for picked_item in picked_userid_uninteracted:
            print("pRINTING OICKED ITYEMEMNENDEHSDHFHFJDH")
            print(picked_item)
            # Calculate the similarity score of the picked item with other movies
            picked_item_similarity_score = item_similarity[[picked_item]].reset_index().rename(columns={picked_item:'similarity_score'})
            # Rank the similarities between the picked user interacted item and the picked uninteracted item.
            picked_userid_interacted_similarity = pd.merge(left=picked_userid_interacted, 
                                                        right=picked_item_similarity_score, 
                                                        on='product_id', 
                                                        how='inner')\
                                                .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
            # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
            predicted_rating = round(np.average(picked_userid_interacted_similarity['rating'], 
                                                weights=picked_userid_interacted_similarity['similarity_score']), 6)
            # Save the predicted rating in the dictionary
            rating_prediction[picked_item] = predicted_rating
            # Return the top recommended movies
        return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
    recommended_item = item_based_rec(picked_userid=request.user.id, number_of_similar_items=5, number_of_recommendations =3)
    # recommended_item
    # print(request.user.id)
    print("resulr.....")
    print(recommended_item)
    # print(df)
    # print(interactions_df)
    print(df)
    # for item_id in recommended_item:
    #     filtered_dfs = df[df['id'] == item_id[0]]
    #     if not filtered_dfs.empty:
    #         print(filtered_dfs)
    #     else:
    #         print(f"No data found for item ID {item_id[0]}")
    html_tables = []

    for item_id in recommended_item:
        filtered_dfs = df[df['id'] == item_id[0]]
        if not filtered_dfs.empty:
            html_table = filtered_dfs.to_html(index=False)
            html_tables.append(html_table)
        else:
            html_tables.append(f"<p>No data found for item ID {item_id[0]}</p>")

    html_content = "\n".join(html_tables)
    # # Convert DataFrame to HTML table
    #     html_table = filtered_dfs.to_html(index=False)

    #     # Print HTML table
    #     print(html_table)

    
       

    dist = {
        'product' : specific_data.to_html(index=False),
        'data':html_table
    }
    
    return render(request, "productDetails.html", dist)



def LoginView(request):
    if request.method == 'POST':
        username = request.POST['email_user']
        password = request.POST['password']

        log= authenticate(request, username = username, password = password)

        if log != None:
            login(request, log)
            return HttpResponseRedirect(reverse('product:homepage'))
        else:
            messages.error(request, "Failed to authenticate user")

    
    return render(request, "login.html")

def logoutView(request):
    logout(request)
    return HttpResponseRedirect(reverse('product:login'))

def filterBike(request):
    return render(request, "bike.html")



def filterClothes(request):
    return render(request, "cloth.html")


def filterBook(request):
    return render(request, "book.html")
