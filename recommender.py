import requests
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_books_from_api(query, max_results=10):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    response = requests.get(url)
    books = response.json()
    
    book_data = []
    for item in books.get('items', []):
        book = {
            'title': item['volumeInfo'].get('title', 'No Title'),
            'subject': item['volumeInfo'].get('categories', ['No Subject']),  
            'authors': item['volumeInfo'].get('authors', ['Unknown']),
            'averageRating': item['volumeInfo'].get('averageRating', 0),
            'ratingsCount': item['volumeInfo'].get('ratingsCount', 0),
        }
        book_data.append(book)
    
    return book_data

def create_hybrid_model(query, model_save_path):
    
    books = get_books_from_api(query)
    df = pd.DataFrame(books)
    
    
    df = df.dropna(subset=['subject', 'averageRating', 'ratingsCount'])
    df['subject'] = df['subject'].fillna('No subject provided')  
    df['ratingsCount'] = df['ratingsCount'].replace(0, 1)  
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['subject'].apply(lambda x: ' '.join(x)))  # subjects are now lists, so join them into a string
    
    
    ratings = df[['title', 'averageRating', 'ratingsCount']]
    ratings.loc[:, 'normalizedRating'] = ratings['averageRating'] / ratings['ratingsCount']  # Fixed setting warning
    

    hybrid_scores = []
    for i in range(len(df)):
        content_score = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
        collaborative_score = ratings['normalizedRating'].values
        
        hybrid_score = 0.7 * content_score + 0.3 * collaborative_score
        hybrid_scores.append(hybrid_score)
    
    hybrid_scores_matrix = np.array(hybrid_scores).T
    
    
    hybrid_scores_matrix = np.nan_to_num(hybrid_scores_matrix)
    
    
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    model.fit(hybrid_scores_matrix)
    
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at {model_save_path}")


