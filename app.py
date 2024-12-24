import streamlit as st
import requests
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_URL = "https://openlibrary.org/subjects/literature.json"


def load_hybrid_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def get_all_books(per_page=500):
    response = requests.get(API_URL, params={"limit": per_page})
    if response.status_code == 200:
        data = response.json()
        books = data.get("works", [])
        return books
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return []


def recommend_books(title, model, books_data):
    selected_book = next((book for book in books_data if book['title'] == title), None)

    if selected_book:
        
        subjects = [', '.join(book['subject']) if isinstance(book['subject'], list) else book['subject'] for book in books_data]
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(subjects)

        selected_book_index = books_data.index(selected_book)
        content_score = cosine_similarity(tfidf_matrix[selected_book_index], tfidf_matrix).flatten()

        ratings = [book.get('average_rating', 0) for book in books_data]
        normalized_ratings = np.array(ratings) / np.max(ratings) if np.max(ratings) > 0 else np.zeros(len(ratings))

        hybrid_scores = 0.7 * content_score + 0.3 * normalized_ratings
        hybrid_scores = hybrid_scores[:10]

        distances, indices = model.kneighbors(hybrid_scores.reshape(1, -1), n_neighbors=10)

        st.write(f"### Recommended Books for '{title}':")
        for i in indices[0]:
            recommended_book = books_data[i]
            book_title = recommended_book['title']
            book_subject = recommended_book.get('subject', 'No subject available.')
            book_authors = ", ".join([author.get('name', 'Unknown Author') for author in recommended_book.get('authors', [{'name': 'Unknown Author'}])])
            book_cover_id = recommended_book.get('cover_id')

            st.write(f"**Title**: {book_title}")
            st.write(f"**Subject**: {book_subject}")
            st.write(f"**Authors**: {book_authors}")

            if book_cover_id:
                cover_url = f"http://covers.openlibrary.org/b/id/{book_cover_id}-M.jpg"
                st.image(cover_url, width=100)
            else:
                st.write("No cover image available.")


def display_books_dropdown(model):
    st.title("Book Recommender System")

    books = get_all_books(per_page=500)

    if books:
        book_titles = [book['title'] for book in books]

        selected_book = st.selectbox("Select a book:", book_titles)

        if selected_book:
            book = next(book for book in books if book['title'] == selected_book)

            authors = ", ".join([author.get("name", "Unknown Author") for author in book.get("authors", [{"name": "Unknown Author"}])])
            publish_year = book.get("first_publish_year", "Unknown Year")
            cover_id = book.get("cover_id")
            subject = ", ".join(book.get("subject", ["Unknown Subject"]))
            publisher = ", ".join(book.get("publishers", ["Unknown Publisher"]))

            if cover_id:
                cover_url = f"http://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                st.image(cover_url, width=100)
            else:
                st.write("No cover image available.")

            st.write(f"**{selected_book}**")
            st.write(f"**Authors**: {authors}")
            st.write(f"**Published Year**: {publish_year}")
            st.write(f"**Publisher(s)**: {publisher}")
            st.write(f"**Subject(s)**: {subject}")

            if st.button(f"Recommend for '{selected_book}'"):
                recommend_books(selected_book, model, books)
    else:
        st.write("No books available.")


if __name__ == "__main__":
    model_path = "D:/Book Recommender System - Copy/hybrid_recommender_model.pkl"
    model = load_hybrid_model(model_path)
    display_books_dropdown(model)
