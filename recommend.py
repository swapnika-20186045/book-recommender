from flask import Flask, request, session, redirect, url_for, abort, render_template, flash
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'BOOK-RECOMMENDER'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "./data/" directory.

# import os
# print(os.listdir("./data"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv('./data/books.csv', encoding = "ISO-8859-1")
# print(books.head())
# print(books.shape)
# print(books.columns)

ratings = pd.read_csv('./data/ratings.csv', encoding = "ISO-8859-1")
# print(ratings.head())
# print(ratings.shape)
# print(ratings.columns)

book_tags = pd.read_csv('./data/book_tags.csv', encoding = "ISO-8859-1")
# print(book_tags.head())
# print(book_tags.shape)
# print(book_tags.columns)

tags = pd.read_csv('./data/tags.csv', encoding = "ISO-8859-1")
# print(tags.head())
# print(tags.shape)
# print(tags.columns)

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
# print(tags_join_DF.head())
# print(tags_join_DF.shape)
# print(tags_join_DF.columns)

tf_authors = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_authors = tf_authors.fit_transform(books['authors'])
cosine_sim_authors = linear_kernel(tfidf_matrix_authors, tfidf_matrix_authors)
# print(cosine_sim_authors)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of book authors
def authors_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_authors[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

# print(authors_recommendations('The Hobbit'))

books_with_genres = pd.merge(books, tags_join_DF, left_on='book_id', right_on='book_id', how='inner')
# print(books_with_genres.head())
# print(books_with_genres.shape)

tf_genres = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_genres = tf_genres.fit_transform(books_with_genres['tag_name'].head(10000))
cosine_sim_genres = linear_kernel(tfidf_matrix_genres, tfidf_matrix_genres)

# Function that get book recommendations based on the cosine similarity score of books tags
def genres_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_genres[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

# print(genres_recommendations('The Hobbit'))

@app.route('/', methods=['GET'])
def getAllBooks():
    return books.to_json(orient='table')
    # return render_template('login.html', error="")#, id = "", data=books, message="")

@app.route('/recommend/author/<string:bookName>/', methods=['GET'])
def getBooksWithAuthor(bookName):
    return authors_recommendations(bookName).to_json(orient='table')
    # return render_template('login.html', error="")#, id = "", data=books, message="")

@app.route('/recommend/genre/<string:bookName>/', methods=['GET'])
def getBooksWithgenre(bookName):
    return genres_recommendations(bookName).to_json(orient='table')
    # return render_template('login.html', error="")#, id = "", data=books, message="")

if __name__ == '__main__':
    app.run(debug=True)
