from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words='english')

# preprocess data
stemmer = SnowballStemmer("english")
stemmed_data = [
    " ".join(stemmer.stem(word) 
              for sent in sent_tokenize(message) 
              for word in word_tokenize(sent))
    for message in newsgroups.data
]

# create DTM
dtm = vectorizer.fit_transform(stemmed_data)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # transform the query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # calculate cosine similarity between the query and the dtm
    similarities = cosine_similarity(query_vector, dtm).flatten()
    
    # get the top 5 indices of the documents with the highest similarity
    top_indices = similarities.argsort()[-5:][::-1]
    
    # get documents and their corresponding similarities
    top_documents = [newsgroups.data[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    
    return top_documents, top_similarities, top_indices.tolist()
    # return documents, similarities, indices 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
