from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

app = Flask(__name__)

jobs_df = pd.read_csv('job_postings.csv')

vectorizer = TfidfVectorizer(stop_words='english')
job_matrix = vectorizer.fit_transform(
    jobs_df['Job Description'].values.astype('U'))
job_titles = jobs_df['Job Title'].tolist()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # get the uploaded resume file
    resume_text = request.files['resume']
    text = resume_text.read().decode('utf-8', errors='ignore')
    resume_matrix = vectorizer.transform([text])

    # calculate cosine similarity between resume and job descriptions
    cosine_similarities = cosine_similarity(
        resume_matrix, job_matrix).flatten()
    print(cosine_similarities)

    # sort jobs by similarity score
    job_indices = cosine_similarities.argsort()[::-1]
    related_jobs = [job_titles[i] for i in job_indices]
    seen = set()
    result = []
    for item in related_jobs:
        if item not in seen:
            seen.add(item)
            result.append(item)
    # render the recommendation results
    return render_template('recommend.html', jobs=result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
