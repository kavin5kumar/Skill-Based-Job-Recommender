from flask import Flask, render_template, request, flash, redirect, url_for, session
import pandas as pd
import nltk
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

app = Flask(__name__)
app.secret_key = b'_5#y2HOLZA"F4Q8z\n\xec]/'


jobs_df = pd.read_csv('job_postings_a.csv')

vectorizer = TfidfVectorizer(stop_words='english')
job_matrix = vectorizer.fit_transform(
    jobs_df['Job Description'].values.astype('U'))
job_titles = jobs_df['Job Title'].tolist()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/jrec', methods=['GET', 'POST'])
def recpage():
    return render_template('recpage.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


@app.route('/links/<int:ds>', methods=['GET', 'POST'])
def links(ds):
    if (ds == 1):
        return render_template('do.html')
    elif (ds == 2):
        return render_template('ds.html')
    elif (ds == 3):
        return render_template('se.html')
    elif (ds == 4):
        return render_template('sqe.html')
    elif (ds == 5):
        return render_template('ce.html')
    elif (ds == 6):
        return render_template('de.html')
    elif (ds == 7):
        return render_template('ae.html')
    elif (ds == 8):
        return render_template('ux.html')
    elif (ds == 9):
        return render_template('bc.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        resume_text = request.files['resume']
        # check if the post request has the file part
        if resume_text.filename == '':
            flash('No file Uploaded. Upload a file and proceed!')
            return redirect(url_for('recpage'))
        else:
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


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
