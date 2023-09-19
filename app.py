from flask import Flask, render_template, request, flash, redirect, url_for, session

import requests

import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = b'_5#y2HOLZA"F4Q8z\n\xec]/'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
url = "https://jsearch.p.rapidapi.com/search"

headers = {"X-RapidAPI-Key": "43a5b8691fmsh5de5fc34ed72a45p1db322jsn4ff7db5e8e93","X-RapidAPI-Host": "jsearch.p.rapidapi.com"}

pagedict = {1 : ['do.html','Devops developer/Engineer'] , 2 : ['ds.html','Data Scientist'], 3 : ['se.html','Security Engineer'], 4 : ['sqe.html','Software Quality Engineer, Tester'], 5: ['ce.html','Cloud Engineer'], 6 : ['de.html','Back End Developer'], 7 : ['ae.html','Android Developer'], 8 : ['ux.html','UI/UX Designer/Developer'], 9 : ['bc.html','Blockchain Developer']}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apireq(st):
    quersa = "Fresher " + st + " in India"
    querystring = {"query":quersa,"page":"1","num_pages":"1"}
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()



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
    an = apireq(pagedict[ds][1])
    return render_template(pagedict[ds][0], data = an['data'])


@app.route('/recommend', methods=['POST'])
def recommend():

    if request.method == 'POST':
        resume_text = request.files['resume']
        # check if the post request has the file part
        if resume_text.filename == '':
            flash('No file Uploaded. Upload a file and proceed!')
            return redirect(url_for('recpage'))
        elif allowed_file(resume_text.filename):
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

        else:
            flash('Not a PDF or DOCX File. Try Again')
            return redirect(url_for('recpage'))


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
