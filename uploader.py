# TODO: CSS Optimization
# TODO: Documentation

import os
from random import randint
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import processor_config as conf

ALLOWED_EXTENSIONS = set(['xls','xlsx','csv'])

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = conf.UPLOAD_FOLDER
app.secret_key = 'fz89n4c983qmrc09q'

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username=conf.USER_SQL,
    password=conf.PWD_SQL,
    hostname=conf.HOST_SQL,
    databasename=conf.DB_SQL,
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Work(db.Model):
    __tablename__ = "works"
    id = db.Column(db.Integer, primary_key=True)
    input_file = db.Column(db.String(4096))
    output_file = db.Column(db.String(4096))
    email = db.Column(db.String(4096))
    targets = db.Column(db.String(4096))
    complete = db.Column(db.Boolean, default=False)
    submission = db.Column(db.DateTime)
    completion = db.Column(db.DateTime)
    notes = db.Column(db.String(4096))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error = None
    if request.method == 'POST':
        # checks if file was uploaded
        if 'file' not in request.files or request.files['file']=='':
            error = 'Error! No selected file'
            return render_template('index.html', message = error)
        file = request.files['file']
        # checks if email was submitted
        if 'email' not in request.form or request.form['email']=='':
            error = 'Error! Email not specified'
            return render_template('index.html', message = error)
        email = request.form['email']
        # checks if targets was submitted
        if 'targets' not in request.form or request.form['targets'] == '':
            error = 'Error! Target not specified'
            return render_template('index.html', message = error)
        targets = request.form['targets']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            stamp = str(randint(1, 10**6))
            filename = filename.replace(".","_"+stamp+".")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            work = Work(input_file=filename,output_file=None,
                        email=email,targets=targets,
                        submission=datetime.now(),completion=None)
            db.session.add(work)
            db.session.commit()
            message = "File saved!\nWe will send soon the results to "+email
            return render_template('index.html', message = message)
    return render_template('index.html', message = error)

if __name__ == '__main__':
    app.run()