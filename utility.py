from flask_wtf import FlaskForm
# from wtforms.fields.html5 import DateField
from wtforms.fields import DateField, EmailField, TelField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms import validators, SubmitField
import os


# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'upload_folder')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class InfoForm(FlaskForm):
    startdate = DateField('Start Date', format='%Y-%m-%d',
                          validators=(validators.DataRequired(),))
    enddate = DateField('End Date', format='%Y-%m-%d',
                        validators=(validators.DataRequired(),))
    submit = SubmitField('Submit')


def upload_file(files):
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
