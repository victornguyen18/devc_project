from flask import Flask, url_for, send_from_directory, request, escape, render_template, jsonify, make_response
import logging
import os
from werkzeug.utils import secure_filename

# from werkzeug import secure_filename

app = Flask(__name__, template_folder='templates')  # still relative to module
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/image-post/', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        data = {
            'username': "Thang",
            'email': "thang181997@gmail.com"
        }
        return make_response(jsonify(data), 200)
        # return send_from_directory(app.config['UPLOAD_FOLDER'], img_name, as_attachment=True)
    else:
        return "Where is the image?"


@app.route('/', methods=['GET'])
def display():
    return render_template('formPartial.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1:8003', debug=False)
