#! usr/bin/python
from flask import Flask, request, render_template, jsonify, make_response
import logging
import os
from werkzeug.utils import secure_filename
import datetime
import base64
import binascii

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))

app = Flask('fe_project', template_folder='{}/templates'.format(PROJECT_HOME))  # still relative to module
file_handler = logging.FileHandler('{}/server.log'.format(PROJECT_HOME))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

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
            'email': "thang181997@gmail.com",
            'time': str(datetime.datetime.now())
        }
        return make_response(jsonify(data), 200)
        # return send_from_directory(app.config['UPLOAD_FOLDER'], img_name, as_attachment=True)
    else:
        return "Where is the image?"


@app.route('/json-post/', methods=['POST'])
def json_post():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST':
        req_data = request.get_json()
        print(req_data)
        # f = open("{}/json_request.txt".format(PROJECT_HOME), "a+")
        f = open("{}/json_request.txt".format(PROJECT_HOME), "w")
        f.writelines(str(datetime.datetime.now()))
        f.writelines(str(req_data))
        f.writelines("\n=========================\n")
        f.close()
        data = {
            'method': "POST",
            'function': "json post",
            'message': "successful",
            'time': str(datetime.datetime.now())
        }
        return make_response(jsonify(data), 200)


@app.route('/json-image-post/', methods=['POST'])
def json_image_post():
    if request.method == 'POST':
        req_data = dict(request.get_json())
        if 'image1' not in req_data or 'image2' not in req_data or 'image3' not in req_data:
            data = {
                'status': 400,
                'message': "Please submit 3 picture!!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)
        path_uploads = UPLOAD_FOLDER + datetime.datetime.now().strftime("%d.%m.%Y")
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_new_folder(path_uploads)
        cccd_front_file = '{}/cccd_front_{}.jpg'.format(path_uploads, time_now)
        cccd_behind_file = '{}/cccd_behind_{}.jpg'.format(path_uploads, time_now)
        cccd_selfie_file = '{}/cccd_selfie_{}.jpg'.format(path_uploads, time_now)
        try:
            cccd_front_data = base64.b64decode(req_data['image1'])
            cccd_behind_data = base64.b64decode(req_data['image2'])
            cccd_selfie_data = base64.b64decode(req_data['image3'])
        except binascii.Error:
            data = {
                'status': 400,
                'message': "Some thing is wrong. Please contact your admin!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)
        with open(cccd_front_file, 'wb') as f:
            f.write(cccd_front_data)
            f.close()
        with open(cccd_behind_file, 'wb') as f:
            f.write(cccd_behind_data)
            f.close()
        with open(cccd_selfie_file, 'wb') as f:
            f.write(cccd_selfie_data)
            f.close()
        with open("{}/image/done-it.jpg".format(PROJECT_HOME), "rb") as image_file:
            image_1_result = str(base64.b64encode(image_file.read()), 'utf-8')
        data = {
            'status': 200,
            'message': "Successful",
            'image1Result': image_1_result,
            'time': str(datetime.datetime.now())
        }
        return make_response(jsonify(data), 200)


#
# @app.route('/', methods=['GET'])
# def display():
#     return render_template('formPartial.html')


@app.route('/', methods=['GET'])
def display_json():
    with open("{}/json_request.txt".format(PROJECT_HOME), "r") as f:
        content = f.read()
    return render_template('json_request.html', content=content)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
