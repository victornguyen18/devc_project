from flask import Flask, request, render_template, jsonify, make_response
import logging
import os
from werkzeug.utils import secure_filename
import datetime
import base64
import binascii
from controller.template_checking import TemplateChecking
from controller.facial_verification import FacialVerification, FaceVerify
from controller.perspective_transform import PerspectiveTransform as OCR
import cv2 as cv
import imutils

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
error_handler = logging.FileHandler('{}/logs/errors.log'.format(PROJECT_HOME))

app = Flask('DevC-Project-BrokenHeart', template_folder='{}/templates'.format(PROJECT_HOME))
app.config.from_envvar('APP_SETTINGS')
app.logger.addHandler(error_handler)
app.logger.setLevel(logging.INFO)
app.config['APPLICATION_ROOT'] = PROJECT_HOME
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    new_path = local_dir
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


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


@app.route('/json-image-post/', methods=['POST'])
def json_image_post():
    print("Running")
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

        cccd_front_file = '{}/{}_cccd_front.jpg'.format(path_uploads, time_now)
        cccd_front_file_scale = '{}/{}_cccd_front_scale.jpg'.format(path_uploads, time_now)
        cccd_behind_file = '{}/{}_cccd_behind.jpg'.format(path_uploads, time_now)
        cccd_behind_file_scale = '{}/{}_cccd_behind_scale.jpg'.format(path_uploads, time_now)
        cccd_portrait_file = '{}/{}_cccd_portrait.jpg'.format(path_uploads, time_now)
        cccd_portrait_file_scale = '{}/{}_cccd_portrait_scale.jpg'.format(path_uploads, time_now)
        try:
            cccd_front_data = base64.b64decode(req_data['image1'])
            cccd_behind_data = base64.b64decode(req_data['image2'])
            cccd_portrait_data = base64.b64decode(req_data['image3'])
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
        with open(cccd_portrait_file, 'wb') as f:
            f.write(cccd_portrait_data)
            f.close()
        cccd_image = cv.imread(cccd_front_file, cv.IMREAD_COLOR)
        cccd_image_image = imutils.resize(cccd_image, height=500)
        cv.imwrite(cccd_front_file_scale, cccd_image_image)
        portrait_image = cv.imread(cccd_portrait_file, cv.IMREAD_COLOR)
        portrait_scale_image = imutils.resize(portrait_image, height=500)
        cv.imwrite(cccd_portrait_file_scale, portrait_scale_image)

        # Template Checking
        result_template_checking_file = '{}/{}_result_template_checking.jpg'.format(path_uploads, time_now)
        try:
            print("Start Template checking")
            template_checking_model = TemplateChecking(cccd_front_file)
            message_template_checking = template_checking_model.processing(result_template_checking_file)
            warped_image_front = template_checking_model.cccd_warped
            print("Finish Template checking")
        except Exception as e:
            data = {
                'status': 400,
                'message': str(e) + ".Something is wrong. Please contact your admin!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)

        # Facial Verification
        try:
            print("Start Facial Verification")
            face_model = FaceVerify()
            img1, img2, message_facial_distance = face_model.get_distance(cccd_front_file_scale,
                                                                          cccd_portrait_file_scale)
            print("Finish Facial Verification")
            if not message_facial_distance:
                data = {
                    'status': 400,
                    'message': "Something in facial verification is wrong. Please contact your admin!!",
                    'time': str(datetime.datetime.now())
                }
                return make_response(jsonify(data), 200)
        except Exception as e:
            data = {
                'status': 400,
                'message': str(e) + ".Something is wrong. Please contact your admin!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)

        # OCR
        try:
            print("Start OCR")
            message_ocr = OCR(cccd_front_file).processing()
            print("Finish OCR")
        except Exception as e:
            data = {
                'status': 400,
                'message': str(e) + ".Something is wrong. Please contact your admin!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)

        # with open(result_template_checking_file, "rb") as image_file:
        #     image_1_result = str(base64.b64encode(image_file.read()), 'utf-8')
        data = {
            'status': 200,
            'message_template_checking': str(message_template_checking),
            'message_OCR': str(message_ocr),
            'message_facial': str(message_facial_distance),
            'message': "Successful",
            # 'image1Result': image_1_result,
            'time': str(datetime.datetime.now())
        }
        print(data)
        return make_response(jsonify(data), 200)


@app.route('/get-image-json/', methods=['POST'])
def get_image_json():
    if request.method == 'POST' and request.files['image']:
        try:
            img = request.files['image']
            img_name = secure_filename(img.filename)
            create_new_folder(app.config['UPLOAD_FOLDER'])
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            app.logger.info("saving {}".format(saved_path))
            img.save(saved_path)
            with open(saved_path, "rb") as image_file:
                image_1_result = str(base64.b64encode(image_file.read()), 'utf-8')
            f = open("{}/image_json.txt".format(PROJECT_HOME), "w")
            f.write(image_1_result)
            f.close()
            return "{}<br>Write image json successful". \
                format(str(datetime.datetime.now()))
        except Exception as e:
            return "{}<br>{}.<br>Something is wrong. Please contact your admin!!!". \
                format(str(datetime.datetime.now()), str(e))
    else:
        return "Where is the image?"


@app.route('/', methods=['GET'])
def display_json():
    return render_template('homepage.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
