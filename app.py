import os
import logging
import traceback
import datetime
import base64

from flask import Flask, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from controller.template_checking import TemplateChecking
from controller.facial_verification import FaceVerify
from controller.perspective_transform import PerspectiveTransform
from controller.preprocessing_image import PreprocesingImage

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
STATIC_FOLDER = '{}/static/'.format(PROJECT_HOME)

app = Flask('DevC-Project-BrokenHeart',
            template_folder='{}/templates'.format(PROJECT_HOME),
            static_folder='{}/static'.format(PROJECT_HOME),
            static_url_path='/static')
app.config.from_envvar('APP_SETTINGS')
app.config['APPLICATION_ROOT'] = PROJECT_HOME + "3123"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
handler = logging.FileHandler('{}/logs/errors.log'.format(PROJECT_HOME))
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


def create_new_folder(local_dir):
    new_path = local_dir
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def error_handling(e, trace_back=True):
    if trace_back:
        error_message = traceback.format_exc()
    else:
        error_message = str(e)
    app.logger.error(error_message)
    data = {
        'status': 400,
        'message': "Some thing is wrong. Please contact your admin!! --- [{}]".format(str(e)),
        'time': str(datetime.datetime.now())
    }
    return make_response(jsonify(data), 200)


@app.route('/', methods=['GET'])
def display_json():
    return render_template('homepage.html')


@app.route('/decode-image/', methods=['POST'])
def decode_image():
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
            app.logger.error('Unhandled Exception: %s', str(e))
            return "{}<br>{}.<br>Something is wrong. Please contact your admin!!!". \
                format(str(datetime.datetime.now()), str(e))
    else:
        return "Where is the image?"


@app.route('/json-image-post/', methods=['POST', 'GET'])
def json_image_post():
    logging.info("Request image is running")
    if request.method == 'POST':
        req_data = dict(request.get_json())
        if 'image1' not in req_data or 'image2' not in req_data or 'image3' not in req_data:
            data = {
                'status': 400,
                'message': "Please submit 3 picture!!!",
                'time': str(datetime.datetime.now())
            }
            return make_response(jsonify(data), 200)

        # Location for save image
        date_now = datetime.datetime.now().strftime("%d.%m.%Y")

        path_uploads = UPLOAD_FOLDER + date_now
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_new_folder(path_uploads)

        folder_save_path = STATIC_FOLDER + date_now
        create_new_folder(folder_save_path)
        file_save_path = '{}/{}_result.jpg'.format(folder_save_path, time_now)

        cccd_front_file = '{}/{}_cccd_front.jpg'.format(path_uploads, time_now)
        cccd_front_file_scale = '{}/{}_cccd_front_scale.jpg'.format(path_uploads, time_now)
        # cccd_behind_file = '{}/{}_cccd_behind.jpg'.format(path_uploads, time_now)
        # cccd_behind_file_scale = '{}/{}_cccd_behind_scale.jpg'.format(path_uploads, time_now)
        cccd_portrait_file = '{}/{}_cccd_portrait.jpg'.format(path_uploads, time_now)
        cccd_portrait_file_scale = '{}/{}_cccd_portrait_scale.jpg'.format(path_uploads, time_now)

        # Decode image from base 64
        try:
            cccd_front_data = base64.b64decode(req_data['image1'])
            # cccd_behind_data = base64.b64decode(req_data['image2'])
            cccd_portrait_data = base64.b64decode(req_data['image3'])
        except Exception as e:
            return error_handling(e, True)

        # Save image after decode from base 64
        with open(cccd_front_file, 'wb') as f:
            f.write(cccd_front_data)
            f.close()
        # with open(cccd_behind_file, 'wb') as f:
        #     f.write(cccd_behind_data)
        #     f.close()
        with open(cccd_portrait_file, 'wb') as f:
            f.write(cccd_portrait_data)
            f.close()

        # Resize image
        try:
            PreprocesingImage.scale_image_with_path(cccd_portrait_file, 500, cccd_portrait_file_scale)
            warped_image = PreprocesingImage.crop_card(cccd_front_file, 500)
            image_cccd_front = PreprocesingImage.scale_image_with_image(warped_image, 500,
                                                                        [cccd_front_file_scale])
        except Exception as e:
            return error_handling(e, True)

        # Template Checking
        # result_template_checking_file = '{}/{}_result_template_checking.jpg'.format(path_uploads, time_now)
        try:
            logging.info("Start Template checking")
            template_checking_model = TemplateChecking(cccd_front_file)
            message_template_checking, image_cccd_front_result = template_checking_model.processing()
            # message_template_checking, image_cccd_front_result = TemplateChecking.processing_with_image(
            #     image_cccd_front, True)
            PreprocesingImage.write_image(image_cccd_front_result, file_save_path)
            logging.info("Finish Template checking")
        except Exception as e:
            return error_handling(e, True)

        # Facial Verification
        try:
            logging.info("Start Facial Verification")
            face_model = FaceVerify()
            img1, img2, message_facial_distance = face_model.get_distance(cccd_front_file_scale,
                                                                          cccd_portrait_file_scale)
            logging.info("Finish Facial Verification")
            if not message_facial_distance:
                return error_handling("In facial verification")
        except Exception as e:
            return error_handling(e, True)

        # OCR
        try:
            logging.info("Start OCR")
            message_ocr = PerspectiveTransform(cccd_front_file). \
                processing_without_preprocessing_image(warped_image, True)
            logging.info("Finish OCR")
        except Exception as e:
            return error_handling(e, True)

        data = {
            'status': 200,
            'message_template_checking': str(message_template_checking),
            # 'message_OCR': str(message_ocr),
            # 'message_facial': str(message_facial_distance),
            'message': "Successful",
            'image_result': "{}/static/{}/{}_result.jpg".format(app.config['APP_URL'], date_now, time_now),
            'time': str(datetime.datetime.now())
        }
        logging.info(data)
        return make_response(jsonify(data), 200)
    else:
        return render_template('homepage.html')


@app.route('/json-image-post-test/', methods=['POST'])
def json_image_post_test():
    with open("uploads/2019-09-30_00-47-05_cccd_front_scale.jpg", "rb") as image_file:
        # image_result = "data:image/jpeg;base64," + str(base64.b64encode(image_file.read()), 'utf-8')
        image_result = str(base64.b64encode(image_file.read()), 'utf-8')
    data = {
        'status': 200,
        'message_template_checking': 'Testing template_checking',
        'message_OCR': "Testing OCR",
        'message_facial': "Testing facial",
        'message': "Successful",
        'image_result': "",
        'time': str(datetime.datetime.now())
    }
    logging.info(data)
    return make_response(jsonify(data), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
