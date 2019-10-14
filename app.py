import warnings

warnings.filterwarnings('ignore')

import os
from werkzeug.utils import secure_filename
from flask import Flask, request
from celery import task
from model import PSPNet101, PSPNet50
from tools import *
import json
from celery_tasks import make_celery

# Indoor model
ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50,
                'weights_path': './model/pspnet50-ade20k/model.ckpt-0'}
# Outdoor model
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': './model/pspnet101-cityscapes/model.ckpt-0'}

IMAGE_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

data = {'road': 0, 'sidewalk': 0, 'building': 0, 'wall': 0, 'fence': 0, 'pole': 0, 'traffic light': 0,
        'traffic sign': 0, 'vegetation': 0, 'terrain': 0, 'sky': 0, 'person': 0, 'rider': 0, 'car': 0
    , 'truck': 0, 'bus': 0, 'train': 0, 'motocycle': 0, 'bicycle': 0}

names = {(128, 64, 128): 'road', (244, 35, 231): 'sidewalk', (69, 69, 69): 'building', (102, 102, 156): 'wall',
         (190, 153, 153): 'fence', (153, 153, 153): 'pole', (250, 170, 29): 'traffic light', (
             219, 219, 0): 'traffic sign', (106, 142, 35): 'vegetation', (152, 250, 152): 'terrain',
         (69, 129, 180): 'sky', (219, 19, 60): 'person', (255, 0, 0): 'rider', (0, 0, 142): 'car', (0, 0, 69):
             'truck', (0, 60, 100): 'bus', (0, 79, 100): 'train', (0, 0, 230): 'motocycle',
         (119, 10, 32): 'bicycle'}

# Init tf Session


result_folder = "./results"
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='pyamqp://guest@localhost//'
    # ,
    # task_serializer='PICKLE',
    # accept_content=['pickle']
    # CELERY_RESULT_BACKEND='amqp://'
)
celery = make_celery(app)
# celery.conf.task_serializer = 'pickle'

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

result_folder = "./results"


@app.route('/hello')
def hello_world():
    return 'Hello World!'


@app.route('/', methods=['GET', 'POST'])
def extract_objects():
    if request.method == 'POST':
        video_name = request.form.get("video_name")
        video_json_file = result_folder + "/" + video_name + ".json"
        file = request.files['file']

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = "./uploads/" + filename

        object_detect.delay(image_path, video_json_file)

    elif request.method == 'GET':
        video_name = request.args.get("video_name")
        video_json_file = result_folder + "/" + video_name + ".json"

        if os.path.exists(video_json_file):
            with open(video_json_file, "r") as jsonFile:
                images = json.load(jsonFile)
        else:
            return "There is no data for this video file"

        images = images["images"]
        aggregate = data
        # print("IMAGES", images)
        # print("AGG", aggregate)
        length = len(images.values())
        for file_name, d in images.items():
            for k, v in d.items():
                aggregate[k] = aggregate[k] + v

        total_perc = {}
        for k, v in aggregate.items():
            total_perc[k] = v / length

        image_Result = {'images': images, 'aggregate': total_perc}
        return image_Result

    return "Success!"


@task(name="object_detect")
def object_detect(image_path, video_json_file):
   # print(str(filename + ' ' + image_path))
    param = cityscapes_param
    img_np, filename = load_img(image_path)
    img_shape = tf.shape(img_np)
    h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
    img = preprocess(img_np, h, w)
    # Create network.
    PSPNet = param['model']
    net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])
    raw_output = net.layers['conv6']

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)

    pred = decode_labels(raw_output_up, img_shape, param['num_classes'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_path = param['weights_path']

    loader = tf.train.Saver(var_list=tf.global_variables())
    loader.restore(sess, ckpt_path)
    # Run and get results image
    preds = sess.run(pred)

    for i in preds[0]:
        for j in i:
            tt = tuple(j)
            data[names[tt]] = data[names[tt]] + 1

    data_perc = {}
    total_sum = sum(data.values())
    for k, v in data.items():
        data_perc[k] = (v * 100) / total_sum

    os.remove("./uploads/" + filename)

    if os.path.exists(video_json_file):
        with open(video_json_file, "r") as jsonFile:
            json_data = json.load(jsonFile)
        json_data = json_data['images']
        json_data[filename] = data_perc
    else:
        json_data = {}
        json_data[filename] = data_perc

    file_result = {"images": json_data}

    with open(video_json_file, 'w') as outfile:
        json.dump(file_result, outfile)


if __name__ == '__main__':
    app.run(DEBUG=True)
