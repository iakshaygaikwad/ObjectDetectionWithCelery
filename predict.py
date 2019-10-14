
from model import PSPNet101, PSPNet50
from tools import *

# Input and output files
input_directory = "input1";
output_directory = "output1";

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

param = cityscapes_param
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
ckpt_path = param['weights_path']

print("Restored model parameters from {}".format(ckpt_path))


def get_image_objects(item):

    image_path = './' + input_directory + '/' + item
    if os.path.isfile(image_path):

        # NOTE: If you want to inference on indoor data, change this value to `ADE20k_param`

        img_np, filename = load_img(image_path)
        img_shape = tf.shape(img_np)
        h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
        img = preprocess(img_np, h, w)
        # Create network.
        PSPNet = param['model']
        net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])
        raw_output = net.layers['conv6']

        print("RAW OUTPUT",raw_output)
        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        print ("RAW OUTPUT UP",raw_output_up)
        pred = decode_labels(raw_output_up, img_shape, param['num_classes'])
        loader = tf.train.Saver(var_list=tf.global_variables())
        loader.restore(sess, ckpt_path)
        # Run and get results image
        preds = sess.run(pred)


        for i in preds[0]:
            for j in i:
                tt = tuple(j)
                data[names[tt]] = data[names[tt]] + 1

        print(data)

        data_perc = {}
        total = sum(data.values())
        for k,v in data.items():
            data_perc[k]= (v*100)/total

        dddd = { filename : data_perc}
    return dddd

print(get_image_objects("indoor_1.jpg"))