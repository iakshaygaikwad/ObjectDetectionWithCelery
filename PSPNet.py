import tensorflow as tf
import cv2
import numpy as np
from scipy import misc
import imutils # Get contour data
# PSPNet
from model import PSPNet101, PSPNet50
from tools import *

# Main directory that contains list of 
# sub-directory with image frames
# TODO: Change this to your main directory name
main_directory = "images"
# TODO: Change these values to where your model files are
# indoor
ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50,
                'weights_path': './model/pspnet50-ade20k/model.ckpt-0'}
# outdoor
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': './model/pspnet101-cityscapes/model.ckpt-0'}

IMAGE_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def write_solid_segment(directory_name, file_name):
    """ TODO: """
    image_path = main_directory + '/' + directory_name + '/' + file_name
    img = cv2.imread(image_path)
    mask = cv2.imread('./masks/' + directory_name + '/' + file_name, 0)
    result = cv2.bitwise_and( img, img, mask = mask)
    create_directory('./solid/raw/' + directory_name)
    cv2.imwrite('./solid/raw/' + directory_name + '/' + file_name, result)

def process_image(predict_output, file_name, directory_name):
    """ Image processing """
    # 1: Save output as mask image
    create_directory('./mask/' + directory_name)
    cv2.imwrite('./masks/' + directory_name + '/' + file_name, predict_output)
    # 2. Write segmented image with solid color
    write_solid_segment(directory_name, file_name)
    # 3. Write segmented image with grayscale
    # TODO: write gray segment
    # 4. Write cropped solid color image
    # 5. Write cropped grayscale image

def predict(directory):
    """ PSPNet prediction and image segmentation """
    for image in os.listdir(directory):
        image_path = directory + '/' + image
        if is_sub_file(image, image_path):
            # TODO For indoor data, change this value to `ADE20k_param`
            param = cityscapes_param
            # Load images and calculate shape
            img_np, fileame = load_img(image_path)
            img_shape = tf.shape(img_np)
            
            h,w = ( tf.maximum(param['crop_size'][0], img_shape[0]),
                    tf.maximum(param['crop_size'][1], img_shape[1]))
            img = preprocess(img_np, h, w)
            # Initialize network and model
            PSPNet = param['model']
            net = PSPNet({ 'data': img},
                            is_training = False,
                            num_classes = param['num_classes'])
            raw_output = net.layers['conv6']
            # Start prediction
            raw_output_up = tf.image.resize_bilinear(   raw_output,
                                                        size = [h, w],
                                                        align_corners = True)
            raw_output_up = tf.image.crop_to_bounding_box(  raw_output_up, 0, 0, 
                                                            img_shape[0],
                                                            img_shape[1])
            raw_output_up = tf.argmax(raw_output_up, dimension = 3)
            prediction = decode_labels( raw_output_up,
                                            img_shape,
                                            param['num_classes'])
            # Init tf session and enable GPU mode
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session( config = config)
            init = tf.global_variables_initializer()
            session.run(init)
            ckpt_path = param['weights_path']
            # Load tensorflow server
            loader = tf.train.Saver(var_list = tf.global_variables())
            loader.restore(session, ckpt_path)
            print("Restored model parameters from {}".format(ckpt_path))
            # Run and get results image
            preds = session.run(prediction)
            process_image(preds[0], image, directory)
            tf.reset_default_graph()

def is_sub_file(subfile, folder_path):
    if not subfile.startswith('.') and os.path.isfile(folder_path):
        return True
    return False

def is_sub_directory(subfolder, folder):
    if not subfolder.startswith('.') and os.path.isdir(os.path.join(folder, subfolder)):
        return True
    return False

def main():
    """ Main program """
    for image_directory in os.listdir(main_directory):
        
        if is_sub_directory(image_directory, main_directory):
            image_path = './' + main_directory + '/' + image_directory
            predict(image_path)

if __name__ == "__main__":
    main()