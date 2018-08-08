import numpy as np
import sys
import cv2
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import argparse
from keras.preprocessing import image
import pylab as plt

sys.path.insert(0, 'src')
import spin_image
import generate_spin_images

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_path", help="Path to point cloud file", required=True)
    parser.add_argument("-m", "--model_path", help="Path to .h5 model", required=True)
    parser.add_argument("-s", "--image_size", help="Size of model image input", type=int, required=True)
    parser.add_argument("-r", "--radius", help="Size of model image input", type=float, required=True)
    parser.add_argument("-o", "--output_path", help="Path to .h5 model", required=True)
    args = parser.parse_args()

    return args

def predict(data_path, model_path, image_size, radius, output_path):

    data, gt = generate_spin_images.read_asc_data(data_path)
    data = data[0:1000]

    print '[INFO] Loading model...'
    model = load_model(model_path)

    labels = []

    for i, pnt in enumerate(data):
        p = spin_image.PointClass(x=pnt[0],y=pnt[1],z=pnt[2])
        si = spin_image.spin_pnt(p=p,                                          \
                                 p_cloud=data,                                 \
                                 radius=radius,                                \
                                 bin_size=(float(radius)/float(image_size-1)), \
                                 bilinear_interp=True)
        si = cv2.cvtColor(si, cv2.COLOR_GRAY2RGB)
        x = image.img_to_array(si)
        x = np.expand_dims(si, axis=0)
        pred_class = model.predict(x)

        labels.append(pred_class)
    labels = np.array(labels)
    #data = np.concatenate((data, labels.T), axis=1)

if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    model_path = args.model_path
    image_size = args.image_size
    output_path = args.output_path
    radius = args.radius
    predict(data_path, model_path, image_size, radius, output_path)
