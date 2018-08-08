import numpy as np
import argparse
import sys
import os
import pylab as plt
import cv2
import time
import progressbar

sys.path.insert(0, 'src')
import spin_image

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="Path to point cloud", required=True)
    parser.add_argument("-i", "--image_size", help="Size of one image side", type=int, required=True)
    parser.add_argument("-r", "--radius", help="Number of training epochs", type=float, required=True)
    parser.add_argument("-s", "--save_dir", help="Path to save directory", required=True)
    args = parser.parse_args()

    return args

def read_asc_data(data_path):

    print '[INFO] Reading in data...'

    data = np.loadtxt(data_path, dtype=float, delimiter=',')
    labels = data[:, 3].astype(int)
    data = data[:,:3]
    return data, labels

def main(data_path, image_size, radius, save_dir):

    data, labels = read_asc_data(data_path)

    for c in np.unique(labels):
        if not os.path.exists(os.path.join(save_dir, str(c))):
            os.makedirs(os.path.join(save_dir, str(c)))

    print '[INFO] Generating spin images...'
    start = time.time()
    bar = progressbar.ProgressBar(maxval=len(data), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i, pnt in enumerate(data):
        bar.update(i+1)
        p = spin_image.PointClass(x=pnt[0],y=pnt[1],z=pnt[2])
        si = spin_image.spin_pnt(p=p,                                          \
                                 p_ind=i,                                      \
                                 p_cloud=data,                                 \
                                 radius=radius,                                \
                                 bin_size=(float(radius)/float(image_size-1)), \
                                 bilinear_interp=True)
        cv2.imwrite(save_dir+'/'+str(labels[i])+'/si_'+str(i)+'.png', si)

    bar.finish()
    print '[INFO] Processed 1,000,000 images in', (time.time()-start)/60,'minutes'

if __name__ == '__main__':

    args = parse_args()

    data_path = args.data_path
    image_size = args.image_size
    radius = args.radius
    save_dir = args.save_dir

    main(data_path, image_size, radius, save_dir)
