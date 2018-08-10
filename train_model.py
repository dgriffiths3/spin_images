import math, json, os, sys
import tensorflow as tf
import argparse
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

def parse_args():

    '''
    Parse user input arguments

    Returns:
    args: parser args
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", help="Path to test data directory", required=True)
    parser.add_argument("-i", "--image_size", help="Size of one image side", type=int, default=224)
    parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("-g", "--num_gpu", help="Number of GPUs to use. Note: Batch size must be divisable by GPUs", default=1)
    parser.add_argument("-l", "--log_dir", help="Directory to save tensorboard logs", default='./logs')
    parser.add_argument("-o", "--output_model", help="Name of output model (.h5)", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")

    return args

def train_resnet(data_dir, image_size, epochs, batch_size, log_dir, output_model):

    '''
    Train a ResNet50 model and save the .h5 output to output_model

    Inputs:
    data_dir: String
    image_size: Int
    epochs: Int
    batch_size: Int
    output_model: String

    data_dir structure:

    - data_dir
        |_ train
            |_ class
                : :
            |_ class_n
        |_ val
            |_ class
                : :
            |_ class_n
    '''

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    size = (image_size, image_size)
    batch_size = batch_size

    len_train_images = sum([len(files) for r, d, files in os.walk(train_dir)])
    len_val_images = sum([len(files) for r, d, files in os.walk(val_dir)])

    num_train_steps = math.floor(len_train_images/batch_size)
    num_val_steps = math.floor(len_val_images/batch_size)

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(train_dir, target_size=size, class_mode='categorical', shuffle=True, batch_size=batch_size)
    val_batches = val_gen.flow_from_directory(val_dir, target_size=size, class_mode='categorical', shuffle=True, batch_size=batch_size)

    classes = list(iter(batches.class_indices))

    with tf.device('/cpu:0'):

        model = keras.applications.resnet50.ResNet50(weights=None)

    if num_gpu >= 2:
        try:
            model = multi_gpu_model(model, gpus=num_gpu)
            print '[INFO] Multi-GPU enabled, training on %s GPUs' %num_gpu
        except:
            print '[INFO] Multi-GPU failed, training on single CPU/GPU'

    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    resnet_model = Model(model.input, x)
    resnet_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    resnet_model.classes = classes

    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    resnet_model.fit_generator(batches,
                               steps_per_epoch=len_train_images/batch_size,     \
                               epochs=epochs,                        \
                               callbacks=[checkpointer, tbCallBack], \
                               validation_data=val_batches,          \
                               validation_steps=num_val_steps)
    resnet_model.save(output_model)

if __name__ == '__main__':

    args = parse_args()

    data_dir = args.data_dir
    image_size = args.image_size
    epochs = args.epochs
    batch_size = args.batch_size
    num_gpu = args.num_gpu
    log_dir = args.log_dir
    output_model = args.output_model

    train_resnet(data_dir, image_size, epochs, batch_size, log_dir, output_model)
