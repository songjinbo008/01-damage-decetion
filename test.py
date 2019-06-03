# -*- coding: utf-8 -*-
from keras.models import Sequential
import tensorflow as tf
from keras.models import load_model
import os
from glob import glob
import numpy as np
import argparse
import math
import scipy.misc
import matplotlib.pyplot as plt

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/255.

def imread(path, is_grayscale = False ,is_buquan = True):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    img=scipy.misc.imread(path).astype(np.float)
    if is_buquan == True:
        lst = [0.] * 512
        for i in range(152):
            img= np.append(img, [lst], axis=0)
    return img

def get_image(image_path, input_height, input_width,is_crop=True, is_grayscale=False,is_buquan = True):
  image = imread(image_path, is_grayscale,is_buquan = is_buquan)
  return transform(image, input_height, input_width,
                   input_height, input_width, is_crop)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def generate(BATCH_SIZE, nice=False):
    print("加载模型")
    g = load_model('my_model.h5')
    # g = multi_gpu_model(g, gpus=4)  # 设置使用GPU数

    data_ty = glob(os.path.join("./data100","yuantu", '*.jpg'))
    for epoch in range(1):
        batch_idxs = int(len(data_ty)/BATCH_SIZE)
        print("当前训练次数为", epoch)
        print("训练共分为几批", int(len(data_ty)/BATCH_SIZE))
        for idx in range(0, batch_idxs):

            batch_files_ty = data_ty[idx * args.ceshi_size:(idx + 1) * args.ceshi_size]

            batch_ty = [
                get_image(batch_file,
                          input_height=args.input_height_ty,
                          input_width=args.input_width_ty,
                          is_buquan=False,
                          is_crop=args.is_crop)for batch_file in batch_files_ty]


            batch_images_ty = np.array(batch_ty).astype(np.float32)[:, :, :, None]

            generated_images = g.predict(batch_images_ty, verbose=1)
            image = combine_images(generated_images*255)
            plt.title(batch_files_ty)
            plt.imshow(image, cmap='gray')
            plt.show()







def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="generate",type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ceshi_size", type=int, default=1)
    parser.add_argument("--nice", dest="nice", action="store_true")

    parser.add_argument("--input_height", type=int, default=512)
    parser.add_argument("--input_width", type=int, default=512)
    parser.add_argument("--input_height_ty", type=int, default=512)
    parser.add_argument("--input_width_ty", type=int, default=512)
    parser.add_argument("--is_crop", type=int, default=1)

    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        print("请选择训练")
        # train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.ceshi_size, nice=args.nice)