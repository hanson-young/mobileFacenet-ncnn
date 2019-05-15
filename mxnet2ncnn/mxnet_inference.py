#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import sklearn
import tensorflow as tf
from scipy import misc
import sys
import os

import numpy as np

from skimage import transform as trans
import cv2


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

class Embedding:
    def __init__(self, prefix, epoch, ctx_id=0):
        print('loading', prefix, epoch)
        # ctx = mx.gpu(ctx_id)
        ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get_feature(self, img):
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = embedding[0]
        return embedding

    def compare(self, feature1, feature2, threshold = 0.46):
        num = np.dot(feature1, feature2)
        denom = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        cosine = num / denom
        # print(feature1)
        # print(feature2)
        print("compare score: %f (@ %f)"%(cosine,threshold))
        if cosine > threshold:
            return True
        else:
            return False

    def get_single_image(self,image_path):
        if not os.path.exists(image_path):
            return None
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
            return None
        else:
            if img.ndim < 2:
                print('Unable to align "%s", img dim error' % image_path)
                # text_file.write('%s\n' % (output_filename))
                exit()
            if img.ndim == 2:
                img = to_rgb(img)
            img = img[:, :, 0:3]
        return img

    def example(self,img_path1, img_path2):
        face1 = self.get_single_image(img_path1)
        face2 = self.get_single_image(img_path2)
        if face1 is None or face2 is None:
            print("read image failed!please check image source and path")

        # face1 = img1
        # face2 = img2
        # cv2.imshow('img1',face1)
        # cv2.imshow('img2', face2)
        # cv2.waitKey(0)
        feature1 = self.get_feature(face1)
        feature2 = self.get_feature(face2)
        print(feature1)
        np.save("B.npy", feature1)
        # print(feature1)
        # print(feature2)
        print("Is same person --> ",self.compare(feature1, feature2, 0.46))



if __name__ == "__main__":
    _embedding = Embedding('/home/hanson/Documents/heils-git/code/mobileFacenet-ncnn/best',0)
    pth1 = '/home/hanson/Downloads/eclipse/images/save01.jpg'
    pth2 = '/home/hanson/Downloads/eclipse/images/save01.jpg'
    _embedding.example(pth1, pth2)

    # [6.87451720e-01 - 3.86249125e-01  2.30041265e-01  1.52421200e+00
    #  - 9.02360082e-01  4.43562150e-01  2.97573256e+00 - 5.56485713e-01
    #  9.01465416e-01  8.10879916e-02  6.09591380e-02  1.31748289e-01
    #  - 1.99870571e-01 - 2.09442005e-02 - 6.56891465e-02 - 4.79707807e-01
    #  - 2.59894699e-01 - 1.53992939e+00 - 8.70721698e-01 - 3.86818647e-02

    #  1.00808370e+00 -1.61289668e+00 -1.20417558e-01 -5.45593798e-01
    #  9.87165213e-01  1.38955742e-01  1.46796143e+00 -1.60226035e+00
    #  5.80027521e-01 -5.57141781e-01  3.01999152e-01  3.05885345e-01
    #  1.57867610e-01 -2.93976627e-03  1.25497568e+00  1.65810013e+00
    #  5.19565582e-01  2.86903441e-01  8.58760238e-01 -1.55558038e+00
    # -1.66779482e+00  2.13016748e+00 -1.28412831e+00  4.70681190e-01]