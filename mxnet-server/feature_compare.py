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

import detect_face

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
        sym = all_layers['pre_fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.9]
        self.factor = 0.85
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            self.sess = tf.Session()
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        self.src = src

    def alignment(self, rimg, landmark,Debug=False):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB2BGR)
        if Debug:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        return img

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

    def detect_max_face(self,img,Debug=False):
            _minsize = self.minsize
            _bbox = None
            _landmark = None
            bounding_boxes, points = detect_face.detect_face(img, _minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                bindex = 0
                if nrof_faces > 1:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    bindex = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                if Debug:
                    show = img.copy()
                    cv2.rectangle(show,(int(_bbox[0]),int(_bbox[1])),(int(_bbox[2]),int(_bbox[3])),color=(255,255,0))
                    for idx in range(_landmark.shape[0]):
                        cv2.circle(show,(_landmark[idx][0], _landmark[idx][1]),2,(0,255,255),2)
                    cv2.imshow('landmarks',show)
                    cv2.waitKey(0)

            return _bbox, _landmark

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
        img1 = self.get_single_image(img_path1)
        img2 = self.get_single_image(img_path2)
        if img1 is None or img2 is None:
            print("read image failed!please check image source and path")
        bbox1, landmarks1 = self.detect_max_face(img1, Debug=False)
        bbox2, landmarks2 = self.detect_max_face(img2, Debug=False)

        face1 = self.alignment(img1, landmarks1, Debug=False)

        face2 = self.alignment(img2, landmarks2, Debug=False)

        # face1 = img1
        # face2 = img2
        # cv2.imshow('img1',face1)
        # cv2.imshow('img2', face2)
        # cv2.waitKey(0)
        feature1 = self.get_feature(face1)
        feature2 = self.get_feature(face2)

        np.save("../mxnet-server/images/B.npy", feature1)
        # print(feature1)
        # print(feature2)
        print("Is same person --> ",self.compare(feature1, feature2, 0.46))



if __name__ == "__main__":
    _embedding = Embedding('/home/hanson/Documents/heils-git/code/mobileFacenet-ncnn/best',0)
    pth1 = '/home/hanson/Downloads/eclipse/images/save01.jpg'
    pth2 = '/home/hanson/Downloads/eclipse/images/save01.jpg'
    _embedding.example(pth1, pth2)
