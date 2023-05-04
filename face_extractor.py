import torch
from facenet_pytorch import InceptionResnetV1
import align.detect_face
from torchvision import models, transforms
import tensorflow as tf
from PIL import Image

import numpy as np
import cv2
MINSIZE = 20
THRESHOLD = [0.7, 0.8, 0.8]
FACTOR = 0.709
IMAGE_SIZE = 128
with tf.Graph().as_default():
    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

class Whitening(object):
    """
    Whitens the image.
    """

    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        y = (img - mean) / std
        return y
class FaceFeaturesExtractor:
    def __init__(self):
        self.facenet = InceptionResnetV1(pretrained='vggface2',classify=False).eval()
        self.transform = transforms.Compose([
            transforms.Resize(size=256,antialias= None),
            transforms.CenterCrop(size=224)

        ])
    def preprocessor(self,img):
        mean = img.mean()
        std = img.std()
        out = (img - mean) / std

        return out
    def extract_face(self,img,bb):
        (startX, startY) = max(0, int(bb[0])), max(0, int(bb[1]))
        (endX, endY) = int(bb[2]), int(bb[3])
        Whiten = Whitening()

        face = np.copy(img[startY:endY, startX:endX])
        # face = Whiten(face)
        return transforms.Resize(size = (256,256),antialias= None)(torch.from_numpy(face).permute(2, 0, 1))

    def extract_feature(self,img):
        bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        if len(bounding_boxes) == 0:
            # if no face is detected
            return None

        # for idx, f in enumerate(bounding_boxes):
        #     print(bounding_boxes)
        #     (startX, startY) = max(0, int(f[0])), max(0, int(f[1]))
        #     (endX, endY) = int(f[2]), int(f[3])
        #
        #     face = np.copy(img[startY:endY, startX:endX])
        faces = torch.stack([self.extract_face(img, bb) for bb in bounding_boxes])

        # faces = self.preprocessor(faces)
        # print(processed_face.shape)
        processed_faces = self.transform(faces)
        # print(processed_face.shape)
        embeddings = self.facenet(processed_faces.float()/255).detach().numpy()

        return bounding_boxes,embeddings

    def __call__(self,img):
        return self.extract_feature(img)
# if __name__ == "__main__":
#     face_extractor = FaceFeaturesExtractor()
#     img_path = r"C:\Users\Admin\Documents\detectFakeFace\Face_Recognition\pic.png"
#     img = cv2.imread(img_path)
#
#     print(face_extractor(img).argmax())