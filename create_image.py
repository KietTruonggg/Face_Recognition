import tensorflow as tf
import cv2
import numpy as np
import align.detect_face


with tf.Graph().as_default():
    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

# open webcam
capture = cv2.VideoCapture(0)
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 128
saved = 0

# loop through frames
while capture.isOpened():
    # read frame from webcam
    status, frame = capture.read()
    frame = cv2.flip(frame,1)


    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    for idx, f in enumerate(bounding_boxes):
        (startX, startY) = max(0,int(f[0])), max(0,int(f[1]))
        (endX, endY) = int(f[2]), int(f[3])

        face_crop = frame[startY-50:endY+50, startX-50:endX+50] #Face crop
        path = "data/Kiet/{}.png".format(saved)
        saved+=1
        cv2.imwrite(path,face_crop)
        print(f"saved image to {path}")
    cv2.imshow("Video", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
capture.release()
cv2.destroyAllWindows()
