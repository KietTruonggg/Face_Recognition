import cv2
import numpy as np
import imutils
import joblib


img_path = 'test.jpg'
img=cv2.imread(img_path)
img = imutils.resize(img, width=400, height= 400)
print(img.shape)
face_recogniser = joblib.load('model/face_recognizer.pkl')
tmp = face_recogniser(img)
if tmp is not None:
    bbs,predictions,classes = tmp
    for idx, f in enumerate(bbs):
        (startX, startY) = max(0, int(f[0])), max(0, int(f[1]))
        (endX, endY) = int(f[2]), int(f[3])
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # if np.max(predictions) > 0.4:
        max_index = np.argmax(predictions[idx])
        label = classes[max_index]
        print(predictions)
        label = "{}: {:.2f}%".format(label, predictions[idx,max_index] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        # else:
        #     idx = np.argmax(predictions)
        #     label = classes[idx]
        #
        #     label = "{}: {:.2f}%".format(label, predictions[0, idx] * 100)
        #     Y = startY - 10 if startY - 10 > 10 else startY + 10
        #     cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.7, (0, 255, 0), 2)


cv2.imshow("Recognized Image", img)

cv2.waitKey(0)   #wait for a keyboard input

cv2.destroyAllWindows()
