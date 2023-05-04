import cv2
import numpy as np

import joblib

# open webcam
capture = cv2.VideoCapture(1)
face_recogniser = joblib.load('model/face_recognizer.pkl')

# loop through frames
while capture.isOpened():
    # read frame from webcam
    status, frame = capture.read()
    frame = cv2.flip(frame,1)
    tmp = face_recogniser(frame)
    if tmp is not None:
        bbs,predictions,classes = tmp
        for idx, f in enumerate(bbs):
            (startX, startY) = max(0, int(f[0])), max(0, int(f[1]))
            (endX, endY) = int(f[2]), int(f[3])
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            max_idx = np.argmax(predictions[idx])
            if predictions[idx,max_idx] > 0.5:
                label = classes[max_idx]
            else:
                label = 'Unknown'
            print(predictions)
            label = "{}: {:.2f}%".format(label, predictions[idx,max_idx] * 100)
            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)


    cv2.imshow("Video", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
capture.release()
cv2.destroyAllWindows()
