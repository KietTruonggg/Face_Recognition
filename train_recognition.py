from face_extractor import FaceFeaturesExtractor
from face_recognizer import FaceRecognizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
import cv2
import joblib
import argparse

def dataset_to_embeddings(data_dir,feature_extractor):
    embeddings = []
    labels = []

    for folder in os.listdir(data_dir):
        count = 0
        for file in os.listdir(os.path.join(data_dir,folder)):
            file_path = os.path.join(data_dir,folder,file)
            img = cv2.imread(file_path)

            # img = cv2.resize(img,(64,128))
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            a = feature_extractor(img)
            if a is None:
                continue
            count += 1
            _,embedding = a
            for i in range(len(embedding)):
                embeddings.append(embedding[i])
                labels.append(folder)
        print(f"{folder}: {count}")

    np.savetxt("embeddings.txt",embeddings)
    np.savetxt("labels.txt",np.array(labels,dtype=np.str).reshape(-1, 1), fmt="%s")
    return np.stack(embeddings),labels



def train(embeddings, labels):
    # softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    model = SVC(kernel='linear', probability = True, C = 100)
    # clf = softmax
    # print(embeddings.shape)
    # print(labels.shape)
    model.fit(embeddings,labels)


    return model

def main(args):
    datadir = 'data'
    feature_extractor = FaceFeaturesExtractor()

    if args.embeddings_path == "None":
        embeddings, labels = dataset_to_embeddings(datadir, feature_extractor)
    else:
        embeddings = np.loadtxt(args.embeddings_path)
        labels = np.loadtxt(args.labels_path, dtype='str')

    print(embeddings[0,:])
    model = train(embeddings,labels)

    print(model)
    print(model.classes_)
    label_encoder = LabelEncoder()
    label_encoder.fit(model.classes_)
    labels_encoded = label_encoder.transform(labels)
    predict = model.predict(embeddings)
    predict = label_encoder.transform(predict)
    print(predict)
    print(labels_encoded)
    predict = np.asarray(predict, dtype='int').reshape(-1, 1)
    labels_encoded = np.asarray(labels_encoded, dtype='int').reshape(-1, 1)
    print('Độ chính xác: ', accuracy_score(labels_encoded, predict))
    if not os.path.isdir('model'):
        os.mkdir('model')

    model_path = os.path.join('model','face_recognizer.pkl')

    joblib.dump(FaceRecognizer(feature_extractor,model,model.classes_),model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--embeddings_path', required=True,
                        help = "None for already have embeddings file. Type embeddings_path if havent have yet or have new data")
    parser.add_argument('-l', '--labels_path', required=True,
                        help="None for already have labels file. Type labels_path if havent have yet or have new data")
    args = parser.parse_args()
    main(args)

    # python train_recognition.py -e 'embeddings.txt' -l 'labels.txt'
    # python train_recognition.py -e None -l None
