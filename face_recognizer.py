class FaceRecognizer:
    def __init__(self,feature_extractor,classifier,classes):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classes = classes

    def recognize_face(self,img):
        tmp  = self.feature_extractor(img)
        if tmp is None:
            return None
        bbs,embedding = tmp
        # if bbs is None:
        #     return None
        predictions = self.classifier.predict_proba(embedding)
        return bbs,predictions,self.classes



    def __call__(self,img):
        return self.recognize_face(img)
