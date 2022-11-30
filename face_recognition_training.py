import os
import cv2 as cv
import numpy as np


people = []
# we can loop over through the folder.
for i in os.listdir(r'C:\Users\Rjbha\Desktop\New folder\photos'):
    people.append(i)
Dir = r'C:\Users\Rjbha\Desktop\New folder\photos'




features = [] # image array and faces
labels = [] # correspong labels/whos face the image belongs to
haar_cascade = cv.CascadeClassifier('haar_face.xml') # instantiating CascadeClassifier
# creating a function
# it will loop over the base folder
# and then again it will loop over the every folder in the base folder
# and in the folder it will loop over every image and add it to training
def create_train():
    for person in people:
        path = os.path.join(Dir, person) # joining the Dir with person(inside the folder with images)
        label = people.index(person)
        # its gonna loop over the images
        for img in os.listdir(path):
            img_path = os.path.join(path, img)# its gonna grab the image path, and join path variable with image

            # now we gonna read that image from this path
            img_array = cv.imread(img_path)
            # convert it to grayscale
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # face detection
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # now we can loop over every faces in faces_rect

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w] # basically here we are croping out face in the image.(faces in region of interest)

                # now we have the region of faces in images, now we can append it to
                # features list and corresponding labels list
                features.append(faces_roi)
                labels.append(label) # label is index of people list (refer line no 32)
                # the idea of reducing label from string to numeric is because to reduce the strain that your computer
                # have and idea behind this is to create some sort of mapping between string and numeric values

create_train()
print('Training done---------------')
# print(f' length of the feature list: {len(features)}')
# print(f' length of the labels list: {len(labels)}')
# now we have features list and labels list
# and now we can train our model


# before training the model we should convert the features and labels list to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)
# instanntiate face_recognizer

face_recognizer = cv.face.LBPHFaceRecognizer_create() # instantiating face recognizer


# train the face_recognizer on the features list and lables list

face_recognizer.train(features, labels)

# save features list and labels list
np.save('features.npy', features)
np.save('lables.npy', labels)


face_recognizer.save('face_trained.yml')

# use this trianed face_trained.yml in face_recognition.py file







