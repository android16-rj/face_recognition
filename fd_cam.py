!pip install opencv-contrib-python
import numpy as np
import cv2 as cv

# create the CascadeClassifier() instance

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Conor McGregor', 'Israel Adesanya', 'Laura Sanko', 'Pooja Hegde', 'Sean Strickland']
# loading the features and labels array

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('lables.npy', allow_pickle=True)

# reading the face_trained.yml file


face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')


# video capture

video_capture = cv.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	face_rect = haar_cascade.detectMultiScale(gray,
				scaleFactor=1.1,
				minNeighbors = 4)
	for (x,y,w,h) in face_rect:
		faces_roi = gray[y:y+h, x:x+w]
		label, confidence = face_recognizer.predict(faces_roi)
		print(f'Label = {people[label]} with a confidence of {confidence}.')
		cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0,(0,255,0), thickness=2)
		cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2) # we can draw rectangle over the image
		cv.imshow('Detected face: ', frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
# release the capture
video_capture.release()
cv2.destroyAllWindows()
