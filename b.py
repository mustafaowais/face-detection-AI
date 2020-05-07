import cv2
import pickle
import numpy

video_capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_label = pickle.load(f)
	labels = {v:k for k,v in og_label.items()}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]

    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=45:
    		print(id_)
    		print(labels[id_])
    		#print("door unlocked")
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		#if labels[id_] == True:
    			#print("door unlocked")
    	img_item = "my-image.png"
    	cv2.imwrite(img_item, roi_color)

    	color = (255, 0, 0)
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x,  end_cord_y), color, stroke)

    	

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()