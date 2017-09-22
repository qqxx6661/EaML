import cv2
import sys
import time
bodyCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
img_root = '/home/zhendongyang/PycharmProjects/EaML/HDA_data_process/HDAimage/camera50/'

for i in range(2000):
    # Capture frame-by-frame
    frame = cv2.imread(img_root+str(i+1)+'.jpg')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=2,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(w, h)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()
