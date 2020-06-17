import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haar_files/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haar_files/haarcascade_eye.xml')

# For each person, enter one numeric face id
student_name = input('\n Enter name of student ==>  ')
roll_no = input('\n Enter roll no for student ==>')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

if student_name.isalpha() and roll_no.isnumeric():
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_detector.detectMultiScale(
                roi_gray,
                scaleFactor=1.5,
                minNeighbors=10,
                minSize=(5, 5),
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                cv2.imwrite("images/"+student_name+'.'+ str(roll_no) + '.' + str(count) + ".jpg", gray)
                count += 1
            cv2.imshow('video', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 100:  # Take 30 face sample and stop video
            break

    cam.release()
    cv2.destroyAllWindows()
