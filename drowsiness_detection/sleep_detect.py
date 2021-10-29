
from keras.models import load_model
import numpy as np
import keras
from imutils import face_utils
import time
import dlib
import cv2
import face_recognition
################## PHAN DINH NGHIA CLASS, FUNCTION #######################



# Class dinh nghia vi tri 2 mat con nguoi
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Ham du doan mat dong hay mo
def predict_eye_state(model, image):
    # Resize anh ve 20x10
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)
    # Chuyen thanh tensor
    image_batch = np.reshape(image, (1, 10, 20, 1))
    # Dua vao mang mobilenet de xem mat dong hay mo
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
    return np.argmax(model.predict(image_batch)[0])


################ CHUONG TRINH CHINH ##############################

# Load model dlib de phat hien cac diem tren mat nguoi - lansmark
predictor = dlib.shape_predictor('68_face_landmarks_predictor.dat')

# Load model predict xem mat nguoi dang dong hay mo
model = load_model('weights.149-0.01.hdf5')

# Lay anh tu Webcam

scale = 0.5
countClose = 0
currState = 0
timeclose=0;
alarmThreshold = 5
alarmThresholds = 200
cap = cv2.VideoCapture(0)
#thêm
timeNgap=0
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector() #Load face detector
vs = cv2.VideoCapture(0)
#hết thêm
while (True):
    c = time.time()
    # Doc anh tu webcam va chuyen thanh RGB
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # thêm nè
    dets = detector(image, 0)
    for rect in dets:
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        landmark = predictor(image, rect)
        lines = []
        for k, d in enumerate(landmark.parts()):
            # xác định khung miệng
            if (k >= 60 and k <= 68):
                lines.append((d.x, d.y))
                # tìm điểm trung bình line
            # tìm điểm trung bình line
        x_line = round((lines[4][0] + lines[0][0]) / 2)
        y_line = round((lines[4][1] + lines[0][1]) / 2)
        # tính toán khoảng cách
        # Khoảng cách đến môi trên
        u_y = (lines[2][1] - y_line) * (lines[2][1] - y_line)
        # Khoảng cách đến môi dưới
        d_y = (lines[6][1] - y_line) * (lines[6][1] - y_line)
        # kết luận
        if abs(u_y - d_y) > 25:
            timeNgap+=1
        else:
            timeNgap=0
    # hết nè
    # Resize anh con 50% kich thuoc goc
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Chuyen sang he mau LAB de lay thanh lan Lightness
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    # Tim kiem khuon mat bang HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    # Neu tim thay it nhat 1 khuon mat
    if len(face_locations):
        timeclose=0;
        # Lay vi tri khuon mat
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)
        #thêm nè

        #hết nè
        # Trich xuat vi tri 2 mat

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
        left_eye = gray[y:y + h, x:x + w]

        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        right_eye = gray[y:y + h, x:x + w]

        # Dung mobilenet de xem tung mat la MO hay DONG

        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'

        print('left eye open: {0}    right eye open: {1}'.format(left_eye_open, right_eye_open))

        # Neu 2 mat mo thi hien thi mau xanh con khong thi mau do
        if left_eye_open == 'yes' and right_eye_open == 'yes':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #currState = 0
            countClose = 0
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #currState = 1
            countClose +=1
    else:
        timeclose+=0.2
    frame = cv2.flip(frame, 1)
    if countClose > alarmThreshold:
        cv2.putText(frame, "De nghi tap trung lai xe", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        timeclose=0
    if timeclose >alarmThreshold:
        cv2.putText(frame, "chua co mat nguoi nha", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    if timeNgap >6 :
        cv2.putText(frame, "khong duoc ngap", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow('Sleep Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
