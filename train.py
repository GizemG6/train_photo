import cv2
import dlib
import os

# Eğitim verilerinin bulunduğu dizin
dataset_path = "photos"

# Yüz tespiti için dlib modelini yükle
detector = dlib.get_frontal_face_detector()

# Yüz tanıma modelini eğitmek için dlib modelini yükle
predictor = dlib.shape_predictor("/shape_predictor_68_face_landmarks.dat")

# Yüz tanıma modelini oluştur
face_recognizer = dlib.face_recognition_model_v1("/dlib_face_recognition_resnet_model_v1.dat")

# Eğitim verilerini yükle
faces_encodings = []
face_names = []

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            # Resmi oku
            image = cv2.imread(image_path)

            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Yüz tespiti
            faces = detector(gray)

            # Eğer yüz bulunduysa, yüz özelliklerini çıkar ve adı ekleyerek kaydet
            if len(faces) > 0:
                face = faces[0]
                shape = predictor(gray, face)
                face_encoding = face_recognizer.compute_face_descriptor(image, shape)

                faces_encodings.append(face_encoding)
                face_names.append(person_name)

# Modeli eğit
model = {"encodings": faces_encodings, "names": face_names}

# Modeli kaydet
import pickle
with open("/face_model.pkl", "wb") as file:
    pickle.dump(model, file)
