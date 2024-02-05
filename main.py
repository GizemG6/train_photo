import cv2
import dlib
import pickle

# Eğitilmiş modeli yükle
with open("face_model.pkl", "rb") as file:
    model = pickle.load(file)

# Yüz tespiti için dlib modelini yükle2
detector = dlib.get_frontal_face_detector()

# Yüz özellik çıkarma için dlib modelini yükle
predictor = dlib.shape_predictor("/shape_predictor_68_face_landmarks.dat")

# Yüz tanıma modelini oluştur
face_recognizer = dlib.face_recognition_model_v1("/dlib_face_recognition_resnet_model_v1.dat")

# Kamera başlat
cap = cv2.VideoCapture(2)

while True:
    # Kameradan bir çerçeve al
    ret, frame = cap.read()

    # Gri tonlamaya çevir (Dlib yüz tespiti siyah-beyaz resimle daha iyi çalışır)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        face_encoding = face_recognizer.compute_face_descriptor(frame, shape)

        # Eğitilmiş modele göre yüzü tanı
        matches = dlib.chinese_whispers_clustering([face_encoding, *model["encodings"]], 0.6)

        # Eşleşen yüzlerin olduğu küme sayısını al
        num_clusters = len(set(matches))

        # Eğer sadece bir küme varsa ve kümenin etiketi 0 ise
        if num_clusters == 1 and matches[0] == 0:
            # Tanılanan kişinin adını yaz
            cv2.putText(frame, model["names"][0], (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Yüzün etrafına dikdörtgen çiz
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    # Çerçeveyi göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapat
cap.release()

# Pencereyi kapat
cv2.destroyAllWindows()
