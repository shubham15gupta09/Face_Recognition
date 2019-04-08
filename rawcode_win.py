import face_recognition
import cv2
import numpy as np
import pandas as pd

video_capture = cv2.VideoCapture(0)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("capture.mp4", cv2.VideoWriter_fourcc(*'VIDX'),20, (width, height))

fout = open("datafile.txt",'w')
name1 , info = [] , []
known_face_encodings , known_face_names = []  , []
df = pd.read_csv("database.csv")

name1 = np.array(df['name'])
info = np.array(df["info"])

for i in name1:
    x = face_recognition.load_image_file( i + ".jpg" )
    known_face_encodings.append(face_recognition.face_encodings(x)[0])
    known_face_names.append(i)

face_locations = []
face_encodings = []
face_names = []
temp = "\n"
process_this_frame = True

while True:
    ret , frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[ : , : , ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
            for i in name1:
                if i == name and temp != name:
                    j = list(name1).index(str(name))
                    string = name, ":", str(info[j]).split(",")
                    fout.write(str(string))
                    fout.write("\n")
                    print(name, ":", str(info[j]).split(","))
                if name == "Unknown" and temp != name:
                    print("Database not found")
            temp = name
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    out.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
fout.close()
video_capture.release()
out.release()
cv2.destroyAllWindows()