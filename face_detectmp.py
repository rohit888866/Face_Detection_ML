import cv2
import mediapipe as mp
import os
IMAGE_FILES = os.listdir('images/')
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
os.mkdir("train")
os.mkdir("train/images")
os.mkdir("train/labels")
os.mkdir("train/annotate")
label2=0

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.85) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread("images/"+file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('train/annotate/'+file, annotated_image)
    cv2.imwrite('train/images/'+file, image)
    labels = 'train/labels/'+file[:-4]+".txt"
    
    box_output = "{} {} {} {} {}".format(label2,results.detections[0].location_data.relative_bounding_box.xmin,results.detections[0].location_data.relative_bounding_box.ymin,results.detections[0].location_data.relative_bounding_box.width,results.detections[0].location_data.relative_bounding_box.height)

    with open(labels, 'w') as f:
        f.write(box_output)
