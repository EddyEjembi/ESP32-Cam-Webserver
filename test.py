import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import label_map_util

# Load the SSD MobileNetV2 FPN Lite model from TensorFlow Hub
#model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
#model = hub.load(model_url)

model = tf.saved_model.load("C:/Users/HP/Documents/Arduino/esp32-cam WebServer/saved_model")

# Load label map
label_map_path = "C:/Users/HP/Documents/Arduino/esp32-cam WebServer/label_map.pbtxt"  # label map file
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

stream_url = 'http://192.168.43.63'
# Initialize the video capture
video_capture = cv2.VideoCapture(0)

#cv2.namedWindow('ESP32-CAM Stream', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('ESP32-CAM Stream', 640, 480)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to read frame from stream")
        break
    
    frame = cv2.resize(frame, (320, 320))

    # Preprocess the frame
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, tf.uint8)  # Convert to uint8
    input_tensor = tf.convert_to_tensor(input_image)

    # Perform object detection
    detections = model(input_tensor)

    # Process the detections
    num_detections = int(detections["num_detections"][0])
    detection_classes = detections["detection_classes"][0].numpy().astype(np.uint8)
    detection_scores = detections["detection_scores"][0].numpy()
    detection_boxes = detections["detection_boxes"][0].numpy()

    # Display bounding boxes and labels on the frame
    for i in range(num_detections):
        if detection_scores[i] >= 0.5:  # Only consider detections with score above a threshold
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            class_id = detection_classes[i]
            class_name = category_index[class_id]['name']

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
