import cv2

url = 'http://192.168.43.63'
cap = cv2.VideoCapture(url)

cv2.namedWindow('ESP32-CAM Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ESP32-CAM Stream', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from stream")
        break

    cv2.imshow('ESP32-CAM Stream', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
