import cv2
cap = cv2.VideoCapture("sources/video.mkv")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    output_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Frame", output_resized)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()