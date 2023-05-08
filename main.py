import numpy as np
import cv2

# Load the reference disk image and convert it to grayscale
ref_disk = cv2.imread("sources/test.png")
ref_disk_gray = cv2.cvtColor(ref_disk, cv2.COLOR_BGR2GRAY)

# Function to rotate an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Load the input video and get video properties
cap = cv2.VideoCapture("sources/video.mkv")
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

    # Ensure at least some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
        # Crop the input image around the detected circle
            cropped = gray[y - r:y + r, x - r:x + r]

            # Initialize a variable to store the maximum correlation value
            max_corr = -1
            max_angle = 0

            # Rotate the reference disk image in steps (e.g., 1 degree) and perform template matching
            for angle in range(0, 360):
                rotated_ref = rotate_image(ref_disk_gray, angle)
                result = cv2.matchTemplate(cropped, rotated_ref, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                # Update the maximum correlation value and corresponding angle
                if max_val > max_corr:
                    max_corr = max_val
                    max_angle = angle

            print(f"Disk at ({x}, {y}) has an initial angle of {max_angle} degrees")

            # Draw the circle and center on the output image
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # Draw the line representing the initial angle of the disk
            end_x = int(x + r * np.cos(np.radians(max_angle)))
            end_y = int(y - r * np.sin(np.radians(max_angle)))
            cv2.line(output, (x, y), (end_x, end_y), (255, 0, 0), 2)

            # Display the angle as text on the output image
            cv2.putText(output, f"{max_angle} deg", (x - 20, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the output image
    output_resized = cv2.resize(output, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("output", output_resized)

    # Keep the imshow window open until the 'q' key is pressed
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
