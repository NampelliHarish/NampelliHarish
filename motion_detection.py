import cv2

# Initialize video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change to the appropriate index if you have multiple cameras

# Initialize variables
prev_frame = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # If there is no previous frame, initialize it
    if prev_frame is None:
        prev_frame = gray
        continue
    
    # Compute absolute difference between the current frame and the previous frame
    frame_delta = cv2.absdiff(prev_frame, gray)
    
    # Apply a threshold to the frame delta
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours
    for contour in contours:
        # If the contour area is too small, ignore it
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Draw the bounding box around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with motion detected
    cv2.imshow('Motion Detection', frame)
    
    # Update the previous frame
    prev_frame = gray
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
