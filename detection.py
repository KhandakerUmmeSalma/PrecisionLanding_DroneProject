import cv2
import numpy as np
import math

# Function to initialise the camera
def initialise_camera():
    # OpenCV VideoCapture with the Pi camera
    camera = cv2.VideoCapture(0)  # Adjust index if necessary
    if not camera.isOpened():
        raise Exception("Could not open the camera.")
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)  # Ensure high FPS for real-time detection
    return camera

# Function to estimate pose of detected ArUco markers
def estimate_pose(frame, corners, ids, camera_matrix, distortion_coeffs, marker_length):
    # Loop through all detected markers
    for corner, marker_id in zip(corners, ids):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corner, marker_length, camera_matrix, distortion_coeffs
        )
        
        # Draw axis for the detected marker
        cv2.aruco.drawAxis(frame, camera_matrix, distortion_coeffs, rvec, tvec, 0.1)
        
        # Print the marker's translation vector
        print(f"Marker ID {marker_id}: Position (x, y, z) -> {tvec[0][0]}")
    return frame

# Main script
if __name__ == "__main__":
    try:
        # Initialise the camera
        camera = initialise_camera()

        # Load the predefined ArUco dictionary
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        
        # Set detection parameters
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Load the camera calibration parameters
        # Replace with your camera's actual calibration data
        camera_matrix = np.array([[620, 0, 320], [0, 620, 240], [0, 0, 1]])  # Example values
        distortion_coeffs = np.array([0.1, -0.05, 0, 0])  # Example values

        # Marker size in meters (adjust to your marker's actual size)
        marker_length = 0.15  # 15 cm

        print("Starting ArUco detection. Press 'q' to quit.")

        while True:
            # Capture frame from the camera
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters)

            # If markers are detected
            if ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                frame = estimate_pose(frame, corners, ids, camera_matrix, distortion_coeffs, marker_length)
                print(f"Detected IDs: {ids.flatten()}")
            else:
                print("No markers detected.")

            # Display the frame
            cv2.imshow("ArUco Detection", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the camera and close windows
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
