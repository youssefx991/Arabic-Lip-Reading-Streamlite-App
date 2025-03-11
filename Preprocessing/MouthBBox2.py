import cv2
import mediapipe as mp

# Define mouth landmark indices (MediaPipe Face Mesh topology)
# MOUTH_LANDMARKS = [
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
#     14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
#     312, 13, 82, 81, 42, 183, 78
# ]
CHIN_LANDMARKS = [
    # Jawline from left to right
    148,152,377,400,132,401,
    # Mouth landmarks
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78
]

def getMouthBBox(video_path: str, landmarks_path: str, output_path: str=""):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Initialize variables for the global bounding box
    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = float('-inf'), float('-inf')
    
    # Lists to collect per-frame bounding boxes (if needed)
    min_Xs = []
    max_Xs = []
    min_Ys = []
    max_Ys = []
    
    # List to store landmarks for each frame
    # Each element is a tuple: (frame_number, [list_of_landmark_tuples])
    landmarks_per_frame = []
    
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # print("Finished processing the video for bounding box calculation.")
                break

            # Convert frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Face Mesh
            results = face_mesh.process(rgb_frame)

            # Process landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Local bounding box for this frame
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')

                    # List to store mouth landmarks for the current frame
                    mouth_landmarks = []

                    # Iterate through mouth landmarks to calculate local bounding box
                    for idx in CHIN_LANDMARKS:
                        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        mouth_landmarks.append((x, y))
                    
                    # Append the landmarks for this frame (with its frame number)
                    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    landmarks_per_frame.append((frame_no, mouth_landmarks))

                    # Update global bounding box
                    min_Xs.append(min_x)
                    min_Ys.append(min_y)
                    max_Xs.append(max_x)
                    max_Ys.append(max_y)
                    
                    global_min_x = min(global_min_x, min_x)
                    global_min_y = min(global_min_y, min_y)
                    global_max_x = max(global_max_x, max_x)
                    global_max_y = max(global_max_y, max_y)

    # Close the video capture after processing
    cap.release()

    # After processing all frames, write the adjusted landmarks to the text file.
    # with open(landmarks_path, "w") as f_landmarks:
    #     f_landmarks.write(f"Global Min: ({global_min_x}, {global_min_y})\n")
    #     f_landmarks.write(f"Global Max: ({global_max_x}, {global_max_y})\n")
    #     f_landmarks.write("\n")
    #     for frame_no, landmarks in landmarks_per_frame:
    #         # Adjust each landmark by subtracting global minimum values
    #         adjusted_landmarks = [(x - global_min_x, y - global_min_y) for x, y in landmarks]
    #         f_landmarks.write(f"Frame {frame_no}:\n")
    #         f_landmarks.write(" ".join([f"({x},{y})" for x, y in adjusted_landmarks]) + "\n")
    
    return min_Xs, min_Ys, max_Xs, max_Ys

# # Example usage:
# if __name__ == "__main__":
#     video_path = "path/to/your/video.mp4"   # Replace with your video file path
#     landmarks_path = "landmarks.txt"         # File where adjusted landmarks will be saved
#     getMouthBBox(video_path, landmarks_path)