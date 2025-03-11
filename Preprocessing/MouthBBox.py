import cv2
import mediapipe as mp

# Define mouth landmark indices (MediaPipe Face Mesh topology)
MOUTH_LANDMARKS = [
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

    # Open the text file to save landmarks
    with open(landmarks_path, "w") as f_landmarks:
        with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Finished processing the video for bounding box calculation.")
                    break

                # Convert frame to RGB for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe Face Mesh
                results = face_mesh.process(rgb_frame)

                # Check if face landmarks are detected
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Local bounding box for this frame
                        min_x, min_y = float('inf'), float('inf')
                        max_x, max_y = float('-inf'), float('-inf')

                        # Initialize a list to store mouth landmarks for the current frame
                        mouth_landmarks = []

                        # Iterate through mouth landmarks to calculate local bounding box
                        for idx in MOUTH_LANDMARKS:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)

                            # Add the landmark coordinates to the list
                            mouth_landmarks.append((x, y))

                        # Save the landmarks for this frame to the text file
                        f_landmarks.write(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}:\n")
                        f_landmarks.write(" ".join([f"({x},{y})" for x, y in mouth_landmarks]) + "\n")

                        # Update global bounding box
                        global_min_x = min(global_min_x, min_x)
                        global_min_y = min(global_min_y, min_y)
                        global_max_x = max(global_max_x, max_x)
                        global_max_y = max(global_max_y, max_y)

    # Close the video capture after the first pass
    cap.release()

    if output_path != "":
        # Second pass: Draw global bounding box on all frames and save the video
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Finished saving the video with global bounding box.")
                break

            # Draw the global bounding box on the frame
            cv2.rectangle(frame, (global_min_x, global_min_y), (global_max_x, global_max_y), (0, 255, 0), 2)

            # Write the frame to the output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # # Print the global bounding box values
    # print("Global Bounding Box Coordinates:")
    # print(f"Top-Left: ({global_min_x}, {global_min_y})")
    # print(f"Bottom-Right: ({global_max_x}, {global_max_y})")

    # print(f"Video with global bounding box saved to {output_path}")

    return global_min_x, global_min_y, global_max_x, global_max_y


# # usage Example
# video_path = r"D:\Materials\Study\Graduation Project\lip-reading\Datasets\01\LRW-AR\train\عليكم\00003004_عليكم.mp4"
# output_path = r"D:\Materials\Study\Graduation Project\lip-reading\New folder\00003004_عليكم_BBox.mp4"
# landmarks_path = r"D:\Materials\Study\Graduation Project\lip-reading\New folder\00003004_عليكم_landmarks.txt"
# getMouthBBox(video_path, landmarks_path, output_path)