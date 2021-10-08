import cv2
import numpy as np
from numba import jit


# This command is utilized by the Numba library in order to speed up the code
@jit(nopython=True)
# Function that changes each pixel's value to the arithmetic average of a 3x3x3 pixel neighborhood
# param v: input video
# return: filtered video
def arithmeticAverage(v):
    output = np.zeros(v.shape)
    f, h, w, c = v.shape
    # Initializing the array that will hold the video after 0-padding has been applied
    padded_frame = np.zeros((f + 2, h + 2, w + 2, c))
    # The video is added to the appropriate positions of the new array, in order be 0-padded
    padded_frame[1:padded_frame.shape[0] - 1, 1:padded_frame.shape[1] - 1, 1:padded_frame.shape[2] - 1, :] = v
    for i in range(1, padded_frame.shape[0] - 1):
        for j in range(1, padded_frame.shape[1] - 1):
            for k in range(1, padded_frame.shape[2] - 1):
                for l in range(c):
                    sum = 0
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                # Calculating the sum of each 3x3x3 window
                                sum += padded_frame[i - 1 + m][j - 1 + n][k - 1 + o][l]
                    # Adding the average of each 3x3x3 window to the appropriate position of the output array
                    output[i - 1][j - 1][k - 1][l] = sum / 27
    return output.astype(np.uint8)


# Reading the video
cap = cv2.VideoCapture('input.avi')
if cap.isOpened():
    # Defining the codec and creating a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720), 3)
    # 3D array that will hold each frame of the input video
    video = []
    current_frame_pos = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # Adding each frame to the 3D array
            video.append(frame)
            if cv2.waitKey(24) & 0xFF == ord('q'):
                break
        else:
            break
    # Performing arithmetic average filtering on the input video
    final_video = arithmeticAverage(np.array(video))
    # Displaying the filtered video
    for frame in final_video:
        cv2.imshow('Filtered Video', frame)
        # Saving the filtered video
        out.write(frame.astype(np.uint8))
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
cap.release()
