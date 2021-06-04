# Import libraries
import cv2
import numpy as np


class FrameEditing:
    # Make original frame smaller
    @staticmethod
    def scale_frame(frame, f_x, f_y):
        # Return smaller frame
        return cv2.resize(frame, (0, 0), fx=f_x, fy=f_y)

    # Convert BGR color into HSV color
    @staticmethod
    def convert_frame_to_hsv(frame):
        # Return HSV color frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert BGR color into Gray scale color
    @staticmethod
    def convert_frame_to_gray(frame):
        # Return gray frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show only the skin color
    @staticmethod
    def show_skin_color(frame, hsv_frame, lower_color_range, upper_color_range):
        """
            mask ~ 0/1 color pixel is a skin pixel or not
                0 : Input pixel is not a skin pixel
                1 : Input pixel is a skin pixel
        """

        # Make a mask with the color that we want to show ~ Get only 0 or 1 result
        mask = cv2.inRange(hsv_frame, np.array(lower_color_range), np.array(upper_color_range))

        # Return result
        return cv2.bitwise_and(frame, frame, mask=mask)

    # Combine frame ~ Show two frame next to each other horizontally
    @staticmethod
    def combine_two_frame(frame1, frame2):
        # Return combined frame
        return np.concatenate((frame1, frame2), axis=1)  # Put two frames next to each other

    # Combine four frames into one
    @staticmethod
    def combine_four_frame(original_frame, width, height, frame1, frame2, frame3, frame4):
        # Create an empty frame to put four small frame inside
        frame = np.zeros(original_frame.shape, np.uint8)

        # Put frame inside the empty frames
        frame[:height // 2, :width // 2] = frame1  # Top left corner
        frame[height // 2:, :width // 2] = frame2  # Bottom left corner
        frame[:height // 2, width // 2:] = frame3  # Top right corner
        frame[height // 2:, width // 2:] = frame4[:, :, None]  # Bottom right corner ~ Add a 3 dimension to the array if is a gray scale frame [:, :, None]

        # Return the empty frame with the four frame inside
        return frame
