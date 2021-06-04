# Import libraries
import cv2


class FrameDrawing:
    # Draw text into the frame
    @staticmethod
    def draw_text(frame, text, org, font, font_scale, color, thickness, line_type=None, bottom_left_origin=None):
        """
            draw text ~ putText(frame, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
                frame: It is the frame on which text is to be drawn.
                text: Text string to be drawn.
                org: It is the coordinates of the bottom-left corner of the text string in the image. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
                font: It denotes the font type. Some of font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, , etc.
                fontScale: Font scale factor that is multiplied by the font-specific base size.
                color: It is the color of text string to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
                thickness: It is the thickness of the line in px.
                lineType: This is an optional parameter.It gives the type of the line to be used.
                bottomLeftOrigin: This is an optional parameter. When it is true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
        """

        # Return the text in the frame
        return cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type, bottom_left_origin)

    # Draw rectangle
    @staticmethod
    def draw_rect(frame, center_position, radius, color, line_thickness):
        """
            draw rect ~ rectangle(frame, center_position, radius, color, line_thickness(-1 to fill))
                frame: draw the rect in this frame
                center_position: (x, y) ~ position of the rect -> (20, 20)
                radius: (width, height) ~ size of the rect -> (200, 200)
                color: (blue, green, red) ~ color of the rect -> (128, 128, 128)
                line_thickness: line thickness of the rect ~ (-1 to fill the rect)
        """

        # Return the rectangle in the frame
        return cv2.rectangle(frame, center_position, radius, color, line_thickness)

    # Draw rectangle
    @staticmethod
    def draw_circle(frame, center_position, radius, color, line_thickness):
        """
            draw circle ~ circle(frame, center_position, radius, color, line_thickness(-1 to fill))
                frame: draw the circle in this frame
                center_position: (x, y) ~ position of the circle -> (20, 20)
                radius: (width, height) ~ size of the circle -> (200, 200)
                color: (blue, green, red) ~ color of the circle -> (128, 128, 128)
                line_thickness: line thickness of the circle ~ (-1 to fill the rect)
        """

        # Return the rectangle in the frame
        return cv2.circle(frame, center_position, radius, color, line_thickness)

    # Draw circle
    @staticmethod
    def draw_ellipse(frame, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness):
        """
            draw ellipse ~ ellipse(frame, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness)
                frame: draw the circle in this frame
                center_coordinates: (x, y) ~ position of the circle -> (20, 20)
                axes_length: (width, height) ~ size of the circle -> (200, 200)
                angle: ellipse rotation angle in degrees
                start_angle: starting angle of the elliptic arc in degrees
                end_angle: ending angle of the elliptic arc in degrees
                color: (blue, green, red) ~ color of the circle -> (128, 128, 128)
                thickness: line thickness of the circle ~ (-1 to fill the rect)
        """

        # Return the rectangle in the frame
        return cv2.ellipse(frame, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness)
