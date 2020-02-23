import cv2
import dlib
import numpy as np
import keyboard

# Define what landmarks you want:
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))


def draw_shape_points(np_shape, w, h):
    for x, y in np_shape:
        rel_x = w / x
        rel_y = h / y
        yield rel_x, rel_y


def shape_to_np(dlib_shape, dtype="int"):
    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates


def landmark_points(show_face=False):
    root_dir = "C:\\Users\\diego\\Desktop\\Scikit, Keras y TensorFlow\\OpenCV\\projects\\blender face tracker\\"
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(root_dir + p)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces:
        rects = detector(gray, 0)

        if len(rects) == 1:
            w = rects[0].right() - rects[0].left()
            h = rects[0].bottom() - rects[0].top()

            if show_face:
                pad = 20
                face = frame[rects[0].top()-pad: rects[0].bottom()+pad, rects[0].left()-pad: rects[0].right()+pad, :]
                cv2.imshow("Face", face)

            shape = predictor(gray, rects[0])
            shape = shape_to_np(shape)
            points = list(iter(draw_shape_points(shape, w, h)))
            yield points

        if keyboard.is_pressed('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
