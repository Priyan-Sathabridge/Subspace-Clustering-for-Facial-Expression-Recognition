"""
semantic_mapping.py

Automatic semantic landmark mapping for the
InsightFace 106-point model.
"""

import numpy as np

# -------------------------------------------------------
# Region Definitions
# -------------------------------------------------------

JAW = [
     0,24,23,22,21,20,19,18,
    32,31,30,29,28,27,26,25,
    17,
     1,9,10,11,12,13,14,15,
    16,2,3,4,5,6,7,8
]

LEFT_EYE = [33,34,35,36,37,38,39,40,41,42]

RIGHT_EYE = [87,88,89,90,91,92,93,94,95,96]

LEFT_BROW = [43,44,45,46,47,48,49,50,51]

RIGHT_BROW = [97,98,99,100,101,102,103,104,105]

NOSE = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86]

MOUTH = [52,53,54,55,56,57,58,59,60,61,
         62,63,64,65,66,67,68,69,70,71]


# -------------------------------------------------------
# Generic Helper
# -------------------------------------------------------

def extreme_points(points):

    return {

        "left"   : np.argmin(points[:,0]),
        "right"  : np.argmax(points[:,0]),

        "top"    : np.argmin(points[:,1]),
        "bottom" : np.argmax(points[:,1])

    }


# -------------------------------------------------------
# Mouth
# -------------------------------------------------------

def mouth_map(landmarks):

    mouth = landmarks[MOUTH]

    idx = extreme_points(mouth)

    semantic = {

        "left_corner":
            mouth[idx["left"]],

        "right_corner":
            mouth[idx["right"]],

        "upper_lip":
            mouth[idx["top"]],

        "lower_lip":
            mouth[idx["bottom"]],

        "centre":
            mouth.mean(axis=0),

        "all_points":
            mouth

    }

    return semantic


# -------------------------------------------------------
# Left Eye
# -------------------------------------------------------

def left_eye_map(landmarks):

    eye = landmarks[LEFT_EYE]

    idx = extreme_points(eye)

    return {

        "left_corner":
            eye[idx["left"]],

        "right_corner":
            eye[idx["right"]],

        "upper":
            eye[idx["top"]],

        "lower":
            eye[idx["bottom"]],

        "centre":
            eye.mean(axis=0),

        "all_points":
            eye

    }


# -------------------------------------------------------
# Right Eye
# -------------------------------------------------------

def right_eye_map(landmarks):

    eye = landmarks[RIGHT_EYE]

    idx = extreme_points(eye)

    return {

        "left_corner":
            eye[idx["left"]],

        "right_corner":
            eye[idx["right"]],

        "upper":
            eye[idx["top"]],

        "lower":
            eye[idx["bottom"]],

        "centre":
            eye.mean(axis=0),

        "all_points":
            eye

    }


# -------------------------------------------------------
# Nose
# -------------------------------------------------------

def nose_map(landmarks):

    nose = landmarks[NOSE]

    idx = extreme_points(nose)

    return {

        "left":
            nose[idx["left"]],

        "right":
            nose[idx["right"]],

        "bridge":
            nose[idx["top"]],

        "tip":
            nose[idx["bottom"]],

        "centre":
            nose.mean(axis=0),

        "all_points":
            nose

    }


# -------------------------------------------------------
# Eyebrows
# -------------------------------------------------------

def eyebrow_map(indices, landmarks):

    brow = landmarks[indices]

    idx = extreme_points(brow)

    return {

        "inner":
            brow[idx["left"]],

        "outer":
            brow[idx["right"]],

        "highest":
            brow[idx["top"]],

        "lowest":
            brow[idx["bottom"]],

        "centre":
            brow.mean(axis=0),

        "all_points":
            brow

    }


# -------------------------------------------------------
# Jaw
# -------------------------------------------------------

def jaw_map(landmarks):

    jaw = landmarks[JAW]

    idx = extreme_points(jaw)

    return {

        "left":
            jaw[idx["left"]],

        "right":
            jaw[idx["right"]],

        "chin":
            jaw[idx["bottom"]],

        "top":
            jaw[idx["top"]],

        "centre":
            jaw.mean(axis=0),

        "all_points":
            jaw

    }


# -------------------------------------------------------
# Master Function
# -------------------------------------------------------

def semantic_mapping(landmarks):

    return {

        "mouth":
            mouth_map(landmarks),

        "left_eye":
            left_eye_map(landmarks),

        "right_eye":
            right_eye_map(landmarks),

        "left_brow":
            eyebrow_map(
                LEFT_BROW,
                landmarks
            ),

        "right_brow":
            eyebrow_map(
                RIGHT_BROW,
                landmarks
            ),

        "nose":
            nose_map(landmarks),

        "jaw":
            jaw_map(landmarks)

    }

"""
semantic_mapping.py

Automatic semantic landmark mapping for the
InsightFace 106-point model.
"""

import numpy as np

# -------------------------------------------------------
# Region Definitions
# -------------------------------------------------------

JAW = [
     0,24,23,22,21,20,19,18,
    32,31,30,29,28,27,26,25,
    17,
     1,9,10,11,12,13,14,15,
    16,2,3,4,5,6,7,8
]

LEFT_EYE = [33,34,35,36,37,38,39,40,41,42]

RIGHT_EYE = [87,88,89,90,91,92,93,94,95,96]

LEFT_BROW = [43,44,45,46,47,48,49,50,51]

RIGHT_BROW = [97,98,99,100,101,102,103,104,105]

NOSE = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86]

MOUTH = [52,53,54,55,56,57,58,59,60,61,
         62,63,64,65,66,67,68,69,70,71]


# -------------------------------------------------------
# Generic Helper
# -------------------------------------------------------

def extreme_points(points):

    return {

        "left"   : np.argmin(points[:,0]),
        "right"  : np.argmax(points[:,0]),

        "top"    : np.argmin(points[:,1]),
        "bottom" : np.argmax(points[:,1])

    }


# -------------------------------------------------------
# Mouth
# -------------------------------------------------------

def mouth_map(landmarks):

    mouth = landmarks[MOUTH]

    idx = extreme_points(mouth)

    semantic = {

        "left_corner":
            mouth[idx["left"]],

        "right_corner":
            mouth[idx["right"]],

        "upper_lip":
            mouth[idx["top"]],

        "lower_lip":
            mouth[idx["bottom"]],

        "centre":
            mouth.mean(axis=0),

        "all_points":
            mouth

    }

    return semantic


# -------------------------------------------------------
# Left Eye
# -------------------------------------------------------

def left_eye_map(landmarks):

    eye = landmarks[LEFT_EYE]

    idx = extreme_points(eye)

    return {

        "left_corner":
            eye[idx["left"]],

        "right_corner":
            eye[idx["right"]],

        "upper":
            eye[idx["top"]],

        "lower":
            eye[idx["bottom"]],

        "centre":
            eye.mean(axis=0),

        "all_points":
            eye

    }


# -------------------------------------------------------
# Right Eye
# -------------------------------------------------------

def right_eye_map(landmarks):

    eye = landmarks[RIGHT_EYE]

    idx = extreme_points(eye)

    return {

        "left_corner":
            eye[idx["left"]],

        "right_corner":
            eye[idx["right"]],

        "upper":
            eye[idx["top"]],

        "lower":
            eye[idx["bottom"]],

        "centre":
            eye.mean(axis=0),

        "all_points":
            eye

    }


# -------------------------------------------------------
# Nose
# -------------------------------------------------------

def nose_map(landmarks):

    nose = landmarks[NOSE]

    idx = extreme_points(nose)

    return {

        "left":
            nose[idx["left"]],

        "right":
            nose[idx["right"]],

        "bridge":
            nose[idx["top"]],

        "tip":
            nose[idx["bottom"]],

        "centre":
            nose.mean(axis=0),

        "all_points":
            nose

    }


# -------------------------------------------------------
# Eyebrows
# -------------------------------------------------------

def eyebrow_map(indices, landmarks):

    brow = landmarks[indices]

    idx = extreme_points(brow)

    return {

        "inner":
            brow[idx["left"]],

        "outer":
            brow[idx["right"]],

        "highest":
            brow[idx["top"]],

        "lowest":
            brow[idx["bottom"]],

        "centre":
            brow.mean(axis=0),

        "all_points":
            brow

    }


# -------------------------------------------------------
# Jaw
# -------------------------------------------------------

def jaw_map(landmarks):

    jaw = landmarks[JAW]

    idx = extreme_points(jaw)

    return {

        "left":
            jaw[idx["left"]],

        "right":
            jaw[idx["right"]],

        "chin":
            jaw[idx["bottom"]],

        "top":
            jaw[idx["top"]],

        "centre":
            jaw.mean(axis=0),

        "all_points":
            jaw

    }


# -------------------------------------------------------
# Master Function
# -------------------------------------------------------

def semantic_mapping(landmarks):

    return {

        "mouth":
            mouth_map(landmarks),

        "left_eye":
            left_eye_map(landmarks),

        "right_eye":
            right_eye_map(landmarks),

        "left_brow":
            eyebrow_map(
                LEFT_BROW,
                landmarks
            ),

        "right_brow":
            eyebrow_map(
                RIGHT_BROW,
                landmarks
            ),

        "nose":
            nose_map(landmarks),

        "jaw":
            jaw_map(landmarks)

    }