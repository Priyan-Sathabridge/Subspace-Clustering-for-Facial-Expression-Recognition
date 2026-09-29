"""
insightface_detector.py

Loads InsightFace once and returns
106 facial landmarks.
"""

import cv2

from insightface.app import FaceAnalysis


class InsightFaceDetector:

    def __init__(
            self,
            provider="CPUExecutionProvider"
    ):

        self.app = FaceAnalysis(
            providers=[provider]
        )

        self.app.prepare(ctx_id=0)

    ###################################################

    def detect_landmarks(self, image):

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        return faces[0].landmark_2d_106

    ###################################################

    def detect_from_file(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return None

        landmarks = self.detect_landmarks(image)

        return image, landmarks