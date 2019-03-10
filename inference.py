import argparse
import pickle
import sys

from cust_face_utils.helpers import shape_to_np, calculateHOG, load_dlib_detector, face_detector_dlib
from cust_face_utils.facealigner import FaceAligner
import cv2
import os
import numpy as np


def process_data(args):
    detector, predictor = load_dlib_detector("models/shape_predictor_68_face_landmarks.dat")

    aligner = FaceAligner(predictor=predictor)
    _dir =  args.test_dir

    loaded_model = pickle.load(open(args.clasisfier_path, 'rb'))

    image_list = []
    img_names = []

    for fname in os.listdir(_dir):
        img = cv2.imread(os.path.join(_dir, fname))
        # преобразуем изображение в чернобелый формат
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rect = face_detector_dlib(detector, gray)
        if rect is None:
            continue

        aligned_face = aligner.align(img, gray, rect)
        image_list.append(aligned_face)
        img_names.append(fname)

    hist_features = np.squeeze(calculateHOG(image_list))

    y = loaded_model.predict(hist_features)

    if np.count_nonzero(y) == 0:
        print("No detections.")
    else:
        print("Glasses detected on following images:")

    for i,v in enumerate(y):
        if v: #glasses present
            print(os.path.join(_dir,img_names[i]))

    return


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--clasisfier_path',  help='Path to classifier', default='models/svm_classifier.sav', type=str)
    parser.add_argument('--test_dir', help='Path to classifier', default='', type=str)
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    process_data(args)

    return


if __name__ == "__main__":
    main()