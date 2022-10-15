import json

import cv2

import constants
from ores import Ores
from Analyzer import Analyzer
from visualize import draw_bboxes

import streamlit as st


def main():
    analyzer = Analyzer(model_path=constants.ROOT_DIR + r'/model/best.pt', mode_type='yolo')
    cap = cv2.VideoCapture('resources/example.mp4')

    st.title("Miners DataHack")
    st.header("Camera:")

    radio_place = st.sidebar.empty()
    selected_class = radio_place.radio("Select Class", list(Ores), 4,
                                       format_func=lambda o: o.description)

    imageLocation = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()

        bboxes_list = json.loads(analyzer.get_bboxes(frame))
        img = draw_bboxes(frame, bboxes_list, selected_class)

        imageLocation.image(img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
