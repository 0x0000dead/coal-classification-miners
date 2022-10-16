import json

import cv2
import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt

import constants
from Analyzer import Analyzer
from ores import Ores
from visualize import draw_bboxes, get_size

MAX_HIST_LEN = 5000


def max_size(bboxes):
    res = 0.0
    for bbox in bboxes:
        if bbox['size'] > res:
            res = bbox['size']
    return res


def draw_max_size_hist(histogram_max_size, list_max_size, bboxes_list):
    if len(list_max_size) >= MAX_HIST_LEN:
        del list_max_size[0]
    list_max_size.append(max_size(bboxes_list))

    fig, ax = plt.subplots()
    ax.plot(list_max_size)
    ax.set_title('Max ores size, mm')
    ax.set_xlabel("time in frames")
    histogram_max_size.pyplot(fig)



def bboxes_add_sizes(bboxes):
    for bbox in bboxes:
        bbox['size'] = get_size(bbox)


def main():
    analyzer = Analyzer(model_path=constants.ROOT_DIR + r'/model/best.pt', mode_type='yolo')

    list_max_sizes = []
    df_percents = pd.DataFrame()

    st.title("Miners DataHack")
    st.subheader("Camera:")

    # # left menu
    selected_class = st.sidebar.radio("Select Class", list(Ores), 4, format_func=lambda o: o.description)
    flag_one_size = st.sidebar.checkbox("Only this size")
    flag_draw_boxes = st.sidebar.checkbox("Draw bboxes", True)

    image_location = st.empty()
    histogram_max_size = st.empty()

    # 0 for web camera
    cap = cv2.VideoCapture('resources/example.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        bboxes_list = json.loads(analyzer.get_bboxes(frame))
        bboxes_add_sizes(bboxes_list)

        if flag_draw_boxes:
            frame = draw_bboxes(frame, bboxes_list, selected_class, flag_one_size)
        image_location.image(frame, channels="BGR")

        draw_max_size_hist(histogram_max_size, list_max_sizes, bboxes_list)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
