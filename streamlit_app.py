import json

import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

import constants
from Analyzer import Analyzer
from ores import Ores
from visualize import draw_bboxes, get_size


def max_size(bboxes):
    res = 0.0
    for bbox in bboxes:
        if bbox['size'] > res:
            res = bbox['size']
    return res


def draw_max_size_hist(histogram_max_size, list_max_size, bboxes_list):
    if len(list_max_size) >= constants.MAX_HIST_LEN:
        del list_max_size[0]
    list_max_size.append(max_size(bboxes_list))

    fig, ax = plt.subplots()
    ax.plot(list_max_size)
    ax.set_title('Max ores size, mm')
    ax.set_xlabel("time in frames")
    histogram_max_size.pyplot(fig)


def bboxes_types_list(bboxes_list):
    res = [0] * len(Ores)
    for bbox in bboxes_list:
        res[bbox['type']] += bbox['type']
    return res


def draw_area_chart(place, list_bboxes_types, bboxes_list):
    if len(list_bboxes_types) >= constants.MAX_HIST_LEN:
        del list_bboxes_types[0]
    if len(list_bboxes_types) == 0:
        list_bboxes_types.append(bboxes_types_list(bboxes_list))
    else:
        list_bboxes_types.append(list(map(sum, zip(list_bboxes_types[-1], bboxes_types_list(bboxes_list)))))

    fig, ax = plt.subplots()
    y = np.array(list_bboxes_types)
    y = y.transpose()
    y = y / y.sum(axis=0).astype(float) * 100
    ax.stackplot(range(len(list_bboxes_types)), y, labels=[o.description for o in Ores])
    ax.set_title('100 % stacked area chart')
    ax.set_ylabel('Percent (%)')
    ax.set_xlabel("time in frames")
    place.pyplot(fig)


def bboxes_add_sizes(bboxes):
    for bbox in bboxes:
        bbox['size'] = get_size(bbox)
        bbox['type'] = 0
        for i in range(len(Ores)):
            ore = list(Ores)[i]
            if bbox['size'] >= ore.min_size and (ore.max_size < 0 or bbox['size'] < ore.max_size):
                bbox['type'] = i


def main():
    analyzer = Analyzer(model_path=constants.ROOT_DIR + r'/model/best.pt', mode_type='yolo')

    list_max_sizes = []
    list_bboxes_types = []

    st.title("Miners DataHack")
    st.subheader("Camera:")

    # # left menu
    selected_class = st.sidebar.radio("Select Class", list(Ores), 4, format_func=lambda o: o.description)
    flag_one_size = st.sidebar.checkbox("Only this size")
    flag_draw_boxes = st.sidebar.checkbox("Draw bboxes", True)

    image_location = st.empty()
    place_max_size = st.empty()
    place_stackplot = st.empty()

    # 0 for web camera
    cap = cv2.VideoCapture('resources/example.mp4')
    i = 0
    while cap.isOpened():
        i += 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        if not res:
            break

        bboxes_list = json.loads(analyzer.get_bboxes(frame))
        bboxes_add_sizes(bboxes_list)

        if flag_draw_boxes:
            frame = draw_bboxes(frame, bboxes_list, selected_class, flag_one_size)
        image_location.image(frame, channels="BGR")

        draw_max_size_hist(place_max_size, list_max_sizes, bboxes_list)
        draw_area_chart(place_stackplot, list_bboxes_types, bboxes_list)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
