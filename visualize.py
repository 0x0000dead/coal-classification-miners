import cv2

from constants import CONVEYER_WIDTH, CONVEYER_BOTTOM_PX, CONVEYER_TOP_PX, IMG_HIGH

LOW_SIZE_COLOR = (0, 255, 0)  # green
BIG_SIZE_COLOR = (0, 0, 255)  # red
NORMAL_SIZE_COLOR = (255, 255, 0)  # light blue
BOX_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, ore, one_size=False, thickness=3):
    x_min, y_min, x_max, y_max, = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    # choose color
    size = bbox['size']
    if ore.max_size < 0:
        color = LOW_SIZE_COLOR
    elif size >= ore.max_size:
        color = BIG_SIZE_COLOR
    elif ore.min_size <= size < ore.max_size:
        color = NORMAL_SIZE_COLOR
    else:
        color = LOW_SIZE_COLOR

    if color == LOW_SIZE_COLOR and one_size:
        return img

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


# diagonal of the bbox in pixels
def base_size(bbox):
    x_min, y_min, x_max, y_max, = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    return ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5


# size of the bbox in mm
def get_size(bbox, conveyer_len=IMG_HIGH):
    pix_to_mm = CONVEYER_WIDTH / CONVEYER_BOTTOM_PX
    ratio = 1.0 + int(bbox['ymax']) * (CONVEYER_BOTTOM_PX / CONVEYER_TOP_PX - 1.0) / conveyer_len
    return base_size(bbox) * pix_to_mm * ratio


def draw_bboxes(image, bboxes, ore, one_size=False):
    img = image.copy()
    for bbox in bboxes:
        img = visualize_bbox(img, bbox, ore, one_size=one_size)
    return img
