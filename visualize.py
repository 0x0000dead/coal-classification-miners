import cv2

from constants import CONVEYER_WIDTH, CONVEYER_BOTTOM_PX, CONVEYER_TOP_PX

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
BOX_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, ore, img_high, thickness=3):
    x_min, y_min, x_max, y_max, = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    # choose color
    size = get_size(bbox, img_high)
    if ore.max_size < 0:
        color = GREEN_COLOR
    elif size >= ore.max_size:
        color = RED_COLOR
    elif ore.min_size <= size < ore.max_size:
        color = BLUE_COLOR
    else:
        color = GREEN_COLOR

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    cv2.imshow("", img)
    return img


# diagonal of the bbox in pixels
def base_size(bbox):
    x_min, y_min, x_max, y_max, = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    return ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5


# size of the bbox in mm
def get_size(bbox, conveyer_len):
    pix_to_mm = CONVEYER_WIDTH / CONVEYER_BOTTOM_PX
    ratio = 1.0 + int(bbox['ymax']) * (CONVEYER_BOTTOM_PX / CONVEYER_TOP_PX - 1.0) / conveyer_len
    return base_size(bbox) * pix_to_mm * ratio


def draw_bboxes(image, bboxes, ore):
    img = image.copy()
    img_high = img.shape[0]
    for bbox in bboxes:
        img = visualize_bbox(img, bbox, ore, img_high)
    return img
