import os

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.utils.data
import torchvision
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import re
import json

from constants import ROOT_DIR


class Analyzer:
    def __init__(self,
                 mode_type='yolo',
                 box_conf_th=0.5,
                 nms_th=0.2,
                 amp=True,
                 model_path=ROOT_DIR + "/model/my_model.pth"):
        self.model_type = mode_type
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.model_type == 'rcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)

        elif self.model_type == 'yolo':
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=ROOT_DIR + r'/model/best.pt',
                                        )
            # force_reload=True)
            self.model.conf = box_conf_th
            self.model.iou = nms_th
            self.model.amp = amp

        self.model.eval()

    def get_data_rcnn(self, imgs):
        preds = self.model(imgs)
        return preds

    def get_bboxes(self, img):
        results = self.model(img)
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == '__main__':
    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='stone0'))
    coco.add_category(CocoCategory(id=1, name='stone1'))

    analyzer = Analyzer(model_path=ROOT_DIR + '/model/best.pt',
                        mode_type='yolo',
                        box_conf_th=0.9)
    glob_path = ROOT_DIR + '/data/public/'
    for path in (os.listdir(glob_path)):
        js = (analyzer.get_data(glob_path + path))
        js = json.loads(js)
        image_id = int(re.findall(r'\d+', path)[0])
        coco_image = CocoImage(file_name=path, height=1080, width=1920, id=image_id)
        for i in js:
            x_min = i['xmin']
            y_min = i['ymin']
            width = i['xmax'] - x_min
            height = i['ymax'] - y_min
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[x_min, y_min, width, height],
                    category_id=1,
                    category_name='stone1',
                    image_id=image_id
                )
            )
        coco.add_image(coco_image)
    save_json(data=coco.json,
              save_path=ROOT_DIR + "/model/submission.json")
