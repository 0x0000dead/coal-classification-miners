from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.utils.data
import torchvision

import constants


class Analyzer:
    def __init__(self,
                 mode_type='yolo',
                 box_conf_th=0.5,
                 nms_th=0.2,
                 amp=True,
                 model_path=constants.ROOT_DIR + "/model/my_model.pth"):
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
                                        path=model_path,
                                        force_reload=True)
            self.model.conf = box_conf_th  # NMS confidence threshold
            self.model.iou = nms_th  # NMS IoU threshold
            self.model.amp = amp  # Automatic Mixed Precision (AMP) inference

        self.model.eval()

    def get_json_yolo(self, model):
        pass
        # return model.predict(self.data)

    def get_json_rcnn(self, model):
        pass
        # return model.predict(self.data)

    def get_bboxes(self, img):
        results = self.model(img)
        return results.pandas().xyxy[0].to_json(orient="records")
