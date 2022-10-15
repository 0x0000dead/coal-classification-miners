from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.utils.data
import torchvision


class Analyzer:
    def __init__(self,
                 mode_type='yolo',
                 box_conf_th=0.5,
                 nms_th=0.2,
                 amp=True,
                 model_path="C:/Users/Sergey/Documents/GitHub/coal-classification-miners/model/my_model.pth"):
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
                           path=r'C:\Users\Sergey\Documents\GitHub\coal-classification-miners\model\best.pt',
                                        )
                           #force_reload=True) # если не работает то раскоментить
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

    def get_data(self, img):
        results = self.model(img)
        return results.render(), results.pandas().xyxy[0].to_json(orient="records")
        # print(results.xyxy[0])  # print img1 predictions
        # len_dataloader = len(data_loader)
        # for imgs, annotations, paths in data_loader:
        #     i += 1
        #     imgs = list(img.to(device) for img in imgs)
        #     preds = model(imgs)
        #     print(f'Iteration: {i}/{len_dataloader}')
        #     for batch_item in preds:
        #         count_best_boxes = 1
        #         for score in batch_item['scores']:
        #             if score > threshold:
        #                 count_best_boxes += 1
        #
        #         boxes = batch_item['boxes'][:count_best_boxes]
        #         file_name = paths[counter % train_batch_size]
        #         image_id = int(re.findall(r'\d+', file_name)[0])
        #         coco_image = CocoImage(file_name=file_name, height=1080, width=1920, id=image_id)
        #         coco_image
        #         img = imgs[counter % train_batch_size].cpu().detach().numpy()
        #         img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
        #         images = cv2.imread(str(test_dir_path) + str("/") + str(file_name))
        #
        #         drawBBox(images, boxes, i)
        #
        #         continue
        #         for box in boxes:
        #             print(test_dir_path)
        #             print(file_name)
        #             width, height = Image.open(test_dir_path / file_name).size
        #             x_min = box[0].item()
        #             y_min = box[1].item()
        #             width = box[2].item() - x_min
        #             height = box[3].item() - y_min
        #             coco_image.add_annotation(
        #                 CocoAnnotation(
        #                     bbox=[x_min, y_min, width, height],
        #                     category_id=1,
        #                     category_name='stone1',
        #                     image_id=image_id
        #                 )
        #             )
        #         coco.add_image(coco_image)
        #         counter += 1

        # save_json(data=coco.json, save_path=submission_path)

import cv2
if __name__ == '__main__':
    analyzer = Analyzer(model_path=r'C:\Users\Sergey\Documents\GitHub\coal-classification-miners\model\best.pt',
                        mode_type='yolo')
    img = cv2.imread(r"C:\Users\Sergey\Documents\GitHub\coal-classification-miners\data\public\frame1254.jpg")
    print(analyzer.get_data(img))

# threshold = 0.5
# пустой конвеер - джсон пустой