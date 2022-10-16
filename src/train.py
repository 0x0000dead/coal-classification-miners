import git
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import os


def train_yolo():
    git.Git("./").clone("git@github.com:ultralytics/yolov5.git")
    os.system(
        "python train.py --img 640 --batch 4 --epochs 20 --optimizer Adam --data polus_yolov5.yaml --weights yolov5s6.pt  --entity coal-composition-control")


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))

        num_objs = len(coco_annotation)
        boxes = []
        try:
            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
        except:
            pass
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms:
            augmented = self.transforms(image=np.asarray(img), )
            img = augmented['image']
        return img, my_annotation, path

    def __len__(self):
        return len(self.ids)


def train_rcnn():
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import albumentations.pytorch

    get_transform = albumentations.Compose([
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(),
            albumentations.HueSaturationValue()
        ], p=0.7),

        albumentations.OneOf([
            albumentations.MotionBlur(p=1),
            albumentations.GaussNoise(p=1)
        ], p=0.7),
        albumentations.OneOf(
            [
                albumentations.HorizontalFlip(p=1.0),
                albumentations.VerticalFlip(p=1.0),
                albumentations.RandomRotate90(p=1.0),

            ], p=0.8),

        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()

    ])

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_batch_size = 4

    train_data_dir = '../data/train'
    train_coco = '../data/annot_local/train_annotation.json'

    my_dataset = myOwnDataset(root=train_data_dir,
                              annotation=train_coco,
                              transforms=get_transform,

                              )
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    num_classes = 2

    model = get_model_instance_segmentation(num_classes)
    model_path = r'C:\Users\Sergey\Documents\GitHub\coal-classification-miners\model\my_model.pth'
    model.to(device)

    num_epochs = 2
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)
    print(len_dataloader)
    last_losses = 100
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations, path in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
        if losses.item() < last_losses:
            last_losses = losses.item()
            torch.save(model.state_dict(),
                       model_path)
