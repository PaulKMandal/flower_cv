import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes=91):  # COCO has 80 classes, but the indices go up to 90
    # Load a pre-trained model for classification and return
    # only the features
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the classifier with a new one, that has
    # num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model