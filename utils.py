import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, data_loader, device=torch.device('cuda')):
    model.train()
    model.to(device)
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # Perform backpropagation, optimizer step, etc.

def test(model, data_loader, device=torch.device('cuda')):
    model.eval()
    model.to(device)
    # Initialize COCO ground truth and predictions
    coco_gt = COCO()
    coco_pred = coco_gt.loadRes([])
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        # Convert outputs to COCO format and accumulate in coco_pred
        # ...
    # Evaluate the predictions
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0], coco_eval.stats[1]  # Return loss and accuracy