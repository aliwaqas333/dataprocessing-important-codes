from sklearn.metrics import jaccard_score
import numpy as np
import os
import cv2

def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
    labels = [0, 255]
    jaccards = []
    for label in labels:
        jaccard = jaccard_score(y_pred.flatten(),y_true.flatten(), pos_label=label)
        jaccards.append(jaccard)
    # print(f'avg={np.mean(jaccards)}')
    return np.mean(jaccards)

if __name__ == "__main__":
    preds = os.listdir("./pred")
    gts = os.listdir("./true")
    miou = 0
    print(f"len of pred = {len(preds)}, len of gts = {len(gts)}")
    
    for idx, image in enumerate(preds):
        pred = cv2.imread(f"./pred/{image}")
        gt = cv2.imread(f"./true/{gts[idx]}")
        iou = compute_iou(pred, gt)
        miou += iou
        print(f"{image}: {iou}")
    print("================")
    print(f"MIOU: {miou/len(preds)}")
    print("================")


