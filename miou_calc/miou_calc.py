from sklearn.metrics import jaccard_score, f1_score
import numpy as np
import os
import cv2

def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
    labels = np.unique(y_pred)
    jaccards = []
    for label in labels:
        jaccard = jaccard_score(y_pred.flatten(),y_true.flatten(), pos_label=label)
        jaccards.append(jaccard)
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), pos_label= label)
    return np.mean(jaccards), np.mean(f1)

if __name__ == "__main__":
    preds = os.listdir("./pred")
    gts = os.listdir("./true")
    miou = 0
    mf1 = 0
    print(f"len of pred = {len(preds)}, len of gts = {len(gts)}")
    
    for idx, image in enumerate(preds):
        pred = cv2.imread(f"./pred/{image}", 0)
        gt = cv2.imread(f"./true/{gts[idx]}", 0)
        gt[gt>0] = 255
        pred[pred>0] = 255
        iou, f1 = compute_iou(pred, gt)
        # print(np.unique(gt))
        miou += iou
        mf1 += f1
        print(f"{image}: {iou} f1: {f1}")
    print("================")
    print(f"MIOU: {miou/len(preds)}, F1_Score: {mf1/len(preds)}")
    print("================")


