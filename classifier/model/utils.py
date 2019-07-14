import sys
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def report(y_true, y_pred, actual_true):
    """
    Params:
        y_true: numpy array
        y_pred: numpy array
    Return evaluatation result
    """
    total_pred = 0
    true_positive = 0

    for i in range(y_pred.shape[0]):
        pred_label = y_pred[i]
        if pred_label != 0:
            total_pred += 1
            if pred_label == y_true[i]:
                true_positive += 1

    precision = float(true_positive) / total_pred
    recall = float(true_positive) / actual_true
    f1 = 2 * precision * recall / (precision + recall)
    print("total_true", true_positive, "\ntotal_predict", total_pred,
            "\ntotal_entity", actual_true)
    print("precision:", precision, "\nrecall:", recall, "\nf1:", f1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    return f1

def test_report():
    a = np.array([0, 1, 2, 2, 3, 4, 2, 3, 3], dtype=np.int32)
    b = np.array([0, 1, 1, 2, 3, 4, 2, 2, 3], dtype=np.int32)
    report(a, b, 10)

if __name__ == '__main__':
    sys.exit(test_report())
