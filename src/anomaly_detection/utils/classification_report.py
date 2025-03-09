import pandas as pd
import numpy as np

def getClassificationReport(_confusion_matrix=None, traffic_classes=None, byte_count=False):
    total_flows_bytes = np.sum(_confusion_matrix, axis=1)
    recall = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=1)
    precision = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=0)
    _weighted_precision = [[a * b for a, b in zip(*l)] for l in zip(np.array(_confusion_matrix).T.tolist(),
                                                                    [[x / y for y in total_flows_bytes] for x in
                                                                    total_flows_bytes])]
    weighted_precision = np.diag(np.array(_confusion_matrix).T.tolist()) / [sum(l) for l in _weighted_precision]
    f1_Score = [2 * (a * b) / (a + b) for a, b in zip(weighted_precision, recall)]
    df_conf_matrix = pd.DataFrame(_confusion_matrix, columns=traffic_classes, index=traffic_classes)

    if byte_count:
        df_conf_matrix['total_bytes'] = total_flows_bytes
    else:
        df_conf_matrix['total_flows'] = total_flows_bytes
    df_conf_matrix['recall'] = np.round(recall * 100, 2)
    df_conf_matrix['precision'] = np.round(precision * 100, 2)

    df_conf_matrix['weighted_precision'] = np.round(weighted_precision * 100, 2)
    df_conf_matrix["f1_score"] = np.round(np.array(f1_Score) * 100, 2)
    return df_conf_matrix