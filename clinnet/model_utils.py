import os
import logging 
import random
import numpy as np 
import itertools
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score)
from matplotlib import pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
#------------------------------------------------------------------------------------------------------ 
def get_unique_filename(file_path):
    filename_with_ext = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)
    filename_without_ext, extension = os.path.splitext(filename_with_ext)

    unique_file_path = file_path
    counter = 0
    while True:
        if os.path.exists(unique_file_path):
            counter += 1 
            unique_file_path = os.path.join(dirname,f'{filename_without_ext}_{counter}{extension}')
        else:
            return unique_file_path
#------------------------------------------------------------------------------------------------------ 
def plot_history(history_df, saving_dir, history_path=None, multiple_output=False):
    if history_path: 
        history_df = pd.read_csv(history_path)
    matrices = ['loss', 'accuracy','f1', 'recall', 'precision', 'auc']
   # history_df = history_df[[col for col in history_df.columns if not col.startswith('GS')]]

    if not multiple_output:
        for m in matrices:
            plot_filename = get_unique_filename(os.path.join(saving_dir, f'{m}.png'))
            his_matrices = [p for p in history_df.columns if p.__contains__(m)]
            if len(his_matrices) == 0:
                continue
            for h in his_matrices:
                plt.plot(history_df[h], label=h)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(m.title())
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            logging.info(f"{m} plot saved: {plot_filename}")
    else:
        if 'lr' in history_df.columns:
            history_df.drop('lr', axis=1, inplace=True)

        def get_row_col_idx(i, ncols):
            col_idx = i%ncols
            row_idx = int(i/ncols)
            return row_idx, col_idx

        train_metric = [i for i in history_df.columns if not i.startswith('val')]
        valid_metric = [i for i in history_df.columns if i.startswith('val')]

        metrics = {}
        for m in ['loss', 'accuracy','f1', 'recall','precision', 'auc']: 
            l = []
            for i, j in zip(train_metric, valid_metric):
                if i.__contains__(m) and j.__contains__(m):
                    l.append((i,j))
            metrics[m] = l

        for name, train_val in metrics.items():
            plot_filename = get_unique_filename(os.path.join(saving_dir, f'{name}.png'))
            ncols = 3
            nrows = int(np.ceil(len(train_val)/ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(nrows*6, ncols*6))
            for idx, (t, v) in enumerate(train_val):
                row_idx, col_idx = get_row_col_idx(idx, ncols)
                axs[row_idx, col_idx].plot(history_df[t], label=t)
                axs[row_idx, col_idx].plot(history_df[v], label=v)
                axs[row_idx, col_idx].set_title(t)
                axs[row_idx, col_idx].set_xlabel('Epochs')
                axs[row_idx, col_idx].set_ylabel(name)
                axs[row_idx, col_idx].legend()
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()
            logging.info(f"{name} plot saved: {plot_filename}")
#------------------------------------------------------------------------------------------------------ 
def plot_confusion_matrix(y_true, y_pred, save_dir,
                          classes=['control','case'],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plot_filename = get_unique_filename(os.path.join(save_dir,'confusion_matrix.png'))
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)+'%',
                 horizontalalignment="center",fontsize=20,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix plot saved: {plot_filename}")
#------------------------------------------------------------------------------------------------------
def plot_ROC(y_true, y_prob, save_dir):
    plot_filename = get_unique_filename(os.path.join(save_dir,'ROC_curve.png'))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"ROC curve plot saved: {plot_filename}")
#------------------------------------------------------------------------------------------------------
def get_metrics(y_true, y_prob, save_dir=None, th=0.5):
    y_pred = y_prob > th
    Accuracy = accuracy_score(y_true, y_pred)
    Precision = precision_score(y_true, y_pred)
    Sensitivity_recall = recall_score(y_true, y_pred)
    Specificity = recall_score(y_true, y_pred, pos_label=0)
    F1_score = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    AUC = auc(fpr, tpr)
    metrics_dict = {'Threshold':th,
                    'Accuracy':Accuracy,
                    'Precision':Precision,
                    'Sensitivity_recall':Sensitivity_recall,
                    'Specificity':Specificity,
                    'F1_score':F1_score,
                    'AUC':AUC}
    logging.info(f"Accuracy, Precision, Sensitivity_recall, Specificity, F1_score, and AUC calculated.")
    if save_dir:
        metrics_path = get_unique_filename(os.path.join(save_dir, 'metrics.txt'))
        with open(metrics_path, 'w') as file:
            for key, value in metrics_dict.items():
                file.write(f'{key}: {value}\n')
        logging.info(f"Metrices saved: {metrics_path}")
        
    specificity = 1 - fpr
    sensitivity_specificity_data = pd.DataFrame({'Sensitivity': tpr, '1-Specificity':specificity})
    sensitivity_specificity_data.to_csv(os.path.join(save_dir, 'sensitivity_specificity_Sydney.csv'), index=False)
    return metrics_dict
#------------------------------------------------------------------------------------------------------
def save_precision_recall(y_true, y_prob, save_dir):
      thresholds = np.arange(0.0, 1.1, 0.01)
      precisions = []
      recalls = []
      for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
    
      df = pd.DataFrame({ 'Precision': precisions, 'Recall': recalls})
      file_path = os.path.join(save_dir, 'precision_recall_all_thresholds.csv')
      df.to_csv(file_path, index=False)
#------------------------------------------------------------------------------------------------------
def save_sensitivity_specificity(y_true, y_prob, save_dir):
      fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
      sensitivity = tpr
      specificity = 1 - fpr
    
      df = pd.DataFrame({ 'Sensitivity': sensitivity, '1-Specificity': specificity})
      file_path = os.path.join(save_dir, 'sensitivity_specificity_all_thresholds.csv')
      df.to_csv(file_path, index=False)
   
      logging.info(f"Sensitivity and 1-Specificity saved to: {file_path}")
#------------------------------------------------------------------------------------------------------ 
def get_th(y_true, y_prob, return_result=False):
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    for th in thresholds:
        y_pred = y_prob > th
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        score = {}
        score['accuracy'] = accuracy
        score['precision'] = precision
        score['f1'] = f1
        score['recall'] = recall
        score['th'] = th
        scores.append(score)
    ret = pd.DataFrame(scores)
    best = ret[ret.accuracy == max(ret.accuracy)]
    th = best.th.values[0]
    logging.info(f'Best threshold to maximize the accuracy: {th:.4f}\n')
    # logging.info(f'Metrics for best threshold:\n{best}')
    # if return_result:
    #     return ret
    return th