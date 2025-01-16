import os
import logging
import datetime
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def train_evaluate_models(X_train, X_test, y_train, y_test):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder_name = f"experiment_{current_time}"
    base_output_path = 'model_ml'  # Base folder within the current directory
    full_output_path = os.path.join(base_output_path, output_folder_name)
    
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    
    jobs = os.cpu_count() // 2  
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=10, n_jobs=jobs),
        'LogisticRegression': LogisticRegression(n_jobs=jobs),
        'SGDClassifier (SVM)': SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3, n_jobs=jobs),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=jobs), 
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', n_jobs=jobs, n_estimators=100) 
    }
    pca_models = {'LogisticRegression', 'SGDClassifier (SVM)', 'KNN'}

    all_results = {}
    precision_recall_data = []
    sensitivity_specificity_data = []
    model_colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    plt.rcParams.update({'font.size': 18})

    # Define figures for plots
    pr_fig, pr_ax = plt.subplots(figsize=(8, 6))
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

    for (model_name, model), color in zip(models.items(), model_colors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if model_name in pca_models:
                logging.info(f'Applying PCA for {model_name}')
                pca = IncrementalPCA(n_components=149, batch_size=200)
                X_train_transformed = pca.fit_transform(X_train)
                X_test_transformed = pca.transform(X_test)
            else:
                X_train_transformed, X_test_transformed = X_train, X_test

            model.fit(X_train_transformed, y_train)
            predictions = model.predict(X_test_transformed)
            proba_predictions = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_transformed)

            metrics = calculate_metrics(y_test, predictions, proba_predictions, model_name, color, pr_ax, roc_ax)
            all_results[model_name] = metrics['results']  # Change here to assign by model name
            precision_recall_data.extend(metrics['pr_data'])
            sensitivity_specificity_data.extend(metrics['roc_data'])

    finalize_plots(pr_ax, roc_ax, pr_fig, roc_fig, full_output_path)
    results_df, precision_recall_df, sensitivity_specificity_df = save_to_csv(all_results, precision_recall_data, sensitivity_specificity_data, full_output_path)
    return results_df, precision_recall_df, sensitivity_specificity_df


def train_evaluate_models_p(X_train, X_test, y_train, y_test):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder_name = f"experiment_{current_time}"
    base_output_path = 'model_ml'  # Base folder within the current directory
    full_output_path = os.path.join(base_output_path, output_folder_name)
    
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    
    jobs = os.cpu_count() // 2  
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=jobs),
        'LogisticRegression': LogisticRegression(n_jobs=jobs),
        'SGDClassifier (SVM)': SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3, n_jobs=jobs),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=jobs), 
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', n_jobs=jobs, n_estimators=100) 
    }
    pca_models = {}

    all_results = {}
    precision_recall_data = []
    sensitivity_specificity_data = []
    model_colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    plt.rcParams.update({'font.size': 18})

    pr_fig, pr_ax = plt.subplots(figsize=(8, 6))
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

    for (model_name, model), color in zip(models.items(), model_colors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if model_name in pca_models:
                logging.info(f'Applying PCA for {model_name}')
                pca = IncrementalPCA(n_components=149, batch_size=200)
                X_train_transformed = pca.fit_transform(X_train)
                X_test_transformed = pca.transform(X_test)
            else:
                X_train_transformed, X_test_transformed = X_train, X_test

            model.fit(X_train_transformed, y_train)
            predictions = model.predict(X_test_transformed)
            proba_predictions = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_transformed)

            metrics = calculate_metrics(y_test, predictions, proba_predictions, model_name, color, pr_ax, roc_ax)
            all_results[model_name] = metrics['results']  # Change here to assign by model name
            precision_recall_data.extend(metrics['pr_data'])
            sensitivity_specificity_data.extend(metrics['roc_data'])

    finalize_plots(pr_ax, roc_ax, pr_fig, roc_fig, full_output_path)
    results_df, precision_recall_df, sensitivity_specificity_df = save_to_csv(all_results, precision_recall_data, sensitivity_specificity_data, full_output_path)
    return results_df, precision_recall_df, sensitivity_specificity_df

def calculate_metrics(y_test, predictions, proba_predictions, model_name, color, pr_ax, roc_ax):
    auc_score = roc_auc_score(y_test, proba_predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision_val = precision_score(y_test, predictions, zero_division=0)
    recall_val = recall_score(y_test, predictions)  
    f1 = f1_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    specificity_val = tn / (tn + fp) 

    results = {
        model_name: {
            'AUC': auc_score,
            'Accuracy': accuracy,
            'Precision': precision_val,
            'Recall (Sensitivity)': recall_val,
            'Specificity': specificity_val,
            'F1 Score': f1
        }
    }

    logging.info(f"{model_name}: F1 Score: {f1:.4f} , Accuracy: {accuracy:.4f}, Precision: {precision_val:.4f}, Recall (Sensitivity): {recall_val:.4f}, Specificity: {specificity_val:.4f}, AUC: {auc_score:.4f} ")

    precision, recall, _ = precision_recall_curve(y_test, proba_predictions)
    pr_auc = auc(recall, precision)
    pr_ax.plot(recall, precision, label=f'{model_name} (AUC-PR = {pr_auc:.2f})', color=color)

    fpr, tpr, _ = roc_curve(y_test, proba_predictions)
    roc_auc = auc(fpr, tpr)
    roc_ax.plot(fpr, tpr, label=f'{model_name} ROC (AUC = {roc_auc:.2f})', color=color)

    pr_data = [{'Model': model_name, 'Precision': p, 'Recall': r} for p, r in zip(precision, recall)]
    roc_data = [{'Model': model_name, 'Sensitivity (Recall)': tpr, '1-Specificity (FPR)': fpr} for tpr, fpr in zip(tpr, fpr)]

    return {'results': results, 'pr_data': pr_data, 'roc_data': roc_data}

def finalize_plots(pr_ax, roc_ax, pr_fig, roc_fig, output_folder):
    for ax, fig, title, filename in [(pr_ax, pr_fig, 'Precision-Recall Curves', 'precision_recall_curves.png'), (roc_ax, roc_fig, 'ROC Curves', 'roc_curves.png')]:
        ax.set_xlabel('Recall' if 'Precision' in title else 'False Positive Rate')
        ax.set_ylabel('Precision' if 'Precision' in title else 'True Positive Rate')
        ax.legend(loc='best')
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, filename))
        plt.close(fig)


def save_to_csv(all_results, precision_recall_data, sensitivity_specificity_data, output_folder):
    results_df = pd.DataFrame.from_dict(all_results, orient='index')  
    precision_recall_df = pd.DataFrame(precision_recall_data)
    sensitivity_specificity_df = pd.DataFrame(sensitivity_specificity_data)

    results_df.to_csv(os.path.join(output_folder, 'model_performance_metrics.csv'), index=True)
    precision_recall_df.to_csv(os.path.join(output_folder, 'precision_recall_curve_data.csv'), index=False)
    sensitivity_specificity_df.to_csv(os.path.join(output_folder, 'roc_curve_data.csv'), index=False)

    return results_df, precision_recall_df, sensitivity_specificity_df


