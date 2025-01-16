import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from kneed import KneeLocator
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score)

from .model_utils import get_unique_filename
from model import converge_layers_prob



def predict_probabilities(model, x_valid, x_test, y_valid, converge_method):
    y_prob_valid = np.array(model.predict2(x_valid)).squeeze()
    y_prob_test = np.array(model.predict2(x_test)).squeeze()

    if y_prob_valid.ndim > 1:
        y_prob_valid = np.transpose(y_prob_valid)
    if y_prob_test.ndim > 1:
        y_prob_test = np.transpose(y_prob_test)

    y_prob_test, y_prob_valid = converge_layers_prob(y_prob_test, y_prob_valid, y_valid, method=converge_method)
    return y_prob_test.ravel(), y_prob_valid.ravel()  
 

def get_unique_filename(filepath):
    base, ext = os.path.splitext(filepath)
    counter = 1
    unique_filepath = filepath
    while os.path.exists(unique_filepath):
        unique_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return unique_filepath

def converge_layers_prob(layers_prob_test, layers_prob_valid, y_valid=None, method='first-layer'):
    if method == 'average':
        y_prob_test = np.average(layers_prob_test, axis=0).squeeze()
        y_prob_valid = np.average(layers_prob_valid, axis=0).squeeze()
    else:
        y_prob_test = layers_prob_test[0].squeeze()
        y_prob_valid = layers_prob_valid[0].squeeze()
    return y_prob_test, y_prob_valid


class UncertaintyModel:
    def __init__(self, y_test, y_prob_test, prediction_threshold, cv_dir):
        self.y_test = y_test
        self.y_prob_test = y_prob_test
        self.labels = [0, 1]
        self.prediction_threshold = prediction_threshold
        self.results_df = None
        self.confidence_scores = None
        self.cv_dir = cv_dir
        self.th = None

    def evaluate_un(self, step=0.01, min_confidence=0):
     if self.y_prob_test.ndim == 1:
        self.confidence_scores = 2 * abs(self.y_prob_test - 0.5)
        positive_class_probs = self.y_prob_test
     else:
        self.confidence_scores = 2 * self.y_prob_test[:, 1] - 1
        positive_class_probs = self.y_prob_test[:, 1]

     plt.figure(figsize=(8, 6))
     plt.hist(self.confidence_scores, bins=50, alpha=0.75)
     plt.title('Distribution of Confidence Scores', fontsize=16)
     plt.xlabel('Confidence Score', fontsize=14)
     plt.ylabel('Frequency', fontsize=14)
     plt.grid(False)
     plt.tight_layout()
     plt.show()

     self.th = self.prediction_threshold
     logging.info(f'Using threshold: {self.th}')

     results_list = []
     for confidence_threshold in np.arange(min_confidence, 1 + step, step):
        if confidence_threshold == 0:
           high_confidence_indices = np.ones(len(self.confidence_scores), dtype=bool)
        else:
          high_confidence_indices = self.confidence_scores >= confidence_threshold
       # print(f"Confidence Threshold: {confidence_threshold}, High Confidence Indices: {np.sum(high_confidence_indices)}")

        if not any(high_confidence_indices):
            continue

        y_pred_filtered = (positive_class_probs[high_confidence_indices] > self.th).astype(int)
        y_true_filtered = self.y_test[high_confidence_indices]

        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
        precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=self.labels)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        youden_index = recall + specificity - 1
        coverage = len(y_true_filtered) / len(self.y_test)

        results_list.append({
            'Confidence Threshold': confidence_threshold,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall (Sensitivity)': recall,
            'Precision': precision,
            'Specificity': specificity,
            'Youden\'s Index': youden_index,
            'Coverage': coverage
        })

     self.results_df = pd.DataFrame(results_list)
     results_path = os.path.join(self.cv_dir, 'model_evaluation_results_df.csv')
     self.results_df.to_csv(results_path, index=False)
     return self.results_df

    
    def find_dynamic_optimal_point(self, min_coverage=0.7):
     if self.results_df is None:
        print("No evaluation data available. Please run evaluate_un() first.")
        return None
     logging.info("Coverage value distribution:\n%s", self.results_df['Coverage'].describe())

     filtered_df = self.results_df[self.results_df['Coverage'] >= min_coverage].copy()
    
     if filtered_df.empty:
        print("No data points meet the minimum coverage requirement. Try lowering the min_coverage value.")
        return None

     filtered_df.sort_values(by='Youden\'s Index', inplace=True, ascending=True)

     x = filtered_df.index
     y = filtered_df['Youden\'s Index'].values

     knee_locator = KneeLocator(x, y, curve='convex', direction='increasing')
     knee_index = knee_locator.knee

     if knee_index is not None:
      optimal_point = filtered_df.loc[knee_index]
      optimal_point = filtered_df.loc[knee_index]
      youdens_index = optimal_point["Youden's Index"]
      coverage = optimal_point["Coverage"]
      print(f"Optimal Youden's Index at Coverage >= {min_coverage:.2f}")  # Format min_coverage to 2 decimal places
      print(f"Youden's Index: {youdens_index}")
      print(f"Coverage: {coverage}")  
      optimal_point_df = optimal_point.to_frame().T
      optimal_point_path = os.path.join(self.cv_dir, 'optimal_point.csv')
      optimal_point_df.to_csv(optimal_point_path, index=False)
      logging.info(f"Optimal point saved to '{optimal_point_path}'")
      return optimal_point.to_dict()
     else:
        print("No knee point found. Consider adjusting the minimum coverage or evaluating the data distribution.")
        return None


    def plot_evaluation_metrics_with_optimal(self, save_dir='.'):
      optimal_data = self.find_dynamic_optimal_point()
      if optimal_data is None:
        print("No optimal point found. Unable to plot.")
        return
      optimal_coverage = optimal_data['Coverage']

      font = FontProperties()
      font.set_family('Times New Roman')
      font.set_size(18)
      plt.rcParams["font.family"] = "Times New Roman"
      plt.rcParams["font.size"] = 18

      fig, ax1 = plt.subplots(figsize=(9, 6))
      # Plot Youden's Index on the same graph
      ax1.set_xlabel('Coverage', fontsize=20)
      ax1.set_ylabel("Youden's Index", fontsize=20, color='black')
      ax1.plot(self.results_df['Coverage'], self.results_df["Youden's Index"], label="Youden's Index",
             marker='o', linestyle='dashed', color='darkorange', linewidth=2.5)
      ax1.tick_params(axis='x', labelsize=18)  # Increased font size for x-axis numbers
      ax1.tick_params(axis='y', labelsize=18, labelcolor='black')
      
      ax1.axvline(x=optimal_coverage, color='firebrick', linestyle='--', lw=2, label=f'Optimal  Coverage: {optimal_coverage:.2f}')

      ax1.set_xlim(left=0, right=self.results_df['Coverage'].max())  
      ax1.grid(False)
       
    # Plot Accuracy on the same graph
      ax2 = ax1.twinx()
      ax2.set_ylabel('Accuracy', fontsize=20, color='#1f77b4')
      ax2.plot(self.results_df['Coverage'], self.results_df['Accuracy'], label='Accuracy',
             marker='x', linestyle='--', color='#1f77b4', linewidth=2)
      ax2.tick_params(axis='y', labelsize=18,labelcolor='#1f77b4')
      ax2.spines['top'].set_visible(False)  # Remove top spine
      ax2.grid(False)
    # Plot F1 Score on the same graph
      ax3 = ax1.twinx()
      ax3.spines['right'].set_position(('outward', 60))
      ax3.set_ylabel('F1 Score', fontsize=20, color='#2ca02c')
      ax3.plot(self.results_df['Coverage'], self.results_df['F1 Score'], label='F1 Score',
             marker='^', linestyle='-.', color='#2ca02c', linewidth=2)
      ax3.tick_params(axis='y',labelsize=18, labelcolor='#2ca02c')
      ax3.spines['top'].set_visible(False)  # Remove top spine
      ax3.grid(False)
      fig.tight_layout(pad=3) 
      lines, labels = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      lines3, labels3 = ax3.get_legend_handles_labels()
      ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='lower left', fontsize=16,frameon=False)

      plt.savefig(os.path.join(save_dir, 'Uncertainty_prediction.png'), dpi=600, bbox_inches='tight')
     
    def filter_predictions_with_point(self, x_data, y_data, y_prob, save_dir=None):
    
     confidence_scores = 2 * abs(y_prob - 0.5)
     optimal_data = self.find_dynamic_optimal_point()
     if optimal_data is None:
        logging.warning("No optimal point found. Filtering cannot proceed.")
        return None

     optimal_margin = optimal_data['Confidence Threshold']

    # Apply filtering based on the optimal margin
     high_confidence_indices = confidence_scores >= optimal_margin
     filtered_x_data = x_data[high_confidence_indices]
     filtered_y_data = y_data[high_confidence_indices]
     filtered_y_prob = y_prob[high_confidence_indices]

     if filtered_y_prob.ndim == 1:
        y_pred = (filtered_y_prob > self.th).astype(int)
     else:
        y_pred = (filtered_y_prob[:, 1] > self.th).astype(int)

     accuracy = accuracy_score(filtered_y_data, y_pred)

     uncertain_predictions_count = len(y_data) - len(filtered_y_data)
     total_predictions = len(y_data)
     percent_uncertain = (uncertain_predictions_count / total_predictions) * 100

     if save_dir:
        results = {
            "Accuracy": accuracy,
            "Total Predictions": total_predictions,
            "Uncertain Predictions Count": uncertain_predictions_count,
            "Percentage of Uncertain Predictions": percent_uncertain
        }
        if save_dir:
         results_df = pd.DataFrame([results])
         results_path = get_unique_filename(os.path.join(save_dir, 'filtered_predictions_evaluation.csv'))
         results_df.to_csv(results_path, index=False)
         logging.info("Filtered predictions evaluation saved to '%s'", results_path)

     return {
        "Filtered X_data": filtered_x_data,
        "Filtered Y_data": filtered_y_data,
        "Filtered Y_prob": filtered_y_prob,
        "Accuracy": accuracy,
        "Total Predictions": total_predictions,
        "Uncertain Predictions Count": uncertain_predictions_count,
        "Percentage of Uncertain Predictions": percent_uncertain
     }



class ModelCalibrator:
    def __init__(self, model):
        self.model = model
        self.iso_reg = None
        self.threshold = 0.5  

    def predict_probabilities(self, x, converge_method='average'):
        layers_prob = np.array(self.model.predict2(x))
        y_prob, _ = converge_layers_prob(layers_prob, layers_prob, method=converge_method)
        return y_prob

    def calibrate_model(self, x_train, y_train, x_valid, y_valid, converge_method='average'):
        layers_prob_train = np.array(self.model.predict2(x_train))
        layers_prob_valid = np.array(self.model.predict2(x_valid))
        y_prob_train, y_prob_valid = converge_layers_prob(layers_prob_train, layers_prob_valid, y_valid, method=converge_method)

        self.iso_reg = IsotonicRegression(out_of_bounds='clip')

        y_train = y_train.ravel()
        y_valid = y_valid.ravel()

        if len(y_prob_valid) != len(y_valid):
            raise ValueError(f"Inconsistent lengths: y_prob_valid has length {len(y_prob_valid)} but y_valid has length {len(y_valid)}")

        self.iso_reg.fit(y_prob_valid, y_valid)
        y_prob_valid_calibrated = self.iso_reg.transform(y_prob_valid)
        y_prob_train_calibrated = self.iso_reg.transform(y_prob_train)

        # Set optimal threshold based on validation set
        self.threshold = self.get_best_threshold(y_valid, y_prob_valid_calibrated)

        return y_prob_train_calibrated, y_prob_valid_calibrated

    def calibrate_probabilities(self, x, converge_method='average'):
        y_prob = self.predict_probabilities(x, converge_method)
        return self.iso_reg.transform(y_prob)

    def calibrate_test(self, x_test, y_test, converge_method='average'):
        layers_prob_test = np.array(self.model.predict2(x_test))
        y_prob_test, _ = converge_layers_prob(layers_prob_test, layers_prob_test, method=converge_method)
        y_prob_test_calibrated = self.iso_reg.transform(y_prob_test)

        test_accuracy = accuracy_score(y_test, (y_prob_test_calibrated >= self.threshold).astype(int))
        test_brier_score = brier_score_loss(y_test, y_prob_test_calibrated)


        logging.info(f'Test Accuracy after calibration: {test_accuracy}')
        logging.info(f'Test Brier Score after calibration: {test_brier_score}')

        return y_prob_test_calibrated

    def predict_with_threshold(self, x):
        y_prob = self.predict_probabilities(x)
        return (y_prob >= self.threshold).astype(int)

    def plot_calibration_curve(self, y_true, y_prob_uncalibrated, y_prob_calibrated, title):
     plt.figure(figsize=(8, 6))

     prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10)
     plt.plot(prob_pred_uncalibrated, prob_true_uncalibrated, marker='o', linewidth=2, label='Uncalibrated')

    # Plot calibrated curve
     prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
     plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', linewidth=2, label='Calibrated (Isotonic)')
     plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Perfectly Calibrated')
     plt.xlabel('Mean Predicted Probability', fontsize=14)
     plt.ylabel('Fraction of Positives', fontsize=14)
     plt.title(title, fontsize=16)
     plt.legend()
     plt.grid(False)
     plt.tight_layout()
     plt.show()

    @staticmethod
    def get_best_threshold(y_true, y_prob):
        thresholds = np.arange(0.0, 1.0, 0.01)
        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold





