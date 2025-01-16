import os 
import numpy as np 
from clinnet.model import CLinNET
from clinnet.shap import SHAP
from clinnet.sankey import Sankey
import pandas as pd

class CV:
    def __init__(self, data_class, data_params, model_params, tissue, saving_dir):
        self.data_class = data_class
        self.data_params = data_params
        self.model_params = model_params
        self.tissue = tissue
        self.saving_dir = saving_dir

    def run_one_fold(self, fold):
        # load specific fold data
        data = self.data_class(**self.data_params)
        x_train, y_train, x_valid, y_valid, x_test, y_test, genes, gene_status, class_weight = data.get_kf(kf=fold)

        # build and train model
        clinnet_model = CLinNET(genes, gene_status, tissue=self.tissue, saving_dir=f"{self.saving_dir}/fold_{fold}", **self.model_params['build'])
        clinnet_model.train(x_train, y_train, x_valid, y_valid, class_weight=class_weight, **self.model_params['train'])
        clinnet_model.evaluate(x_valid, y_valid, x_test, y_test, converge_method='average')
        clinnet_model.save_predictions(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test)

        # shap
        shap = SHAP(clinnet_model, train_n_sample=1000, test_n_sample=1500)
        shap.get_layer_shap(clinnet_model.model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        shap.save_shap_csv()

        # sankey
        sankey = Sankey(shap.graph, shap.interpret_dir, sv_norm=shap.sv_norm, gene_status=gene_status)
        sankey.plot_sankey(use_abb=True,display=False)

    def run_cross_validation(self):
        for fold in range(1, self.data_params['n_split'] + 1):
            print(f"Fold: {fold}")
            self.run_one_fold(fold)
    
        
        metrics_df = self.collect_cv_metrics()
        if not metrics_df.empty:
            # Calculate mean and std across folds
            mean_series = metrics_df.mean()
            std_series = metrics_df.std()

            # Insert mean and std as new rows
            metrics_df.loc['mean'] = mean_series
            metrics_df.loc['std'] = std_series

            # Create directory for aggregated results (if needed)
            result_dir = f"result/{self.saving_dir}/aggregated_result"
            os.makedirs(result_dir, exist_ok=True)

            # Write out CSV
            csv_path = os.path.join(result_dir, "metrics_summary.csv")
            metrics_df.to_csv(csv_path)
            print(f"Cross-validation metrics saved to {csv_path}.")
        else:
            print("No metrics were found to aggregate.")

    def collect_cv_metrics(self):
        """
        Collects metrics from each fold's metrics.txt.
        """
        n_splits = self.data_params['n_split']  # integer
        all_metrics = []

        for fold in range(1, n_splits + 1):
            metrics_file = f"result/{self.saving_dir}/fold_{fold}/{self.tissue}/metrics.txt"
            if not os.path.exists(metrics_file):
                print(f"Warning: {metrics_file} not found. Skipping.")
                continue

            fold_metrics = {'Fold': fold}
            with open(metrics_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ':' in line:
                        key, val = line.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                        fold_metrics[key] = val

            all_metrics.append(fold_metrics)

        # Create DataFrame, index by fold
        if not all_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(all_metrics)
        df.set_index('Fold', inplace=True, drop=True)
        return df

    
    def aggregate_result(self):
        result_dir = f'result/{self.saving_dir}/aggregated_result'
        os.makedirs(result_dir)
        data = self.data_class(**self.data_params)
        _, _, _, _, _, _, genes, gene_status, _ = data.get_kf(kf=1)
        clinnet_model = CLinNET(genes, gene_status, tissue=self.tissue, saving_dir=f"{self.saving_dir}/fold_1", **self.model_params['build'])
        shap = SHAP(clinnet_model, train_n_sample=1000, test_n_sample=1000, saving_dir=result_dir)
        shap_values = np.load(f'result/{self.saving_dir}/fold_1/{self.tissue}/interpretability/SHAP/shap_values_normalized.npz', allow_pickle=True)
        sv, sv_norm = shap_values.values()
        sv, sv_norm = sv.item(), sv_norm.item()
        for fold in range(2, self.data_params['n_split'] + 1):
            shap_values = np.load(f'result/{self.saving_dir}/fold_{fold}/{self.tissue}/interpretability/SHAP/shap_values_normalized.npz', allow_pickle=True)
            sv_, sv_norm_ = shap_values.values()
            sv_, sv_norm_ = sv_.item(), sv_norm_.item()
            for k in sv.keys():
                if sv_[k].shape != sv[k].shape:
                    idx = min(sv[k].shape[0], sv_[k].shape[0])
                    sv[k] = sv[k][:idx]
                    sv_norm[k] = sv_norm[k][:idx]
                    sv_[k] = sv_[k][:idx]
                    sv_norm_[k] = sv_norm_[k][:idx]
                sv[k] += sv_[k]
                sv_norm[k] += sv_norm_[k]
        for k in sv.keys():
            sv[k] /= self.data_params['n_split']
            sv_norm[k] /= self.data_params['n_split']
        shap.sv_norm = sv_norm
        shap.sv = sv
        shap.get_rank_index()
        shap.save_shap_csv()
        sankey = Sankey(shap.graph, shap.interpret_dir, sv_norm=shap.sv_norm, gene_status=gene_status, saving_dir=result_dir)
        print("Agtigated Sankey for all folds:")
        sankey.plot_sankey(use_abb=True,display=True)