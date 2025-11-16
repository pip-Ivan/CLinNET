import os 
import numpy as np 
from clinnet.model import CLinNET,get_callbacks
from clinnet.shap import SHAP
from clinnet.sankey import Sankey
import pandas as pd
import yaml
from copy import deepcopy

class CV:
    def __init__(self, config_path='config/cv_config.yaml'):
        """
        Initialize CV with configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set default attributes
        self.data_class = None  # Set this before running CV
        self.tissue = self.config['defaults']['tissue']
        self.saving_dir = self.config['defaults']['saving_dir']
        self.n_split = self.config['defaults']['data_params']['n_split']
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_fold_params(self, fold):
        """
        Get parameters for a specific fold by merging defaults with fold-specific overrides.
        
        Args:
            fold: Fold number
            
        Returns:
            Tuple of (data_params, model_params, tissue, saving_dir)
        """
        # Start with deep copy of defaults
        data_params = deepcopy(self.config['defaults']['data_params'])
        model_params = deepcopy(self.config['defaults']['model_params'])
        tissue = self.config['defaults']['tissue']
        saving_dir = self.config['defaults']['saving_dir']
        
        # Apply fold-specific overrides if they exist
        if 'folds' in self.config and fold in self.config['folds']:
            fold_config = self.config['folds'][fold]
            
            # Check if fold_config is not None (empty folds in YAML become None)
            if fold_config is not None:
                if 'data_params' in fold_config:
                    self._deep_update(data_params, fold_config['data_params'])
                
                if 'model_params' in fold_config:
                    self._deep_update(model_params, fold_config['model_params'])
                
                if 'tissue' in fold_config:
                    tissue = fold_config['tissue']
                
                if 'saving_dir' in fold_config:
                    saving_dir = fold_config['saving_dir']
        
        return data_params, model_params, tissue, saving_dir
    
    @staticmethod
    def _deep_update(base_dict, update_dict):
        """Recursively update base_dict with values from update_dict."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                CV._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def run_one_fold(self, fold):
        """Run one fold with fold-specific parameters."""
        # Get fold-specific parameters
        data_params, model_params, tissue, saving_dir = self._get_fold_params(fold)
        
        # Load specific fold data
        data = self.data_class(**data_params)
        x_train, y_train, x_valid, y_valid, x_test, y_test, genes, gene_status, class_weight = data.get_kf(kf=fold)

        # Build and train model
        clinnet_model = CLinNET(genes, gene_status, tissue=tissue, 
                                saving_dir=f"{saving_dir}/fold_{fold}", 
                                **model_params['build'])
        
        train_params = model_params['train'].copy()
        train_params.pop('callbacks', None)
        

        clinnet_model.train(x_train, y_train, x_valid, y_valid, 
                           class_weight=class_weight, **model_params['train'])
        clinnet_model.evaluate(x_valid, y_valid, x_test, y_test, converge_method='average')
        clinnet_model.save_predictions(x_train=x_train, y_train=y_train, 
                                      x_valid=x_valid, y_valid=y_valid, 
                                      x_test=x_test, y_test=y_test)

        # SHAP
        shap = SHAP(clinnet_model, train_n_sample=500, test_n_sample=300)
        shap.get_layer_shap(clinnet_model.model, x_train=x_train, x_test=x_test, 
                           y_train=y_train, y_test=y_test)
        shap.save_shap_csv()

        # Sankey
        sankey = Sankey(shap.graph, shap.interpret_dir, sv_norm=shap.sv_norm, 
                       gene_status=gene_status)
        sankey.plot_sankey(use_abb=True, display=False)

    def run_cross_validation(self):
        """Run cross-validation across all folds."""
        for fold in range(1, self.n_split + 1):
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

            # Create directory for aggregated results
            result_dir = f"result/{self.saving_dir}/aggregated_result"
            os.makedirs(result_dir, exist_ok=True)

            # Write out CSV
            csv_path = os.path.join(result_dir, "metrics_summary.csv")
            metrics_df.to_csv(csv_path)
            print(f"Cross-validation metrics saved to {csv_path}.")
        else:
            print("No metrics were found to aggregate.")

    def collect_cv_metrics(self):
        """Collect metrics from each fold's metrics.txt."""
        all_metrics = []

        for fold in range(1, self.n_split + 1):
            # Get fold-specific saving_dir
            _, _, _, saving_dir = self._get_fold_params(fold)
            _, _, tissue, _ = self._get_fold_params(fold)
            
            metrics_file = f"result/{saving_dir}/fold_{fold}/{tissue}/metrics.txt"
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
        """Aggregate results across all folds."""
        result_dir = f'result/{self.saving_dir}/aggregated_result'
        os.makedirs(result_dir, exist_ok=True)
        
        # Use fold 1 parameters to get gene info
        data_params, _, tissue, _ = self._get_fold_params(1)
        data = self.data_class(**data_params)
        _, _, _, _, _, _, genes, gene_status, _ = data.get_kf(kf=1)
        
        # Load fold 1 SHAP values
        clinnet_model = CLinNET(genes, gene_status, tissue=tissue, 
                                saving_dir=f"{self.saving_dir}/fold_1", 
                                **self.config['defaults']['model_params']['build'])
        shap = SHAP(clinnet_model, train_n_sample=1000, test_n_sample=1000, 
                   saving_dir=result_dir)
        
        shap_values = np.load(f'result/{self.saving_dir}/fold_1/{tissue}/interpretability/SHAP/shap_values_normalized.npz', 
                             allow_pickle=True)
        sv, sv_norm = shap_values.values()
        sv, sv_norm = sv.item(), sv_norm.item()
        
        # Aggregate across folds
        for fold in range(2, self.n_split + 1):
            shap_values = np.load(f'result/{self.saving_dir}/fold_{fold}/{tissue}/interpretability/SHAP/shap_values_normalized.npz', 
                                 allow_pickle=True)
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
        
        # Average
        for k in sv.keys():
            sv[k] /= self.n_split
            sv_norm[k] /= self.n_split
        
        shap.sv_norm = sv_norm
        shap.sv = sv
        shap.get_rank_index()
        shap.save_shap_csv()
        
        sankey = Sankey(shap.graph, shap.interpret_dir, sv_norm=shap.sv_norm, 
                       gene_status=gene_status, saving_dir=result_dir)
        print("Aggregated Sankey for all folds:")
        sankey.plot_sankey(use_abb=True, display=True)