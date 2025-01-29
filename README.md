# CLinNET

## **A CLinically-Oriented Multi-Modal Deep Neural NETwork**

CLinNET is a cutting-edge deep learning framework designed to analyze and interpret clinical datasets effectively. This repository includes the source code, pre-trained models, and sample datasets supporting the scientific paper **"CLinNET"**.

---

## **Key Features**
- **Deep Learning for Clinical Data**: Specially tailored for clinical datasets with high-dimensional features and complex dependencies.
- **Reproducibility**: Pre-configured scripts and Jupyter notebooks for easy reproduction of experiments.
- **Modular Design**: Easily extendable for various clinical data formats and preprocessing pipelines.
- **Integration-Ready**: Seamlessly integrates into larger workflows for both research and applied settings.

---

## **Installation**

### **Prerequisites**
Before starting, ensure the following dependencies are installed:
- Python 3.8 or higher
- TensorFlow
- Jupyter Notebook
- Additional dependencies listed in `requirements.txt` or `environment.yml`

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/pip-Ivan/CLinNET.git
   cd CLinNET
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate tf_mps
   ```

3. Alternatively, use `pip` to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How to Use**

### **Example Usage**
Below is an example of how to load data, train the model, evaluate it, and generate visualizations:

```python
# Import necessary libraries
from clinnet.data_loader_sydney import SydneyData
from clinnet.model import CLinNET
from clinnet.shap import SHAP
from clinnet.sankey import Sankey

# Load data
data = SydneyData(data_dir='data/sydney_data', balance='undersample')
x_train, y_train, x_valid, y_valid, x_test, y_test, genes, gene_status, class_weight = data.get_kf(kf=3)  # Get specific fold data

# Initialize the model
clinnet_model = CLinNET(genes, gene_status, tissue='brain', saving_dir='SydneyDataset_Run1')

# Train the model
clinnet_model.train(x_train, y_train, x_valid, y_valid, batch_size=1024, epochs=5, verbose=2)

# Evaluate the model
clinnet_model.evaluate(x_valid, y_valid, x_test, y_test, converge_method='average')

# Generate SHAP values for features
shap = SHAP(clinnet_model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
shap.save_shap_plot()
shap.save_shap_csv()

# Generate Sankey Diagram
sankey = Sankey(shap)
sankey.plot_sankey(use_abb=True)
```

---

## **Project Structure**
- `clinnet/`: Contains core modules for data loading, model training, and evaluation.
- `data/`: Directory to store datasets.
- `notebooks`: Pre-configured Jupyter notebooks for running experiments.
- `results/`: Directory to save results, including visualizations and evaluation metrics.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

### **Contributions**
Contributions to CLinNET are welcome! If you have any ideas or improvements, feel free to open an issue or submit a pull request.

---

### **Citation**
If you use CLinNET in your research, please cite the associated paper:

```
[CLinNET: An Interpretable and Uncertainty-Aware Deep Neural Network for Multi-Modal Clinical Genomics]
```

---

### **Acknowledgements**
Special thanks to the contributors and collaborators who made this project possible.

---

### **Contact**
For questions or issues, please contact:
- Repository owner: [pip-Ivan](https://github.com/pip-Ivan)




