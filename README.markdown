# System Threat Forecaster

## Project Overview

The **System Threat Forecaster** project aims to predict the probability of a system being infected by various families of malware based on its telemetry data. This data is generated from threat reports collected by the system's antivirus software. The objective is to develop a robust machine learning model that can forewarn systems of potential compromises, enabling proactive security measures.

The dataset, sourced from a Kaggle competition, consists of telemetry data with features describing system properties and a binary target variable indicating malware infection. The goal is to predict the `target` variable for the test set using the training data provided in `train.csv`.

## Dataset Description

The dataset contains telemetry data for systems, with each row corresponding to a unique machine identified by a `MachineID`. The `target` column in `train.csv` indicates whether malware was detected (1) or not (0). The task is to predict the `target` for machines in `test.csv`.

### Files
- **train.csv**: Training dataset with features and `target` labels.
- **test.csv**: Test dataset for predicting `target`.
- **sample_submission.csv**: Sample submission file format.

### Key Columns
The dataset includes 75 features, with some of the most relevant ones listed below:
- **MachineID**: Unique identifier for each machine.
- **ProductName**: Name of the installed antivirus product.
- **EngineVersion**: Version of the antivirus engine.
- **AppVersion**: Version of the antivirus application.
- **SignatureVersion**: Version of the antivirus signatures.
- **IsBetaUser**: Indicates if the user is on a beta version (0 or 1).
- **RealTimeProtectionState**: Status of real-time protection.
- **IsPassiveModeEnabled**: Whether passive mode is enabled (0 or 1).
- **AntivirusConfigID**: Identifier for antivirus configuration.
- **NumAntivirusProductsInstalled**: Number of installed antivirus products.
- **NumAntivirusProductsEnabled**: Number of enabled antivirus products.
- **HasTpm**: Whether the machine has a Trusted Platform Module (0 or 1).
- **CountryID**: Identifier for the machine's country.
- **CityID**: Identifier for the machine's city.
- **GeoRegionID**: Identifier for the machine's organization or industry.
- **PlatformType**: Platform type derived from OS and processor.
- **Processor**: Processor architecture of the installed OS.
- **OSVersion**: Operating system version.
- **OSBuildNumber**: OS build number.
- **IsSystemProtected**: Whether the system has active protection (0 or 1).
- **FirewallEnabled**: Whether Windows Firewall is enabled (0 or 1).
- **TotalPhysicalRAMMB**: Total physical RAM in megabytes.
- **IsSecureBootEnabled**: Whether secure boot is enabled (0 or 1).
- **IsVirtualDevice**: Whether the machine is a virtual device (0 or 1).
- **IsGamer**: Whether the system is used for gaming (0 or 1).
- **RegionIdentifier**: Identifier for the region.
- **DateAS**: Malware signature dates.
- **DateOS**: Timestamps for the last OS update.

For a complete list of columns, refer to the dataset documentation.

## Methodology

### Data Preprocessing
- **Exploratory Data Analysis (EDA)**: Analyzed the dataset to understand feature distributions, null values, and correlations. Key findings include:
  - The training dataset has 100,000 rows and 75 features.
  - The test dataset has 10,000 rows.
  - Null values were present in several columns, with percentages ranging from 0% to 98% (e.g., `SMode` had 98% missing values).
- **Imputation**:
  - Median imputation for numerical columns like `NumAntivirusProductsEnabled` and `ProcessorCoreCount`.
  - Mode imputation for categorical/binary columns like `IsSystemProtected`, `IsVirtualDevice`, and `IsGamer`.
- **Feature Engineering**:
  - Created new features such as `EngineVersion_Risk`, `is_high_risk`, `protection_level`, `single_av`, `av_risk_category`, `Processor_Encoded`, `MDC2FormFactor_Encoded`, `ram_category`, `high_ram_risk`, `ram_scaled`, `arch_risk`, `is_amd64`, and `LicenseRisk_Encoded` to capture additional patterns.
  - Encoded categorical variables and scaled numerical features as needed.

### Model Development
Three machine learning models were developed and evaluated:
1. **Random Forest Classifier**:
   - Hyperparameters tuned using `RandomizedSearchCV`.
   - Best parameters: `n_estimators=100`, `min_samples_split=2`, `min_samples_leaf=1`, `max_depth=10`.
   - Score: **0.59820** (on the test set).
2. **XGBoost Classifier**:
   - Hyperparameters tuned using `RandomizedSearchCV`.
   - Best parameters: `subsample=0.7`, `n_estimators=100`, `min_child_weight=1`, `max_depth=3`, `learning_rate=0.1`.
   - Score: **0.60230** (on the test set, best performing model).
3. **LightGBM Classifier**:
   - Hyperparameters tuned using `RandomizedSearchCV`.
   - Best parameters: `subsample=0.9`, `num_leaves=70`, `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `colsample_bytree=0.9`.
   - Score: **0.60120** (on the test set).

### Visualization
Visualizations were generated to aid in understanding the data and model performance:
- **Null Value Analysis**: Bar plot showing the percentage of missing values per column, highlighting columns like `SMode` and `CityID` with high missingness.
- **Feature Importance**: Plots for XGBoost and LightGBM models, showing top features like `AntivirusConfigID`, `NumAntivirusProductsInstalled`, and `TotalPhysicalRAMMB` as strong predictors of malware infection.
- **Correlation Heatmap**: Visualized correlations between numerical features to identify multicollinearity.
- **Class Distribution**: Bar plot of the `target` variable, indicating a balanced or imbalanced dataset (depending on EDA findings).

Example code for feature importance visualization (using XGBoost):
```python
import matplotlib.pyplot as plt
import xgboost as xgb

xgb_model = XGBClassifier(subsample=0.7, n_estimators=100, min_child_weight=1, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb.plot_importance(xgb_model, max_num_features=10)
plt.savefig('feature_importance_xgb.png')
```

The visualizations are saved as PNG files (e.g., `feature_importance_xgb.png`, `null_values.png`) and can be found in the project repository.

## Results
The models were evaluated based on their performance on the test set, with the following scores:
- **Random Forest**: 0.59820
- **XGBoost**: **0.60230** (best performer)
- **LightGBM**: 0.60120

The **XGBoost model** was selected as the final model due to its highest score. Predictions were generated for the test set and saved in `submission.csv`, following the format of `sample_submission.csv`.

## Installation and Usage
To replicate the project:
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook (`21f1005422-notebook-t12025.ipynb`) to preprocess data, train models, and generate predictions.
4. Ensure the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) are in the correct directory.

### Requirements
- Python 3.10+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`

## Future Improvements
- **Feature Selection**: Use techniques like recursive feature elimination to reduce the feature set and improve model efficiency.
- **Ensemble Methods**: Combine predictions from Random Forest, XGBoost, and LightGBM using stacking or voting classifiers.
- **Handle Imbalanced Data**: If the `target` variable is imbalanced, apply techniques like SMOTE or class weighting.
- **Advanced Hyperparameter Tuning**: Use Bayesian optimization or Optuna for more efficient hyperparameter search.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Kaggle for providing the dataset and competition platform.
- The open-source community for tools like `scikit-learn`, `xgboost`, and `lightgbm`.