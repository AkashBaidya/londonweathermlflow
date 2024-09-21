
# 📊 Predicting London's Mean Temperature using MLflow 🌤️

This project focuses on predicting the mean temperature in London 🌆, using machine learning techniques and **MLflow** to track and compare models. We explore different regression models to forecast temperature over time, providing a robust approach to weather prediction.

---

## 📁 Project Structure

```
├── data/
│   └── london_weather.csv  # Dataset used for model training and evaluation
├── notebooks/
│   └── Weather_Prediction_with_MLflow.ipynb  # Main Jupyter notebook containing code
├── models/
│   └── model_artifacts/  # Saved model artifacts from MLflow
├── requirements.txt  # List of project dependencies
└── README.md  # Project overview
```

---

## 📚 Dataset

The dataset used is historical weather data for London, England, covering the years from 1940 to 2020 🌍. It contains various weather attributes, including:

- 🌡️ Mean Temperature (our target variable)
- 🌧️ Precipitation
- 💨 Wind Speed
- 🌦️ Cloud Cover

---

## 🚀 Project Objectives

The main goals of this project are to:
1. 📈 **Analyze** the weather dataset for London and perform exploratory data analysis (EDA).
2. 🤖 **Develop** regression models to predict the mean temperature using features like precipitation, wind speed, and cloud cover.
3. ⚙️ **Implement** MLflow to track the experiments, model parameters, metrics, and version control for the models.
4. 📊 **Evaluate** the models' performance using metrics such as RMSE, MAE, and R².

---

## 🛠️ Technology Stack

- **Programming Languages**: Python 🐍
- **Libraries**:
  - Scikit-learn for model development and evaluation
  - Pandas and NumPy for data manipulation
  - Matplotlib and Seaborn for data visualization 📊
- **Model Tracking**: MLflow
- **Version Control**: Git

---

## 🤖 Machine Learning Models

We implemented and tested the following machine learning models:

1. **Linear Regression**: A simple linear approach to predict temperature based on the available features.
2. **Random Forest Regressor**: A robust ensemble model to handle non-linearity in the dataset.
3. **Gradient Boosting Regressor**: A boosting technique to improve model performance.
4. **XGBoost**: An advanced ensemble model optimized for speed and performance.

Each model is evaluated using cross-validation techniques to ensure robust predictions.

---

## 🔬 Experiment Tracking with MLflow

MLflow is used to track and manage experiments during model training. Features include:

- 📝 Logging model parameters, metrics (e.g., RMSE, MAE, and R²), and versions.
- 📂 Saving model artifacts such as trained models and visualizations.
- 📊 Comparing the performance of different models and hyperparameter settings.

---

## 📊 Model Evaluation

We evaluated the models using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures the average squared difference between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.
- **R² Score**: Indicates how well the independent variables explain the variance in the target variable.

---

## ⚙️ Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/AkashBaidya/londonweathermlflow
   cd londonweathermlflow
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to start training the models:
   ```bash
   jupyter notebook notebooks/Predicting weather with MLflow.ipynb
   ```

4. Run MLflow to track experiments:
   ```bash
   mlflow ui
   ```

---

## 📈 Results

The best-performing model was **XGBoost**, which achieved an RMSE of **X.X** and an R² score of **X.XX**. MLflow was critical in tracking hyperparameters and performance metrics, allowing us to easily compare models and fine-tune them for better results.

---

## 📅 Future Work

- Improve feature engineering by incorporating additional weather attributes.
- Tune hyperparameters using advanced techniques such as Bayesian optimization.
- Deploy the model using a cloud platform like AWS or Google Cloud to provide real-time weather predictions.
- Apply time series analysis to improve prediction accuracy over longer periods.

---

## 🤝 Contribution

Feel free to submit issues or fork this repository and make contributions to improve the project!

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ✉️ Contact

For more details, you can reach out at:
- **Email**: akashbaidya2@gmmail.com
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/akashbaidya/)

