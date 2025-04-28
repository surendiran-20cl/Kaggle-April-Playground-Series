


# Podcast Listening Time Prediction - Kaggle Playground Series (April 2025)

https://www.kaggle.com/competitions/playground-series-s5e4/overview

##  Introduction
This project focuses on predicting the listening time (in minutes) for podcast episodes as part of the Kaggle Tabular Playground Series - April 2025. The goal is to build a robust machine learning model that minimizes Root Mean Squared Error (RMSE) on the test data.

##  Dataset
- **train.csv**: Training dataset with features and target (`Listening_Time_minutes`).
- **test.csv**: Test dataset without target labels.
- **sample_submission.csv**: Format for final predictions.

The dataset is synthetically generated based on a deep learning model trained on real-world podcast listening behavior.

##  Objective
- Predict `Listening_Time_minutes` for each podcast episode.
- Minimize RMSE on unseen data.

##  Approach
- Exploratory Data Analysis (EDA) to understand feature distributions.
- Feature preprocessing including handling categorical variables.
- Built multiple regression models:
  - Random Forest Regressor
  - LightGBM Regressor
  - ExtraTrees Regressor
- Hyperparameter tuning for LightGBM.
- Model evaluation using RMSE on validation set.
- Selected Random Forest as the final model based on performance.

##  Results
| Model                | Validation RMSE |
|----------------------|-----------------|
| Random Forest         | **12.6630**  |
| LightGBM (default)    | 12.9668 |
| ExtraTrees Regressor  | 12.6693 |

Random Forest achieved the best RMSE and was used for the final submission.

##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/surendiran-20cl/Kaggle-April-Playground-Series.git
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn lightgbm
   ```
3. Open the notebook or script in your preferred environment (Google Colab recommended).
4. Run all cells to train the model and generate `submission.csv`.

##  Submission
- Predictions were generated using the trained Random Forest model.
- Submission format:
  ```
  id,Listening_Time_minutes
  750000,45.437
  750001,44.981
  ...
  ```

##  Libraries Used
- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- LightGBM

##  Acknowledgements
- Kaggle Playground Series team for organizing the competition and providing the dataset.
- Open-source community for supporting machine learning development.



> **Note:** This project was implemented using Google Colab for easy access to GPU/TPU acceleration.
