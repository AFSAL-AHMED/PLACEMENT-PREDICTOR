# Student Placement Prediction System

A Machine Learning web application that predicts whether a student will be placed or not based on their academic performance.

## Project Overview

This project uses Machine Learning to predict student placement outcomes based on:
- Academic performance (SSC, HSC, Degree, MBA percentages)
- Educational background (boards, streams, specializations)
- Work experience
- Entrance test scores

## Machine Learning Models

Two models were trained and compared:

1. Logistic Regression - Baseline model (84% accuracy)
2. Random Forest - Best performer (86% accuracy)

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 86.05% |
| Precision | 93.55% |
| Recall | 90.32% |
| F1-Score | 91.88% |

### Important Features

1. MBA Percentage (25.43%)
2. Degree Percentage (18.76%)
3. Entrance Test Score (16.54%)
4. SSC Percentage (12.34%)
5. Work Experience (9.87%)

## Project Structure

```
PEP AIML PROJECTS/
|
|-- eda.py                              # ML pipeline (training and evaluation)
|-- app.py                              # Streamlit web application
|-- Placement_Data_Full_Class.csv       # Dataset
|-- placement_prediction_model.pkl      # Trained ML model
|-- label_encoders.pkl                  # Label encoding mappings
|-- feature_names.pkl                   # Feature names list
|-- confusion_matrices.png              # Evaluation visualization
|-- model_comparison.png                # Performance comparison chart
|-- requirements.txt                    # Python dependencies
|-- README.md                           # Project documentation
```

## Requirements

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone or download this repository

2. Install required packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit pillow
```

3. Run the training script (if models not created):

```bash
python eda.py
```

4. Launch the web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Go to "Make Prediction" tab
2. Enter student details:
   - SSC, HSC, Degree, MBA percentages
   - Board types and streams
   - Work experience (Yes/No)
   - Specialization and gender
3. Click "Predict Placement"
4. View prediction result and confidence score

## Dataset Information

Source: Kaggle - Campus Placement Dataset
Samples: 215 students
Features: 15 (14 input features + 1 target)

### Features Description

- gender: Male/Female
- ssc_p: Secondary school percentage
- ssc_b: Board of education (Central/Others)
- hsc_p: Higher secondary percentage
- hsc_b: Board of education
- hsc_s: Stream (Science/Commerce/Arts)
- degree_p: Degree percentage
- degree_t: Field of degree (Sci&Tech/Comm&Mgmt/Others)
- workex: Work experience (Yes/No)
- etest_p: Entrance test percentage
- specialisation: MBA specialization (Mkt&Fin/Mkt&HR)
- mba_p: MBA percentage
- status: Placed/Not Placed (Target variable)

## Model Training Process

1. Data Cleaning - Remove unnecessary columns, handle missing values
2. Label Encoding - Convert categorical text to numbers
3. Train/Test Split - 80% training, 20% testing with stratified split
4. Model Training - Train Logistic Regression and Random Forest
5. Evaluation - Calculate accuracy, precision, recall, F1-score
6. Model Persistence - Save best model using pickle

## Technologies Used

- Python 3.8+
- Pandas - Data manipulation
- NumPy - Numerical operations
- Scikit-learn - Machine learning
- Matplotlib and Seaborn - Visualizations
- Streamlit - Web application
- Pickle - Model persistence

## Results

- Best Model: Random Forest
- Test Accuracy: 86.05%
- The model correctly identifies 93% of placed students
- MBA percentage is the most important factor for placement prediction
