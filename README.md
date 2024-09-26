# Human Activity Recognition using ML and Deep Learning on Raw Signals and Feature-Engineered Data

## Project Overview
This project focuses on classifying human activities using a dataset from UCI's Human Activity Recognition (HAR) dataset. We perform extensive **Exploratory Data Analysis (EDA)**, apply **Machine Learning (ML)** models on **feature-engineered data**, and use **deep learning models (LSTM & GRU)** on raw accelerometer and gyroscope signals to classify six different activities. The overall goal is to compare the effectiveness of traditional ML approaches against advanced deep learning methods on both structured and unstructured data.

### Activities Classified:
The dataset contains the following activities:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## 1. Exploratory Data Analysis (EDA)

EDA was conducted on the feature-engineered dataset. The features extracted were related to:
- **Body Acceleration Magnitude**
- **Body Gyroscope Magnitude**
- **Total Acceleration Magnitude**
- **Angles between acceleration components and the gravity vector**

Visualizations were plotted for both **stationary** and **moving activities** to gain insights into how the feature distributions vary between activities.

### Key Insights:
- **Stationary activities** (e.g., Sitting, Standing, and Laying) showed clear separation based on Body Acceleration and Gyroscope Magnitude.
- **Moving activities** (e.g., Walking, Walking Upstairs, and Walking Downstairs) were better differentiated using Total Acceleration and Gyroscope components.
  
Below are a couple of example visualizations from the EDA phase:

![Stationary Activities](stationary_activities_plot.png)
![Moving Activities](moving_activities_plot.png)

The **angles** between body acceleration components and the gravity vector were also insightful in distinguishing between standing, sitting, and laying.

---

## 2. Machine Learning on Feature-Engineered Data

We applied several machine learning models to classify the activities using the feature-engineered dataset. A series of models were trained and evaluated, including:
- Logistic Regression
- Linear SVC
- rbf SVM Classifier
- Decision Tree
- Random Forest
- XGBoost

### Model Comparison:

| Model                | Accuracy | Error  |
|----------------------|----------|--------|
| Logistic Regression  | 96.27%   | 3.733% |
| Linear SVC           | 96.64%   | 3.359% |
| rbf SVM Classifier   | 96.27%   | 3.733% |
| Decision Tree        | 87.07%   | 12.93% |
| Random Forest        | 92.57%   | 7.431% |
| XGBoost              | 94.44%   | 5.565% |

#### Logistic Regression and Linear SVC emerged as the best-performing models on the feature-engineered data, each achieving over **96% accuracy**.

---

## 3. Deep Learning on Raw Signal Data

To push the boundaries beyond feature engineering, we trained **LSTM** and **GRU** models on raw signal data (accelerometer and gyroscope readings). The data consisted of three axes (x, y, z) for both accelerometer and gyroscope data, which were stacked and used as time-series input for the models.

### Data Structure:
- **Input Shape:** (samples, timesteps, features)
- **Timesteps:** 128
- **Features:** 9 (3 accelerometer axes + 3 gyroscope axes + 3 total acceleration axes)

### Model Architectures:

#### LSTM Model
The LSTM model was constructed using:
- **32 units** in the LSTM layer
- **Dropout of 0.5** to prevent overfitting
- **Dense layer** with 6 output nodes and **sigmoid activation** for multi-class classification.

#### GRU Model
Similarly, the GRU model used:
- **32 GRU units**
- **Dropout of 0.5**
- **Dense output layer** with 6 classes using **softmax activation**.

### Results:

#### LSTM Performance:
- Training Accuracy: **94.55%**
- Validation Accuracy: **90.77%**
  
#### GRU Performance:
- Training Accuracy: **94.77%**
- Validation Accuracy: **90.40%**

![LSTM Results](lstm_training_results.png)
![GRU Results](gru_training_results.png)

### Model Evaluation:
Both LSTM and GRU performed exceptionally well, with over **90% validation accuracy** on raw signals. However, **GRU** showed a slightly higher **training accuracy** compared to LSTM, making it a strong candidate for time-series modeling of human activities. Below is a comparison of both models' performances:

| Model   | Training Accuracy | Validation Accuracy |
|---------|-------------------|---------------------|
| LSTM    | 94.55%            | 90.77%              |
| GRU     | 94.77%            | 90.40%              |

---

## 4. Model Predictions & Evaluation

To test the models, we implemented a function to predict activities for a single sample and compared the predicted labels with the actual labels. The **confusion matrix** and **classification reports** were generated for better insights into the model's performance on each activity class.

### Confusion Matrix:

![Confusion Matrix](confusion_matrix.png)

The **confusion matrix** shows that both models correctly predicted the majority of activities, with occasional misclassifications between similar activities such as **walking upstairs** and **walking downstairs**.

---

## 5. Conclusion

In this project, we compared **traditional machine learning** models on **feature-engineered data** with **deep learning models** (LSTM, GRU) on **raw signal data**. The key takeaways are:
- **Feature engineering** paired with traditional models like **Logistic Regression** and **SVM** provided solid results with over **96% accuracy**.
- **LSTM** and **GRU** deep learning models excelled with raw signal data, achieving over **90% validation accuracy**.
- **GRU** slightly outperformed LSTM in terms of training accuracy but both models were comparable overall.

This project highlights the **effectiveness of deep learning** in leveraging raw signal data and showcases the power of **feature engineering** in traditional machine learning workflows.

---

## Future Work
- Further experimentation with hyperparameter tuning for deep learning models.
- Exploration of **Transformer-based models** to see if self-attention mechanisms can improve performance.
- Investigating real-time activity recognition using edge devices like smartphones or wearable sensors.

## References
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

