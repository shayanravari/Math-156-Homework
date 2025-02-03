# Logistic Regression with Mini-Batch SGD on the Breast Cancer Dataset

---

## 1. Data Loading and Preprocessing

- **load_breast_cancer:** Loads the Breast Cancer dataset using scikit-learn’s built-in function.

- **train_test_split:** Splits the dataset into three parts:
  - **Training Set:** 60%
  - **Validation Set:** 20%
  - **Test Set:** 20%

- **Data Augmentation:** A bias is added to the feature matrix $X$ by augmenting it with a column of ones, resulting in $X_{bias}$.  
  Additionally, the class distribution in the combined training and validation set is reported to ensure that the classes are balanced.

*In this section, the dataset is loaded and prepared for modeling. Preprocessing includes splitting the data appropriately and augmenting X with a bias term to facilitate the logistic regression formulation.*

---

## 2. Model Training

- **LogisticRegressionSGD:** A logistic regression model is implemented using mini-batch SGD. The model:
  - Uses the sigmoid function to map linear combinations of features to probabilities.
  - Iteratively updates the weights using mini-batches of data.
  - Includes hyperparameters such as the learning rate, batch size, and maximum number of iterations.
  - Initializes weights randomly from a standard Gaussian distribution.

- **Sigmoid Function:** The sigmoid function is used in Logistic regression to determine the probabilities that an example belongs to a certain class.

*In this section, the model is trained iteratively. For each mini-batch, the gradient of the loss (negative log-likelihood) is computed and the weights are updated.*

---

## 3. Evaluation

- **Prediction Function:** The model’s `predict` method generates binary class predictions by applying the sigmoid function and thresholding the output (default threshold = 0.5).

- **Evaluation Metrics:** The performance on the test set is evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

- **Performance Reporting:** Metrics are computed using scikit-learn’s metric functions and the results are printed.

*This section verifies how well the model performs on unseen data. By comparing the predicted labels with the true labels, we can quantify the model's effectiveness.*
