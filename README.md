# **Customer-Churn-Detection-Modeling**
This project builds a deep learning model using a Feedforward Neural Network (FNN) to predict customer churn with the Telco Customer Churn Dataset from Kaggle. Extensive feature engineering, including interaction, temporal, and behavioral features, enhances prediction accuracy and offers actionable insights for customer retention strategies.

## **A. Dataset Description**
The Telco Customer Churn Dataset is a widely used dataset in data science and machine learning, providing insights into customer churn behavior for a telecommunications company. The dataset contains information about customers' demographics, account details, service usage, and churn status, which is essential for understanding the factors that influence customer retention.

### **Key Features:**
* **1. Customer ID:** Unique identifier for each customer.
* **2. Demographics:** Information such as gender, seniority, and partner status.
* **3. Account Information:**
   * **Tenure:** Number of months the customer has been with the company.
   * **Contract Type:** Type of contract (e.g., month-to-month, one year, two year).
   * **Payment Method:** How the customer pays for their services.
   * **Paperless Billing:** Whether the customer uses paperless billing.
* **4. Service Usage:**
   * **Internet Service:** Whether the customer has internet service (and the type of service).
   * **Online Security, Online Backup, Device Protection, Tech Support:** Whether the customer subscribes to these services.
   * **Streaming TV/Movies:** Whether the customer uses streaming services.
* **6. Churn:** The target variable indicating whether the customer has churned (1) or not (0).
* **7. Other Attributes:** Includes monthly charges, total charges, and customer location (state, region).
### **URL:**
You can access the dataset on Kaggle through the following link:
Telco Customer Churn Dataset

This dataset is useful for developing customer churn prediction models, performing customer segmentation, and understanding patterns in customer behavior, which can inform strategies for improving customer retention.

## **B. Data Overview**
The Telco Customer Churn dataset includes 7,000+ customer records from a telecom company. Each row represents a unique customer and contains features related to their demographic profile, service usage, account details, and whether they churned. The target variable Churn is binary (Yes/No), making this a supervised classification problem.

## **C. Exploratory Data Analysis (EDA)**
* **Univariate Analysis:**
   * **Numerical Features:** Distribution plots, box plots, and summary statistics were used to explore features like tenure, MonthlyCharges, and TotalCharges.
   * **Categorical Features:** Count plots were used to assess feature distributions (e.g., Contract, InternetService, PaymentMethod).
* **Bivariate Analysis:**
   * **Numerical vs Churn:** Violin plots and grouped summary statistics showed how numerical values varied across churn groups.
   * **Categorical vs Churn:** Crosstabs and stacked bar plots revealed churn likelihoods across categories (e.g., higher churn among month-to-month customers).

## **D. Handling Missing Values**
* **Missing values were primarily found in TotalCharges. These were handled by:**
   * **Converting invalid string entries to NaN**
   * **Imputing with median values for numerical consistency**
* **Other features had negligible or no missingness, and were retained after confirming integrity.**

## **E. Feature Engineering**
To enhance predictive performance, we performed advanced feature engineering, expanding the dataset meaningfully:
* **Interaction & Aggregate Features:**
   * Combined features like **Contract × InternetService** to create compound behavioral categories.
   * Aggregated service indicators (e.g., number of services subscribed).
* **Behavioral Features:**
   * Derived **ServiceCount** as total number of add-on services per customer.
   * Created **HighUsage** indicators from monthly charges and total tenure.
* **Temporal Features:**
   * Derived **TenureGroup** to group customers based on lifecycle (e.g., new vs loyal).
   * Extracted tenure-based activity metrics to assess customer maturity.
* **Customer Engagement Features:**
   * Features like **HasStreaming**, **HasSecurity**, and **DigitalBundle** to capture engagement levels.
* **Churn Propensity Features:**
   * Created risk groups using combinations of contract type, support services, and paperless billing.
* **Demographic Features:**
   * Consolidated demographic indicators such as **IsSeniorCitizen**, **HasPartner**, and **HasDependents**.
* **Categorical Encoding:**
   * All categorical variables were transformed using one-hot encoding to prepare data for the neural network.

## **F. Data Splitting**
The dataset was split into training and testing subsets using an 80/20 ratio to ensure generalization

## **G. Model Building: Feedforward Neural Network (FNN)**

**Architecture:**
* **Input Layer:** Matches the number of processed features
* **Hidden Layers:** Multiple dense layers with ReLU activation
* **Dropout:** Applied to reduce overfitting
* **Output Layer:** Single neuron with sigmoid activation for binary classification

**Training:**
* **Binary cross-entropy loss**
* **Adam optimizer**
* **Early stopping to prevent overfitting**

**Evaluation:**
* **Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC**
* **Confusion matrix and ROC curve plotted to assess model performance**

**Interpretation:**
* **Feature importance examined via SHAP or permutation importance**
* **Insights translated into business terms for actionable retention strategies**
