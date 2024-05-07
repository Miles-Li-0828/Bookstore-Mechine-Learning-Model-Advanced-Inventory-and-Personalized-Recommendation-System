# Machine Learning Approaches for Bookstore Inventory Management: Book Rating Predictive System and Segmented Book Recommendation System #

## Project Description
--------

This project explores the application of machine learning techniques in revolutionizing bookstore management strategies and customer interactions. Leveraging data analysis and preprocessing, decision tree analysis, and K-Means clustering, the project aims to enhance decision-making processes and improve business management practices within the bookstore industry.

The project begins with a comprehensive data preprocessing phase, addressing missing values, anomalies, and standardizing textual data. This ensures the dataset is clean, consistent, and ready for analysis. Subsequently, a decision tree prediction system is implemented to predict book ratings, enabling informed inventory management decisions.

Additionally, K-Means clustering is employed for books segmentation, facilitating tailored book recommendations based on customer preferences. Through feature selection, model training, and evaluation, the project demonstrates the transformative potential of machine learning in optimizing bookstore operations and enriching customer experiences.

# CODE IMPLEMENTATION #

### Preprocess.py

The `preprocess.py` file is an essential component of the project, responsible for preprocessing the raw data before feeding it into the machine learning models. This script employs various preprocessing techniques to ensure the quality and integrity of the dataset, making it suitable for analysis and model training.

#### Functions:

1. **weighted_age_dect**
   - This function calculates the weights for different age groups based on the distribution of user ages in the raw data.
   - Returns a dictionary containing the weights for each age group.

2. **ages_imputation**
   - Utilizes the weights generated by the `weighted_age_dect` function to impute missing age data.
   - Ensures that the imputed age distribution closely matches that of the raw data.
   - Handles erroneous age data exceeding 100 in a consistent manner.

3. **country_imputation**
   - Imputes missing country data by randomly selecting countries from the known range of countries in the dataset.
   - Ensures that only valid country names are selected to maintain data integrity.

4. **city_imputation**
   - Imputes missing city data by randomly selecting cities from the known range of cities in the dataset.
   - Ensures that only valid city names are selected to maintain data integrity.

5. **state_imputation**
   - Imputes missing state data by randomly selecting states from the known range of states in the dataset.
   - Ensures that only valid state names are selected to maintain data integrity.

6. **author_imputation**
   - Fills missing author data for books with "NO AUTHOR" to maintain consistency in the dataset.

7. **discretising**
   - Discretizes user ages into 10-year bins, ranging from 0 to 100.
   - Discretizes ratings into three categories: "low," "medium," and "high."
   - Merges the DataFrames of book ratings, book information, and user information.

8. **text_process**
   - Normalizes text data by transforming all text into lowercase.
   - Handles mismatches in country formats to ensure uniformity in the dataset.

9. **compute_probability**
   - Computes the probability of a feature in the dataset.

10. **compute_entropy**
    - Computes the entropy of a feature in the dataset.

11. **compute_conditional_entropy**
    - Computes the conditional entropy of two features in the dataset.

12. **compute_information_gain**
    - Computes the information gain of a feature with respect to a target variable in the dataset.

### Final.ipynb

The `Final.ipynb` notebook encapsulates the entire project workflow, from data preprocessing to the implementation and evaluation of machine learning models for bookstore management. Below is a breakdown of the contents and functionality of this notebook:

#### 1. Data Preprocessing

The initial section of the notebook focuses on preparing the raw dataset for analysis and modeling. Key preprocessing steps include:

- **Missing Data Handling**: 
  - Abnormal data points, such as extreme ages, are removed.
  - Missing age values are imputed based on the distribution of age groups in the dataset.
  - Missing location data (country, city, state) is imputed with valid and realistic values.
  
- **Text Processing**: 
  - Standardization of country names and author information.
  - Removal of special characters and normalization of text data.
  
- **Data Integration and Discretization**: 
  - Discretization of ages and ratings into meaningful categories.
  - Merging of datasets and final validation for missing values.

#### 2. Decision Tree Prediction System for Book Store

This section details the implementation of a decision tree model for predicting book ratings. Key steps include:

- **Feature Selection**: 
  - Calculation of Information Gain to select the most discriminative features.
  
- **Data Preparation**: 
  - Encoding of categorical features using OrdinalEncoder.
  
- **Model Training**: 
  - Utilization of an entropy-based decision tree model for training.
  
- **Cross Validation**: 
  - Ten-fold cross-validation for model validation and evaluation.

#### 3. K-Means Clustering for Books Segmentation

The following part of the notebook focuses on books segmentation using K-means clustering. Key steps include:

- **Selection of K-means Clustering**: 
  - Choice of K-means due to its efficiency and clear cluster boundaries.
  
- **Encoding of Book Titles**: 
  - Utilization of Bag-of-Words (BoW) technique for numerical representation.
  
- **Preprocessing of Book Titles**: 
  - Removal of punctuation, stop-words, and lowercase conversion.
  
- **Execution of Elbow Method and K-means Clustering**: 
  - Determination of optimal cluster number using the elbow method.
  - Execution of K-means clustering to group similar books.
  
- **Recommendation Strategy**: 
  - Selection of high-rated books within clusters for tailored recommendations.

# FIGURES FOR DATA ANALYSIS #

- **ConfusionMatrix.png**:
  - This figure shows the confusion matrix of decision tree.

- **DistributionOfAgeGroups.png**:
  - This figure shows the distribution of users' age groups after imputation and discretising.

- **DistributionOfAgesRawData**:
  - This figure shows the distribution of users' ages.

- **DistributionOfRatingCate.png**:
  - This figure shows the distribution of books ratings after discretising from training set and test set.

- **DistributionOfRatings.png**:
  - This figure shows the distribution of books ratings.

- **K-ElbowForBookTitles.png**:
  - This figure shows the K-Elbow method while applying K-Mean technique.

# MAIN DATASETS #

BX-Books.csv
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Title**: Title of the book.
- **Book-Author**: Author(s) of the book.
- **Year-Of-Publication**: Year when the book was published.
- **Book-Publisher**: Publisher of the book.
- Total Rows: 18,185

BX-Ratings.csv
- **User-ID**: Unique identifier for users.
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Rating**: Rating given by users to books.
- Total Rows: 204,146

BX-Users.csv
- **User-ID**: Unique identifier for users.
- **User-City**: City where the user is located.
- **User-State**: State where the user is located.
- **User-Country**: Country where the user is located.
- **User-Age**: Age of the user.
- Total Rows: 48,299

# DATA SETS FOR RECOMMENDATION SYSTEMS #

BX-NewBooks.csv
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Title**: Title of the book.
- **Book-Author**: Author(s) of the book.
- **Year-Of-Publication**: Year when the book was published.
- **Book-Publisher**: Publisher of the book.
- Total Rows: 8,924

BX-NewBooks-Ratings.csv
- **User-ID**: Unique identifier for users.
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Rating**: Rating given by users to books.
- Total Rows: 26,772

BX-NewBooks-Users.csv
- **User-ID**: Unique identifier for users.
- **User-City**: City where the user is located.
- **User-State**: State where the user is located.
- **User-Country**: Country where the user is located.
- **User-Age**: Age of the user.
- Total Rows: 8,520
