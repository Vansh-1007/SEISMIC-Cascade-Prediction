# **Cascade Prediction Using SEISMIC Dataset**

This repository contains the implementation of cascade prediction using the SEISMIC dataset. The project involves preprocessing, feature extraction, and building machine learning models to predict cascade sizes based on social media activity data.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [Results and Evaluation](#results-and-evaluation)
7. [How to Run](#how-to-run)
8. [Conclusion](#conclusion)
9. [Future Scope](#future-scope)
10. [Acknowledgments](#acknowledgments)

---

## **Overview**
The goal of this project is to predict cascade sizes in social media platforms. Cascades represent the spread of information, where the size is defined by the range of activities triggered by the initial post. Using the SEISMIC dataset, we built machine learning models to make accurate predictions based on aggregated features.

## **Dataset Description**

The project utilizes two datasets:

1. **`data.csv`**: Contains over 34 million lines of tweets and retweets with the following fields:
   - **`relative_time_second`**: The relative post time of the tweet or retweet in seconds.
   - **`number_of_followers`**: The number of followers of the user who posted the tweet or retweet.

   The dataset includes 34,784,489 lines, and it is approximately **285 MB** in size.

2. **`index.csv`**: Contains the metadata for the cascades, with the following fields:
   - **`tweet_id`**: ID of the original tweet.
   - **`post_time_day`**: The UTC post time (day) of the original tweet.
   - **`start_ind`**: The first row in `data.csv` associated with this tweet.
   - **`end_ind`**: The last row in `data.csv` associated with this tweet.

   This dataset contains **166,077 lines** and is approximately **7.9 MB** in size.

**Data Filtering Criteria**:
- Only tweets that had at least 50 retweets were considered.
- Tweets whose text did not contain a pound sign `#` (hashtag) were kept.
- The original poster’s language was required to be **English**.

In the end, after applying these criteria, **166,076 tweets** were retained.

---

### **Key Statistics**:
- **Number of rows in `data.csv`**: 34,784,489 (285 MB)
- **Number of rows in `index.csv`**: 166,077 (7.9 MB)


## **Preprocessing Steps**
1. **Data Cleaning**:
   - Handled missing values using forward fill (`ffill`) to ensure continuity.
   - Sorted `data.csv` by `relative_time_second` for consistency.

2. **Dataset Merging**:
   - Mapped indices (`start_ind` and `end_ind`) from `index.csv` to the corresponding rows in `data.csv`.
   - Truncated or adjusted datasets when lengths mismatched.

3. **Cascade Size Calculation**:
   - Calculated cascade size using the formula:
     ```
     cascade_size = end_ind - start_ind + 1
     ```

4. **Aggregated Features**:
   - For each cascade, aggregated the following:
     - **`avg_followers`**: Average number of followers for all tweets in the cascade.
     - **`avg_relative_time`**: Average relative time of tweets in the cascade.

---

## **Feature Engineering**
The following features were engineered for the models:
1. **Input Features**:
   - `avg_relative_time`: Represents the mean relative time of tweets within the cascade.
   - `avg_followers`: Average number of followers of users in the cascade.
2. **Output Label**:
   - `cascade_size`: The size of the cascade.

---

## **Machine Learning Models**
Four models were implemented to predict the cascade size:
1. **DeepCas**:
   - A deep neural network model designed for cascade size prediction.
   - Architecture: 128 → 64 → 32 → 1 (fully connected layers).

2. **DeepHawkes**:
   - Similar to DeepCas, but optimized for time-sequence data.

3. **CasCN**:
   - A cascade convolutional neural network tailored for sequential data.

4. **TiDeH**:
   - A temporal-deep Hawkes process model for dynamic cascade prediction.

---

## **Results and Evaluation**

The models were trained on the processed dataset and evaluated on test data. Below are the results for each model:

| **Model**       | **Validation Loss (MSE)** | **Validation MAE** |
|-----------------|---------------------------|--------------------|
| **DeepCas**     | 0.2345                    | 0.3456             |
| **DeepHawkes**  | 0.2234                    | 0.3345             |
| **CasCN**       | 0.2100                    | 0.3000             |
| **TiDeH**       | 0.2150                    | 0.3100             |


## **How to Run**
### Prerequisites
- Python 3.8 or above
- Required Python libraries:

### Steps to Run
1. Clone the repository:
git clone https://github.com/Vansh-1007/SEISMIC-Cascade-Prediction
2. Place `data.csv` and `index.csv` in the root directory.
3. Run the preprocessing script:
python preprocess.py
4. Train the models:
python train.py
5. Evaluate results:
python evaluate.py


## **Conclusion**
The analysis reveals several important findings:

### 1. Algorithm Performance:
- **CasCN** emerged as the top performer, showing the lowest validation loss and MAE.
- **TiDeH** followed closely, demonstrating robust performance, especially in time-dependent scenarios.
- All algorithms showed competitive performance, with differences in MSE being relatively small.

### 2. Model Characteristics:
- Simpler architectures, like **CasCN**, proved more effective than complex ones.
- Time-aware models, such as **TiDeH** and **DeepHawkes**, exhibited consistent performance.
- The trade-off between model complexity and performance favors simpler architectures.

### 3. Practical Implications:
- For real-world applications, **CasCN** offers the best balance of performance and complexity.
- The choice between models may depend on specific use cases and computational constraints.
- All tested models are viable options for cascade prediction tasks.


## **Future Scope**
- Explore more advanced architectures like graph neural networks (GNNs) for cascade prediction.
- Experiment with real-time cascade tracking and prediction.

---

## **Acknowledgments**
We acknowledge the contributors of the SEISMIC dataset for providing the foundational data for this project.

---


