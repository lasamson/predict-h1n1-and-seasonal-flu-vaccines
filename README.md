## Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

### Problem Statement

In this project, our aim is to predict whether individuals received H1N1 and seasonal flu vaccines based on their social, economic, and demographic background, as well as their opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission.

Vaccination is a key public health measure used to fight infectious diseases by providing immunization for individuals and mitigating the  further spread of diseases in communities through herd immunity.

Original task is described in the [DrivenData competition](https://www.drivendata.org/competitions/66/flu-shot-learning/).

### Data

In the original challenge, we are provided with three CSV files:
- training_set_features.csv
- training_set_labels.csv
- test_set_features.csv

Each row in the above datasets represents an individual who was asked a set of survey questions regarding their backgrounds, opinions, and health behaviors. The files `training_set_features.csv` and `test_set_features.csv` consist of 35 categorical features representing responses to the various questions asked, along with a `respondent_id` column. The file `training_set_labels.csv` consist of two columns corresponding to the two target variables `h1n1_vaccine` and `seasonal_vaccine` that we are interested in predicting, which represents whether the respondent received the h1n1 and seasonal flu vaccines, respectively.

Since the challenge does not provide us with the test set labels, we treat the provided training set as the full data set. Thus, we split the provided training set into training and test sets to train and validate our model, as well to report the final out-of-sample performance on the test set.

### Approach to Dealing with Multi-Label Classification

Since there are two targets of interest we want to predict, this is a multi-label classification problem, where each label is binary. Thus, there are two possible approaches:

1. Predict the two labels jointly; that is, develop one model that predicts combinations of the labels. Since each target varible is binary, this will result in 4 possible outcomes, and will make this a multi-class classification problem.
2. Predict the two labels independently; that is, develop two independent models that each predict a single target. This involves training two binary classifiers.

The advantage of the first method is that the model can exploit any potential relationships between the two target variables to inform the prediction. Intuitively, we might believe that an individual is much more likely to get a second vaccine if they got the first. Thus, there is very likely some dependence betweent the labels that could improve our predictive performance. Another advantage of this method is that we only have to train a single model, rather than two separate models.

However, not all of the algorithms that we want to try for this problem (e.g. `LogisticRegression`) have multilabel support from Scikit-Learn, so we opt for the second option.

### Data Processing, Feature Importance Analysis and Feature Selection

The feature matrix was initially assessed for missing values, and it was discovered the following features had close to 50% of values missing:
- `health_insurance`
- `employment_industry`
- `employment_occupation`

Since there is no feasible way to fill in such a large quantity of missing data, we dropped these features completely from the dataset.

Additionally, all features are categorical, and were therefore, transformed into one-hot representations. This transforms the final 32 features into 107 binary features.

Two methods were employed for feature importance analysis:

1. Computing mutual information between features (prior to one-hot encoding) and the target variable of interest.
2. Training a small `RandomForestClassifier` model on the data and analyzing the feature importances from `feature_importances_`. 

However, in the end, we chose to include all features, as selecting a smaller set of features did not improve performance significantly for any of the models. In fact, including the full set of features improved performance for several models.

### Evaluation Metric

The `seasonal_vaccine` variable is fairly balanced between positive and negative examples, but the `h1n1_vaccine` variable is quite imbalanced, with approximately an 80:20 ratio of negative to positive examples. Thus, we need a metric that takes the performance on both labels into account. The area under the receiver operating characteristic (ROC) curve (AUC) is a good measure of performance on the positive and negative examples, as it considers both the true positive and false positive rates.

In order to get a single metric for the overall performance for the multi-label task, we take the mean AUC for both labels. This is implemented by using `sklearn.metrics.roc_auc_score` with the setting `average='macro'`. This is also the metric used in the original challenge.

### Final Model

We tried four classification algorithms for both problems: logistic regression, decision tree classifier, random forest classifier, and XGBoost classifier.

Hyperparameter tuning (via a randomized search over parameter grid) revealed the XGBoost classifier to be the best performing model for both the h1n1 and seasonal flu vaccine prediction tasks.

The final model achieved a mean AUC of `0.85` on the held-out test set. Below are the ROC curves for the h1n1 and seasonal flu vaccine XGBoost classifiers, respectively:

![h1n1-roc](img/roc_curve_final_h1n1_cls.png)

![seasonal-roc](img/roc_curve_final_seasonal_flu_cls.png)