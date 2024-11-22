# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The classification model developed in this project uses the Support Vector Machine (SVM) model and the scikit-learn library. The parameters are set to default.

## Intended Use
This model is designed to classify records in the U.S. Census Bureau's demographic data. The goal is to predict a person's income based on demographic characteristics such as gender, education level, occupation, marital status, etc.

## Training Data
The model uses publicly available data from the U.S. Census Bureau, which includes information on age, gender, income, education level, and other demographic factors. 80% of the dataset is used for training the SVM model.

## Evaluation Data
The evaluation data is also taken from the publicly available U.S. Census Bureau dataset, comprising 20% of the entire dataset, but is separate from the training data. The purpose is to test the model's performance on unseen data, helping to assess its accuracy and generalizability.

## Metrics
Evaluation on the test set, the model achieved the following metrics:
- precision: 0.8697318007662835
- recall: 0.1433080808080808
- fbeta: 0.246070460704607

## Ethical Considerations
The purpose of this study is not to differentiate or draw conclusions about an individual's income. The model should not be used for discriminatory practices between different socio-cultural groups. This study is solely for the purpose of practicing coding principles, training machine learning models, and utilizing CI/CD in software engineering.

## Caveats and Recommendations
The SVM model needs improvement, as the recall and Fbeta scores are very low.
