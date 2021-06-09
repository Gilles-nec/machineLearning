# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Owner: Gilles nec
- Model date: 01/06/2020
- Model version: 1.0.0
- Model type: Random Forest Classifier.

## Intended Use
This is a machine learning model used to predict whether employee income exceeds $50k/yr  based on cencus data.

## Factors

- Groups: workclass, education, marital-status, occupation,
relationship, race, sex, native-country.

- Instrumentation: Cencus to jot down employees data.

## Metrics
Model performance measures: Precision, recal and their combined weight

## Evaluation Data
- Dataset: Cencus income dataset, also known as 'Adult' dataset.

- Motivation: This data set contains all the necessary information required to make accurate predictions on and employee income based to past data.

- Processing: The data was cleansed by removing all white spaces from the csv data file. Then the data is encoded an plit into train and test sets.

## Training Data
- Data: Cencus income dataset, also known as 'Adult' dataset.
- Motivation: This data set contains all the necessary information required to make accurate predictions on and employee income based to past data
- Processing:  The data was cleansed by removing all white spaces from the csv data file. Then the data is encoded an plit into train and test sets.

## Quantitative Analyses
The ratios of all data is fairly equal

## Ethical Considerations
Here a set of values such asvalues are Community, Transparency, Inclusivity, Privacy, and Topic-neutrality are used to guid work.

## Caveats and Recommendations
When doing data spliting train dataset and test dataset should always be at a ratio of 80:20 or 70:30.
