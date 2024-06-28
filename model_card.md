# Model Card

Original paper introducing the concept of model cards can be found here: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details

This model is a Random Forest using the default hyperparameters in scikit-learn version 0.24.2.

## Intended Use

This model is designed to predict whether individuals earn more than $50,000 or less than $50,000 per year based on a set of attributes.

## Training Data

The training data consists of 80% of the original dataset. The target variable is the salary, categorized into two groups: salaries over $50,000 and salaries below $50,000. The training data was one-hot encoded and label binarized.

## Evaluation Data

The evaluation data, comprising the remaining 20% of the original dataset, was processed similarly to the training data with one-hot encoding but without label binarization.

## Metrics

The model's performance was assessed using the F1 score, precision, and recall. The precision was 0.7454, recall was 0.615, and the F1 score was 0.6739.

## Ethical Considerations

The dataset includes information on race and gender, which could potentially lead to discrimination. Therefore, a more thorough examination of these aspects is necessary.

## Caveats and Recommendations

Since some countries are more represented in the data than others, further efforts are needed to include data from underrepresented countries to create a fairer model.