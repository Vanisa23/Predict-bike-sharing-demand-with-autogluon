# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### VANISA KORINGURA

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

When attempting to submit the predictions, I realized that the output of the predictor needed to be in a specific format in order to be submitted to the Kaggle competition. Initially, the predictor generated predictions in the form of a pandas DataFrame, typically with columns for the target variable (in this case, "count") and other relevant features. However, Kaggle requires submissions to be in a specific CSV format, with two columns: one for the unique identifier of each instance (in this case, "datetime") and another for the predicted value ("count"). Therefore, changes were needed to transform the predictor's output DataFrame into this required format before it could be submitted to Kaggle. This involved extracting the "datetime" column and the corresponding predicted values, and then saving them to a CSV file. Once this transformation was applied, the predictions could be successfully submitted to the Kaggle competition for evaluation.

### What was the top ranked model that performed?

The top-ranked model that performed is the WeightedEnsemble_L3. It achieved the lowest validation score of -51.218735. 

Additional notes 

The weighted ensemble model is often a combination of multiple base models, and it's designed to improve overall predictive performance by aggregating predictions from these base models. Additionally, it has a stack_level of 3, indicating that it is a higher-level model in the ensemble stacking framework.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

I HAVE WRITTEN ALL THE SUMMARY OF THE EDA ON THE NOTEBOOK FOR EASY READING AND ACCESS. Thank you!

### How much better did your model preform after adding additional features and why do you think that is?

After adding additional features, such as the hour, season, and weather categories, the model performance improved significantly. The addition of these features provided the model with more relevant information about the underlying patterns in the data, allowing it to make more accurate predictions.

The hour feature, for example, captures the time of day when bike rentals occur, which helps the model account for variations in demand throughout the day. By including this feature, the model can better capture the peak hours of rental activity, resulting in more accurate predictions.

Similarly, converting the season and weather features into categorical variables ensures that the model understands the categorical nature of these attributes and does not treat them as continuous variables. This prevents the model from making erroneous assumptions about the ordinal relationships between different seasons or weather conditions, leading to more accurate predictions.


## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?

After tuning different hyperparameters, the model's performance improved further compared to the initial training. By fine-tuning the hyperparameters, we can optimize the model's configuration to better fit the training data and generalize well to unseen data.

Hyperparameter tuning involves searching for the best combination of hyperparameters that minimizes the model's validation error. This process can be computationally expensive, but it can lead to significant improvements in predictive performance.

In our case, after hyperparameter tuning, we observed a reduction in the validation error, indicating that the model's ability to generalize to unseen data improved. This suggests that the tuned hyperparameters allowed the model to better capture the underlying patterns in the data and make more accurate predictions.

### If you were given more time with this dataset, where do you think you would spend more time?

Feature Engineering - Continuously exploring and creating new features could potentially enhance the model's ability to capture complex patterns in the data. This could involve extracting additional information from existing features or creating entirely new features based on domain knowledge.

Hyperparameter Tuning - While we performed hyperparameter tuning in this project, there might be additional hyperparameters or combinations that could further improve the model's performance. Exploring more hyperparameter space with advanced optimization techniques like Bayesian optimization or genetic algorithms could be beneficial.

Model Selection - Experimenting with different machine learning algorithms and ensemble methods could lead to better performance. Trying out models that were not included in the initial training phase or combining existing models in novel ways through stacking or blending could yield improvements.

Cross-Validation and Evaluation - Utilizing more sophisticated cross-validation strategies, such as nested cross-validation, could provide a more robust estimate of the model's performance and help prevent overfitting. Additionally, exploring alternative evaluation metrics that are more suitable for the specific characteristics of the dataset could offer additional insights.

Error Analysis and Interpretability - Conducting thorough error analysis to understand the model's weaknesses and biases could guide further improvements. Techniques such as partial dependence plots, SHAP (SHapley Additive exPlanations) values, and feature importance analysis could help interpret the model's predictions and identify areas for refinement.

Data Cleaning and Preprocessing - Spending more time on data cleaning and preprocessing could lead to cleaner and more informative datasets, which in turn could improve model performance. This could involve handling missing values more effectively, detecting and correcting outliers, and exploring different scaling or transformation techniques.


### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|prescribed_values|prescribed_values|presets: 'high quality' (auto_stack=True)|1.81113|
|add_features|prescribed_values|prescribed_values|presets: 'high quality' (auto_stack=True)|0.71771|
|hpo|Tree-Based Models: (GBM, XT, XGB & RF)|KNN|presets: 'optimize_for_deployment|0.47758|

Initial Model:

Hyperparameters Modified: hpo1, hpo2, hpo3
Explanation: The initial model likely used default or predetermined values for the hyperparameters hpo1 and hpo2, while hpo3 was set to 'high quality' presets with auto_stack enabled.
Kaggle Score: 1.81113
Impact of this change
The high Kaggle score suggests that the preset 'high quality' settings for hpo3, particularly with auto_stack enabled, might have contributed significantly to the model's performance. It's possible that these settings improved model accuracy or generalization ability.

Add Features Model:

Hyperparameters Modified: hpo1, hpo2, hpo3
Explanation: Similar to the initial model, this iteration also used prescribed values for hpo1 and hpo2, with hpo3 set to 'high quality' presets with auto_stack enabled.
Kaggle Score: 0.71771
Impact of Change 
Despite using the same hyperparameter settings as the initial model, the addition of features led to a significant drop in the Kaggle score. This suggests that while the original hyperparameter configuration might have been effective for the initial dataset, it might not generalize well to the expanded feature set. Further analysis could reveal whether specific hyperparameters need to be adjusted to accommodate the new features.

HPO Model:

Hyperparameters Modified: hpo1, hpo2, hpo3
Explanation: In this iteration, the hyperparameters were adjusted to use tree-based models (GBM, XT, XGB & RF) for hpo1, KNN for hpo2, and 'optimize_for_deployment' presets for hpo3.
Kaggle Score: 0.47758
Impact of Changes
The substantial decrease in Kaggle score compared to the initial model suggests that the chosen hyperparameter configurations might not have been well-suited for the dataset or the problem at hand. The switch to tree-based models and KNN for hpo1 and hpo2, respectively, could have introduced biases or limitations that affected model performance negatively.

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
my best ranking model was WeightedEnsemble_L3 with validation RMSE score of 32.501212 and a kaggle score of 0.47758
