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

During the exploratory data analysis (EDA), several key insights were discovered:

Seasonal Trends: There were clear seasonal trends in bike rental demand, with higher demand during certain months and lower demand during others. This suggests that the season feature could be an important predictor of bike rental counts.

Weather Impact: Weather conditions such as temperature, humidity, and windspeed appeared to influence bike rental demand. Warmer temperatures and lower humidity were associated with higher rental counts.

Time of Day: There was a noticeable variation in rental counts based on the time of day. Peaks in rental demand were observed during commuting hours, suggesting that the hour feature could be informative.

Based on these insights, additional features were created:

Hour Feature: Extracted the hour component from the datetime feature to capture the time of day when rentals occurred.

Season and Weather Categories: Converted the season and weather features into categorical variables to ensure they are treated as categories rather than numerical values by the models. This step helps models recognize the inherent categories in these features and prevents them from interpreting them as ordinal variables.

These additional features are expected to enhance the predictive power of the models by providing them with more information about the underlying patterns in the data

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
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|


### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
