# Price-Prediction-of-used-Bikes

Problem Statement:

Various companies or other sellers sell used bikes providing best resale value for the product considering various parameters. These companies’ tries to set this resale price in a descent segment, so customers are attracted to these prices.
From given Data, extract insights through various parameters of the price range of the products and predict a reasonable Resale Price of these Bikes (as regression) and predict Price Range for the same (as classification).

Solution:

This data contained data for many owners (2nd owner, 3rd …) but we only focused on 1st owner to predict the resale price for 2nd owner. Basic preprocessing is done for getting the final cleaned data for further processes and visualization is performed to check for trends in various features (outliers). New features like age of bike, distance variation with age, location categorization is performed and categorical data is encoded (i.e. insurance, seller and owner of bike).
This problem was solved with two approaches: as Regression problem and as Classification problem.
For regression, after preprocessing, the outliers in the data are studied and removed. Then this data is split into train-test set and fed through various regression models like Linear Regression, RandomForest Regressor, XGBoost Regressor and multilayer neural network. Then best model is selected for final training through Mean Absolute Percentage error as evaluation metrics.
For classification, after preprocessing, the ‘Price’ feature is grouped into various price ranges as classes. But this grouping is done differently for price as from 5k to 77k price, it was divided with 2.5k price gap and 77k to 200k, it was divided with 6k price gap. After this Price ranges are processed, it is check for imbalance and this data shows high imbalance classes so, SMOTE algorithm is used to balance these classes. Then data is split into train-test set and fed through various models like RandomForest Classifier and XGBoost Classifier. And then best algorithm is selected for training with confusion matrix and accuracy as evaluation metrics for the same. Hyper-parameter tuning for each algorithm is done using GridSearchCV, to get best parameters for each models.
