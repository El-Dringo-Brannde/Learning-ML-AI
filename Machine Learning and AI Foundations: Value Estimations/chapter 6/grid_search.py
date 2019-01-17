import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

df = pd.read_csv('ml_house_data_set.csv')

# remove useless features that we aren't interested in.
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categories with one hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove sale price from feautre data as it is our output data
del features_df['sale_price']

# create X & Y arrays for training
X = features_df.as_matrix()
y = df['sale_price'].as_matrix()

# split data by 70% training 30% test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


model = ensemble.GradientBoostingRegressor()

# Parameters to try
param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}
# Define grid search in parallel across 4 cpus.
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# Run the grid search on the training data
gs_cv.fit(x_train, y_train)

# Print the best results
print(gs_cv.best_params_)
