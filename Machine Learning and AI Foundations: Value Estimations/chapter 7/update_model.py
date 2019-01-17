import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

df = pd.read_csv('ml_house_data_set_updated.csv')

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

# make a model
model = ensemble.GradientBoostingRegressor(
    n_estimators=3000,  # how many decision trees to build, increases time
    # how much each decision tree influences model. Lower rates = high accuracy, with estimator.
    learning_rate=0.1,
    max_depth=6,  # How deep each decision tree can be.
    max_features=0.1,
    min_samples_leaf=9,  # Must show up 9 times to be relevant.
    loss='huber'  # how to calculate error rate.
)
model.fit(x_train, y_train)

joblib.dump(model, 'trained_house_classifier_model.pkl')

# Test the training data.
mse = mean_absolute_error(y_train, model.predict(x_train))
# Test the .. Test data.
mse_test = mean_absolute_error(y_test, model.predict(x_test))
print(f'Training set Absolute Error: {mse}')
print(f'Testing Set Absolute Error: {mse_test}')
