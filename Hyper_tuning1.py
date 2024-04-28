import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Create a pipeline with scaling and logistic regression
lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))



# Load the training dataset
train_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")

# Assuming 'target_column' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

# Define the models
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
xgb_model = xgb.XGBClassifier()
catboost_model = CatBoostClassifier()


# Fit the model
lr_model.fit(X_train, y_train)

# Define hyperparameter grids for each model
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
gb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
xgb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01, 0.001]}
catboost_param_grid = {'iterations': [100, 200, 300], 'depth': [4, 6, 8], 'learning_rate': [0.1, 0.01, 0.001]}

# Perform grid search for each model
lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='accuracy')
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy')
gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=5, scoring='accuracy')
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy')
catboost_grid_search = GridSearchCV(catboost_model, catboost_param_grid, cv=5, scoring='accuracy')

# Fit grid search objects to the data
lr_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)
gb_grid_search.fit(X_train, y_train)
xgb_grid_search.fit(X_train, y_train)
catboost_grid_search.fit(X_train, y_train)

# Get best parameters and best scores for each model
best_lr_params = lr_grid_search.best_params_
best_lr_score = lr_grid_search.best_score_

best_rf_params = rf_grid_search.best_params_
best_rf_score = rf_grid_search.best_score_

best_gb_params = gb_grid_search.best_params_
best_gb_score = gb_grid_search.best_score_

best_xgb_params = xgb_grid_search.best_params_
best_xgb_score = xgb_grid_search.best_score_

best_catboost_params = catboost_grid_search.best_params_
best_catboost_score = catboost_grid_search.best_score_

print("Best parameters and scores:")
print("Logistic Regression:", best_lr_params, best_lr_score)
print("Random Forest:", best_rf_params, best_rf_score)
print("Gradient Boosting:", best_gb_params, best_gb_score)
print("XGBoost:", best_xgb_params, best_xgb_score)
print("CatBoost:", best_catboost_params, best_catboost_score)

# Use the best parameters to train the models
lr_model.set_params(**best_lr_params)
lr_model.fit(X_train, y_train)

rf_model.set_params(**best_rf_params)
rf_model.fit(X_train, y_train)

gb_model.set_params(**best_gb_params)
gb_model.fit(X_train, y_train)

xgb_model.set_params(**best_xgb_params)
xgb_model.fit(X_train, y_train)

catboost_model.set_params(**best_catboost_params)
catboost_model.fit(X_train, y_train)

# Now, you can use these trained models for prediction on new data
