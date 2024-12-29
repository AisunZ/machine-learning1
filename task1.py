# Machine learning
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.svm import SVR  
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from sklearn.preprocessing import StandardScaler , OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  



# Prepair data
dataset=pd.read_csv('housePrice.csv')

# Option 1: Drop missing values  
data = dataset.dropna()  

# Replace the strings with commas with their corresponding float values  
data['Area'] = data['Area'].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) and ',' in x else float(x))   
data['Area'] = data['Area'].apply(lambda x: x / (1e8) if isinstance(x, float) and x > 1e8 else x)  


# Identify numerical and categorical columns  
numerical_features = ['Area', 'Room','Price','Price(USD)']   
categorical_features = ['Address'] 
boolean_features = ['Parking', 'Warehouse','Elevator']    
 

 

preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', StandardScaler(), numerical_features),  
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  
        ('bool', 'passthrough', boolean_features)  # Pass boolean features through without transformation  
    ]  
)  


# Exploratory Data Analysis (EDA)  
# Visualize distribution of Price(USD) 
plt.figure(1)
sns.histplot(data['Price(USD)'], bins=10, kde=True)  
plt.title('Price(USD) Distribution')  
plt.xlabel('Price(USD)')  
plt.ylabel('Frequency')  
plt.show()  


# Analyze feature correlations  
correlation_matrix = data.corr()  
plt.figure(2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  
plt.title('Feature Correlation Matrix')  
plt.show()  

# Identify and visualize outliers 
plt.figure(3) 
sns.boxplot(x=data['Price(USD)'])  
plt.title('Price(USD) Box Plot')  
plt.show() 



# Feature selection  
features = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']  
target = 'Price(USD)'  

X = data[features]  
y = data[target]  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  


# Model Building: Using Random Forest as an example  
model = Pipeline(steps=[('preprocessor', preprocessor),  
                                      ('regressor', RandomForestRegressor)]) 

# Train the model on the training set  
model.fit(X_train, y_train)  

# Model Evaluation  
y_pred = model.predict(X_test)  




# # Calculate metrics  
# rmse = np.sqrt(np.sum((i - j) ** 2 for i, j in zip(y_test, y_pred)) / len(y_test))
# mae = mean_absolute_error(y_test, y_pred)  

# print(f'Root Mean Squared Error: {rmse}')  
# print(f'Mean Absolute Error: {mae}')  

# # # Perform hyperparameter tuning using GridSearchCV  
# # param_grid = {  
# #     'regressor__n_estimators': [50, 100],  
# #     'regressor__max_features': ['auto', 'sqrt'],  
# # }  

# # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')  
# # grid_search.fit(X_train, y_train)  

# # print(f'Best Parameters: {grid_search.best_params_}')  
# # best_model = grid_search.best_estimator_  



# # # Final evaluation of the best model  
# # y_pred_best = best_model.predict(X_test)  
# # rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))  
# # mae_best = mean_absolute_error(y_test, y_pred_best)  

# # print(f'Best Model RMSE: {rmse_best}')  
# # print(f'Best Model MAE: {mae_best}')  

# # # Model Interpretation: Feature Importance  
# # importances = best_model.named_steps['regressor'].feature_importances_  
# # feature_names = (numerical_features +  
# #                  list(best_model.named_steps['preprocessor']  
# #                            .transformers_[1][1].get_feature_names_out(categorical_features)))  

# # feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})  
# # feature_importance = feature_importance.sort_values(by='Importance', ascending=False)  

# # # Plot feature importance  
# # sns.barplot(data=feature_importance, x='Importance', y='Feature')  
# # plt.title('Feature Importance')  
# # plt.show()  
