##################### Imports ########################
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

#################### Functions #######################
def extract_sort_coeff(pipe, month_order):
    coefficients = pipe.named_steps['regressor'].coef_[0]
    intercept = pipe.named_steps['regressor'].intercept_
    #Baseline level is april so we put it back in the arrays to make plot easier
    coefficients = np.insert(coefficients, 3, intercept)
    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
    feature_names = np.insert(feature_names, 3, 'cat__mnth_Apr')
    custom_feature_names = [name.replace("cat__", "").replace("num__", "") for name in feature_names]
    
    coeff_dict = dict(zip(custom_feature_names, coefficients))
    month_coeffs = {name: coeff_dict[name] for name in custom_feature_names if "mnth" in name}
    other_coeffs = {name: coeff_dict[name] for name in custom_feature_names if "mnth" not in name}
    
    # Sort month coefficients based on correct month order
    sorted_month_coeffs = dict(sorted(month_coeffs.items(), key=lambda x: month_order.index(x[0].split("_")[-1])))
    
    # Merge sorted months with other coefficients
    sorted_coeff_dict = {**sorted_month_coeffs, **other_coeffs}
    
    # Convert to numpy array 
    sorted_coefficients = np.array(list(sorted_coeff_dict.values()))
    
    return sorted_coefficients
########################################################
#%%

################  Loading Data #########################
df = pl.read_csv('Bikeshare.csv')
features = ['mnth', 'hr', 'workingday', 'weathersit']
categorical_features = ['mnth','weathersit']
numerical_features = ['hr','workingday']
y = df.select(['bikers']).to_pandas()
X = df[features]
Xpandas = X.to_pandas()
#########################################################


################ Preprocessing ###########################
#Problem for LinReg : categorical variables => one-hot encoding
#Preprocessing dorp first value and the 1/0 for each month 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features),
        ('num', 'passthrough', numerical_features)  # Keep numerical columns unchanged
    ]
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  
])
##########################################################


################# Results and plots ######################
X_train, X_test, y_train, y_test = train_test_split(Xpandas, y, test_size=0.2)
reg = pipe.fit(X_train, y_train)
month_order = ['Jan', 'Feb', 'March', 'Apr', 'May', 'June','July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
sorted_coefficients = extract_sort_coeff(pipe, month_order)
plt.bar(month_order, sorted_coefficients[:12], color='skyblue', edgecolor='black')
plt.show()
plt.scatter(df['hr'].to_numpy(), df['bikers'].to_numpy())
plt.show()

predictions = []
for hour in range(24):
    temp_df = X_test.copy()
    temp_df['hr'] = hour  # Set the hr to the current hour
    print(temp_df)
    pred = pipe.predict(temp_df)
    predictions.append(pred.mean())
    
worst_x = pl.DataFrame({'mnth' :  'Jan', 'hr': 4, 'workingday' : 0, 'weathersit' : 'clear'})
print(worst_x)
print(pipe.predict(worst_x.to_pandas()))
#problem : negative value
#%%
pipe2 = pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor(max_iter = 1000))  
])

pipe2.fit(X_train, y_train)
sorted_coefficients = extract_sort_coeff(pipe, month_order)
plt.bar(month_order, sorted_coefficients[:12], color='skyblue', edgecolor='black')
plt.show()
plt.scatter(df['hr'].to_numpy(), df['bikers'].to_numpy())
plt.show()
