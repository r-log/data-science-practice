import pandas as pd
from sklearn.tree import DecisionTreeRegressor

mb_file_path = 'path/to/your/dataset.csv'
#READING THE DATA
mb_data = pd.read_csv(mb_file_path)
mb_data_drop = mb_data.dropna(axis=0)
mb_model = DecisionTreeRegressor(random_state=1)

#VISUALIZING DATA CLEANER
pd.set_option('display.max_rows', 5) 
pd.set_option('display.max_columns', None)    
pd.set_option('display.width', 1000)          
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3) 

mb_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']


y = mb_data.Price
X = mb_data[mb_features]

mb_model.fit(X,y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(mb_model.predict(X.head()))