
#import relevant libraries: pandas, numpy, matlab
import pandas as pd

#import model frameworks
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#to break up data for validation
from sklearn.model_selection import train_test_split

#import file by extracting location of file
file_path = ../Documents/michelle/coding/data-science/kaggle/practice/train.csv

#store data as object in code
data = pd.read_csv(file_path)

#get to know data more
data.columns

#store target variable as y
y = data.target

#store (relevant) features in feature matrix -- ones
#that make sense from looking at the data
features = ['feature1', 'feature2', 'feature3']

#make corresponding feature matrix
X = data[features]

#check X (this is important !!)
#X.describe()

#if features make sense, split and initialize model
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)
model = DecisionTreeRegressor(random_state=1)

#fit
model.fit(train_X,train_y)

#predict and set matrix as variable
val_predictions = model.predict(val_X)

#if you wanna check
#print(predictions)

#model validation
val_mae = mean_absolute_error(val_y, val_predictions)

#FYI get_mae function to implement to find optimal amount of nodes
def get_mae(max_leaf nodes, train_X, val_X, train_y, val_y):
	test_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
	test_model.fit(train_X, train_y)
	predictions = test_model.predict(val_X)
	mae = mean_absolute_error(val_y, predictions)
	return(mae)















