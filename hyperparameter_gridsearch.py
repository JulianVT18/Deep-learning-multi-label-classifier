# Simpel script voor hyperparameter selection met grid search
# Via https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/


# import packages
import pandas as pandas
import numpy as np
from numpy import mean
from numpy import std
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV

# load data
data = pandas.read_csv("C:/Users/Julian/Downloads/data.csv", index_col= 0) # default header = True
labels = pandas.read_csv("C:/Users/Julian/Downloads/one_hot-labels.csv.txt", sep = ";", index_col= 0) # default header = True

#print(data)
#print(labels)
# Make data compatible for converting to tensors
data_as_array = np.asarray(data).astype('float32')
labels_as_array = np.asarray(labels).astype('float32')

#print(data_as_array)
#print(labels_as_array)


# Maak train en test set
X_train, X_test, y_train, y_test = train_test_split(data_as_array, labels_as_array, test_size=0.20, random_state=33)

# Maak class weights voor class imbalance
label_integers = np.argmax(labels_as_array, axis=1)
class_weights = compute_class_weight('balanced', np.unique(label_integers), label_integers)
d_class_weights = dict(enumerate(class_weights))
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=20531, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Hieronder is voor parameters testen
estimator = KerasClassifier(build_fn=baseline_model, verbose=0)
batch_size = [5,10,15,20]
epochs = [100,200,300,400]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid= GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=3)
grid_search_result = grid.fit(X_train,y_train)
print("Best grid search score: %f using the following parameters: %s" % (grid_search_result.best_score_,grid_search_result.best_params_))
means = grid_search_result.cv_results_['mean_test_score']
stds = grid_search_result.cv_results_['std_test_score']
params = grid_search_result.cv_results_['params']
for mean, stdev, param in zip(means,stds,params):
    print("%f (%f) with: %r" % (mean,stdev,param))
