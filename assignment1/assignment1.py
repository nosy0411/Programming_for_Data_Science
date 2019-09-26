# 1.2 Data load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()

print("Target names:", iris_dataset['target_names'])
print("Feature names:\n", iris_dataset['feature_names'])
print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)
print("=============================")
print("Type of target:", type(iris_dataset['target']))
print("Shape of target:", iris_dataset['target'].shape)

# 1.3 Data preprocessing

X_train_and_valid, X_test, y_train_and_valid, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], test_size=0.2, random_state=0)
print("X_train_and_valid shape:", X_train_and_valid.shape)
print("y_train_and_valid shape:", y_train_and_valid.shape)
print("=============================")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_and_valid, y_train_and_valid, test_size=0.2, random_state=0)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("=============================")
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scale=scaler.transform(X_train)
X_valid_scale=scaler.transform(X_valid)
X_test_scale=scaler.transform(X_test)

# 1.4 KNN and select hyperparameter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

neighbors_settings = list(range(1, 31))
p_settings = list(range(1,6))
data_dict=dict()

def weight_func(weight):
    p_dict=dict()
    for p in p_settings:
        training_accuracy=[]
        valid_accuracy=[]
        for n_neighbors in neighbors_settings:
            # build the model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=p, weights=weight)
            knn.fit(X_train_scale, y_train)

            y_train_hat = knn.predict(X_train_scale)
            training_accuracy.append(accuracy_score(y_train, y_train_hat))

            y_valid_hat = knn.predict(X_valid_scale)
            valid_accuracy.append(accuracy_score(y_valid, y_valid_hat))
        p_dict[p]=dict(training=training_accuracy, valid=valid_accuracy)
        data_dict[weight]=p_dict

weight_func('uniform')
weight_func('distance')
                                  
# print(data_dict)

print(data_dict['uniform'][1]['training'][1])

import pandas as pd

data=dict()

for p in p_settings:
    for n_neighbors in neighbors_settings:
        if data.get('weights'):
            data['weights'].append('uniform')
            data['metric'].append(p)
            data['k'].append(n_neighbors)
            data['training_accuracy'].append(data_dict['uniform'][p]['training'][n_neighbors-1])
            data['valid_accuracy'].append(data_dict['uniform'][p]['valid'][n_neighbors-1])
        else:
            data['weights']=['uniform']
            data['metric']=[p]
            data['k']=[n_neighbors]
            data['training_accuracy']=[data_dict['uniform'][p]['training'][n_neighbors-1]]
            data['valid_accuracy']=[data_dict['uniform'][p]['valid'][n_neighbors-1]]

for p in p_settings:
    for n_neighbors in neighbors_settings:
            data['weights'].append('distance')
            data['metric'].append(p)
            data['k'].append(n_neighbors)
            data['training_accuracy'].append(data_dict['distance'][p]['training'][n_neighbors-1])
            data['valid_accuracy'].append(data_dict['distance'][p]['valid'][n_neighbors-1])

df=pd.DataFrame(data, index=list(range(0,len(neighbors_settings)*len(p_settings)*2)), columns=['weights','metric','k','training_accuracy','valid_accuracy'])
print(df)

m=df['valid_accuracy'].max()
max_index=[i for i, j in enumerate(list(df['valid_accuracy'])) if j == m]
print(max_index, m)
print(df.loc[max_index])

# 2. Conclusion

knn = KNeighborsClassifier(n_neighbors=22, metric='minkowski', p=3, weights='uniform')
knn.fit(X_train_scale,y_train)
y_test_hat = knn.predict(X_test_scale)
print("test accuracy : ", accuracy_score(y_test, y_test_hat))