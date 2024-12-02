import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data4.pickle', 'rb'))

del data_dict['data'][216]
del data_dict['labels'][216]
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
data_dict['labels']

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)


y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model4.p', 'wb')

pickle.dump({'model': model}, f)
f.close()
