import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

X_train = np.load('X.npy')
y_train = np.load('y.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(y_train.shape)
print(y_test.shape)

X_train = np.vstack((X_train, X_test))
y_train = np.concatenate((y_train, y_test.reshape(-1)))

#X_train, X_test, y_train, y_test = train_test_split(X,y,
#    test_size = 0.2, random_state = 42)

clf = LinearSVC(C = 0.01)
#clf = SVC(C = 0.1, kernel = 'rbf')
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

joblib.dump(clf,'clf.pkl')

'''
N = y_test.shape[0]
for i in range(N):
    if (y_predict[i] != y_test[i]):
        img = X_test[i,:3072].reshape((32,32,3))
        img = cv2.resize(img,(100,100))
        s = 'actual'+str(y_test[i])+'pred'+str(y_predict[i])
        cv2.imshow(s,img)
        cv2.waitKey()
        cv2.destroyAllWindows()
'''
