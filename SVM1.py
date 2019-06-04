import numpy as np
#import pandas as pd
import pickle
#import cv2
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import math

p = open('np_final_version.pickle', 'rb')
np_final_v = pickle.load(p)
p.close()

p = open('feature_vector.pickle', 'rb')
feature_vector = pickle.load(p)
p.close()

#p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/image_path.pickle', 'rb')
#image_path = pickle.load(p)
#p.close()
    
#p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/corrupted_image.pickle', 'rb')
#corrupted_image = pickle.load(p)
#p.close()

#p = open('SVM_3_Linear.pickle', 'rb')
#s = pickle.load(p)
#p.close()



#print(len(corrupted_image))
print(len(feature_vector))
feature_vector = np.array(feature_vector)
np_final_v = np_final_v[:, 3]


l = np_final_v.tolist()
# print(l)
for i in range(0,len(l)):
    if math.isnan(l[i]):
        l[i] = float(-1)

np_final_v = np.array(l)

X_train, X_test, y_train, y_test = train_test_split(feature_vector, np_final_v, test_size=0.2)
print(X_test.shape)
print(y_test.shape)
print(X_train.shape, y_train.shape)

clf = SVC(C=3, gamma='auto', kernel='linear')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy*100)
