# import the necessary packages
from sklearn.neighbors import     
from sklearn.cross_validation import train_test_split
import numpy as np
import imutils
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

print("[INFO] describing images...")
df = pd.read_csv("F:/Level 3/Semester 2/Pattern/wiki/wiki/dataset.csv")
rawImages = []
features = []
labels = [] 

for i in range(0,60000):
    image = cv2.imread('F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/'+df['full_path'][i][2:-2])  
    label = df['gender'][i]
    if(image is not None):
        if(label==0 or label ==1):
            if(df['gender'][i] == 1 ):
                label = 'male'
            else:
                label ='female'
            pixels = image_to_feature_vector(image)
            rawImages.append(pixels)
            labels.append(label)

rawImages = np.array(rawImages)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)

#when n_neighbors = 5 ==> the score = 76.37%  on 25000 image on pixels 
#when n_neighbors = 5 ==> the score = 76.80%  on 25000 image on pixels
#when n_neighbors = 5 ==> the score = 73.04%  on 25000 image on feture 
#when n_neighbors = 9 ==> the score = 77.58%  on 25000 image on pixels
#when n_neighbors = 9 ==> the score = 77.10%  on 50000 image on pixels 

#when n_neighbors = 1 ==> the score = 75%  on 60000 image on pixels 
#when n_neighbors = 2 ==> the score = 74%  on 60000 image on pixels 
#when n_neighbors = 3 ==> the score = 78%  on 60000 image on pixels 


neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
#for i, k in enumerate(neighbors):
print("[INFO] evaluating raw pixel accuracy...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(trainRI, trainRL)
  #Compute accuracy on the training set
train_accuracy[i] = knn.score(trainRI, trainRL)

#Compute accuracy on the testing set
test_accuracy[i] = knn.score(testRI, testRL)

acc = knn.score(testRI, testRL)
print("2")
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


#photo = cv2.imread('F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/19/33085219_1933-10-23_2010.jpg')
#if(photo is not None):
#    photo_pixels = image_to_feature_vector(photo)
#    predict = model.predict(photo_pixels)
#    print("[INFO] Model Predict : "+predict[0])
#else:
#    print('Error')
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
