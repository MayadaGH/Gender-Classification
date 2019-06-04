import scipy.io
import numpy as np
import pandas as pd
import pickle
import face_recognition

# training_data_labels = scipy.io.loadmat('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/wiki.mat')
# train_data_matrix = training_data_labels['wiki']
# x = train_data_matrix[0][0]
# # print(x)
# x_final = []
# for i in x:
#     x_final.append(i[0])

# x_final_image_path = x_final[2]
# k=0
# for i in x_final_image_path:
#     x_final_image_path[k] = i[0]
#     k+=1

# x_final_image_path = x_final[4]
# k=0
# # print(x_final_image_path[418])
# for i in x_final_image_path:
#     # print(i[0], k)
#     if len(x_final_image_path[k]):
#         x_final_image_path[k] = i[0]
#     else:
#         x_final_image_path[k] = '?'
#     k+=1

# x_final_image_path = x_final[5]
# k=0
# for i in x_final_image_path:
#     # print(i[0], k)
#     if len(x_final_image_path[k]):
#         x_final_image_path[k] = x_final_image_path[k][0]
#     # else:
#     #     x_final_image_path[k] = float()
#     k+=1

# print(x_final)
# np_final_version = np.array(x_final)
# print(np_final_version)
path = '/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/wiki_crop'
p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/np_final_version.pickle', 'rb')
np_final_version = pickle.load(p)
p.close()
# sample = np_final_version[140, 2]
# print(sample)
# exit(0)
# print(np_final_version)
# p.close()
# np_final_version = np_final_version.transpose()
# print(np_final_version[:, 2])
feature_vector = []
corrupted_image = []
image_path = []

for image_name in np_final_version[:, 2]:
    img_path = '/'.join([path, image_name])
    # face detection
    X_img = face_recognition.load_image_file(img_path)
    X_faces_loc = face_recognition.face_locations(X_img)
    # print('Image path: {} | X_faces_loc: {}'.format(image_name, X_faces_loc))
    print('Image path: {}'.format(image_name))
    # if the number of faces detected in a image is not 1, ignore the image
    if len(X_faces_loc) != 1:
        print('There is Corrupted image: {}'.format(image_name))
        corrupted_image.append(image_name)
        face_encoding = [float(-1) for _ in range(0,128)]
    else:
        # extract 128 dimensional face features
        faces_encoding = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)[0]
    feature_vector.append(faces_encoding)
    image_path.append(image_name)

p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/feature_vector.pickle', 'wb')
pickle.dump(feature_vector, p)
p.close()
p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/image_path.pickle', 'wb')
pickle.dump(image_path, p)
p.close()
p = open('/home/youssef/Downloads/Machine_Learning/Support_Vector_Machine/gender_classification/Feature_Vector_Final_CSV/corrupted_image.pickle', 'wb')
pickle.dump(corrupted_image, p)
p.close()
# print(feature_vector)
# print(len(feature_vector[0]))
# feature_vector = np.array([np.array(x) for x in feature_vector])
# print(feature_vector.shape)
# print(type(feature_vector))
# print(type(feature_vector[0]))
# np.reshape(feature_vector, (1896, 1))
# print(feature_vector.shape)
# print(type(np_final_version[:, 5][0]))
# print(feature_vector)
# feature_vector = np.array(feature_vector)
# print(np_final_version[0, :])
# print(np_final_version.shape)
# print(feature_vector[1, :])
# print(feature_vector.shape)
# print(np_final_version[0:3000, :].shape)
# print(np_final_version[0, :])
# concatenated = np.append(np_final_version[0:3000, :], feature_vector)
# print(concatenated[0])
# cl = ['DOB', 'photo_taken' ,'full_path', 'gender', 'name', 'face_location', 'face_score', 'second_face_score']
# df = pd.DataFrame(np_final_version, columns=cl)
# feature_vector = np.array(feature_vector)
# df['feature_vector'] = np.concatenate([feature_vector, np.zeros(3000 - 1896)])
# print(df.head())

# print(feature_vector)
# print(x_final_image_path)
# print(x_final[2])
# ed = np.array(x)
# print(ed[0, :])
# x_1 = x[2][0][0][0] # Image_path The actual string
# x_3 = x[2][0] # Image_path
# x_2 = x[1]  # Photo taken
# print(len(x_3))
# # x_3 = x
# # #2, 3, 4, 5, 6, 7
# training_new_matrix = []
# # k=0
# for i in range(0,8):
#     training_new_matrix.append(x[i][0])

# df = pd.read_csv('final_version.csv')
# # print(df.info)
# print(df['face_location'].head())
# # print(df.head())
# exit(0)

# # print(training_new_matrix[1])
# ed = np.array(training_new_matrix)
# # # print(ed[0, :])
# ed = ed.transpose()
# # print(ed.shape)
# cl = ['DOB', 'photo_taken' ,'full_path', 'gender', 'name', 'face_location', 'face_score', 'second_face_score']
# # id = [i for i in range(0, ed.shape[0])]
# df = pd.DataFrame(ed, columns=cl)
# df.drop(columns=['DOB', 'photo_taken'], inplace=True)
# # # df.index('id')
# # # print(pd.isnull(df).sum)
# df.fillna(float('-inf'), inplace=True)
# df.to_csv('final_version.csv')
# # print(df.head())
# # # print(ed[0, :])
# # #  = np.array([x_1])
# # # print(len(x))
# # # print(type(x))
# file = open('test2.txt', 'w')
# file.write(np.array2string(np.array(x_final)))
# file.close()
# file = open('test.txt', 'w')
# file.write(np.array2string(x))
# file.close()
# # # print(train_data_matrix.shape)
# # # df = pd.DataFrame(train_data_matrix, index=labels)
