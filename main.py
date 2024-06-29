import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os,cv2

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
dir_name = "dataset/faces/"
y=[]; x=[]; target_names=[]
person_id=0; h=w=300
n_samples=0
class_names=[]
for person_name in os.listdir(dir_name): 
    dir_path = dir_name + person_name + "/"
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        image_path = dir_path + image_name
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray,(h,w))
        #convert matrix to vector
        v = resized_image.flatten()
        x.append(v)
        n_samples = n_samples+1
        y.append(person_id)
        target_names.append(person_name)
    person_id=person_id+1


#transform lists to np arrays
y=np.array(y)
x=np.array(x)
target_names = np.array(target_names)
n_features = x.shape[1]
print(y.shape,x.shape,target_names.shape)
print("no of samples:", n_samples)
print(n_features)

#split into training and test set using stratified k fold
x_train, y_train,x_test, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
n_components = 150
#applying PCA

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" %i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenface_titles,h,w)
plt.show()
 
print("Projecting input data on the eigenfaces orthonormal basis")
x_train_pca = pca.transform(x_train)

x_test_reshaped = np.zeros((337,90000))
x_test_reshaped[:, :337] = x_test
x_test_reshaped.shape
x_test_pca = pca.transform(x_test_reshaped)

#computing fisherfaces
lda = LinearDiscriminantAnalysis()











