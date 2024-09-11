#import
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# Convert image data to PyTorch tensor
X = torch.tensor(lfw_people.data, dtype=torch.float32)
y = torch.tensor(lfw_people.target, dtype=torch.int64)
n_features = X.shape[1]
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Print data set information
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_components = 150
#Center data
mean = torch.mean(X_train, dim=0)
X_train -= mean
X_test -= mean

# Use PyTorch for SVD decomposition to calculate PCA
U, S, V = torch.svd(X_train, some=True)
# Assuming `V` is a PyTorch tensor and `n_components`, `h`, and `w` are defined


components = V[:, :n_components].T
#Eigen-decomposition
eigenfaces = components.reshape((n_components, h, w))
#project into PCA subspace
X_transformed = torch.mm(X_train, components.t())
print(X_transformed.shape)
X_test_transformed = torch.mm(X_test, components.t())
print(X_test_transformed.shape)

# Qualitative evaluation of the predictions using matplotlib  vbb vgfv
import matplotlib.pyplot as plt
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

#Calculate the cumulative interpretation variance ratio
explained_variance = (S ** 2) / (n_samples - 1)
total_var = torch.sum(explained_variance)
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0).numpy()

eigenvalueCount = torch.arange(n_components).numpy()
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()
from sklearn.ensemble import RandomForestClassifier
#The random forest classifier is used for classification
X_transformed = X_transformed.numpy()
X_test_transformed = X_test_transformed.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()


estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train)

predictions = estimator.predict(X_test_transformed)

# Print classification results
print(classification_report(y_test, predictions, target_names=target_names))
