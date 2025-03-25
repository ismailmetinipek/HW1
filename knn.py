import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Öklidyen ve Manhattan Mesafe Fonksiyonları ---
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# --- k-NN Algoritması ---
class KNNClassifier:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for i in range(len(self.X_train)):
            if self.distance_metric == "euclidean":
                dist = euclidean_distance(x, self.X_train[i])
            else:
                dist = manhattan_distance(x, self.X_train[i])
            distances.append((dist, self.y_train.iloc[i]))

        distances = sorted(distances)[:self.k]
        k_neighbors = [label for _, label in distances]

        return Counter(k_neighbors).most_common(1)[0][0]

# --- Veriyi Yükleme ve İşleme ---
df = pd.read_csv("wine.data", names=[
    "Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash",
    "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",
    "Proanthocyanins", "Color_intensity", "Hue", "OD280/OD315", "Proline"
])

X = df.drop(columns=["Class"])
y = df["Class"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Modeli Eğitme ve Test Etme ---
knn = KNNClassifier(k=3, distance_metric="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# --- Sonuçları Yazdırma ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
