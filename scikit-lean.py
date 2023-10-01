from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Caricare il dataset
iris = load_iris()

# Creare array di caratteristiche e etichette
X, y = iris.data, iris.target

# Suddividere il dataset in un set di addestramento e un set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creare un oggetto StandardScaler e adattarlo ai dati di addestramento
scaler = StandardScaler()
scaler.fit(X_train)

# Applicare la trasformazione sia ai dati di addestramento sia a quelli di test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creare un classificatore KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Addestrare il classificatore
knn.fit(X_train, y_train)

# Prevedere le etichette per il set di test
y_pred = knn.predict(X_test)

# Stampare le metriche di valutazione
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
