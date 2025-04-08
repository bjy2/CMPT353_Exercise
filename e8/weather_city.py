import pandas as pd
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


labelled_data = pd.read_csv(sys.argv[1])
unlabelled_data = pd.read_csv(sys.argv[2])

X_labelled = labelled_data.iloc[:, 2:].values
y_labelled = labelled_data.iloc[:, 0].values

X_train, X_valid, y_train, y_valid = train_test_split(X_labelled, y_labelled, test_size=0.3)

"""gnb_model = make_pipeline(StandardScaler(), GaussianNB())
gnb_model.fit(X_train, y_train)
gnb_score = gnb_model.score(X_valid, y_valid)"""

knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
knn_model.fit(X_train, y_train)
knn_score = knn_model.score(X_train, y_train)

"""rf_model = make_pipeline(StandardScaler(),  RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10))
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_train, y_train)"""

#print("GNB Model Score:", gnb_score)
print("KNN Model Score:", knn_score)
#print("RF Model Score:", knn_score)


X_unlabelled = unlabelled_data.iloc[:, 2:].values
predictions = knn_model.predict(X_unlabelled)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)
