import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_wine()
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print(df.head())


X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Bagging
bagging = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

bag_accuracy = accuracy_score(y_test, y_pred_bag)
print("Bagging Accuracy % :", bag_accuracy * 100)

# Boosting
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)

boost_accuracy = accuracy_score(y_test, y_pred_boost)
print("Boosting Accuracy % :", boost_accuracy * 100)


methods = ['Bagging', 'Boosting']
accuracies = [bag_accuracy * 100 , boost_accuracy * 100 ]

plt.bar(methods, accuracies)
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Bagging and Boosting Methods')
plt.show()
