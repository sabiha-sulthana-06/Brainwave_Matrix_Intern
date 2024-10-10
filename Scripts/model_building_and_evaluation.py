
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import joblib

X_train = pd.read_csv('X_train_small.csv')
X_test = pd.read_csv('X_test_small.csv')
y_train = pd.read_csv('y_train_small.csv').values.ravel()
y_test = pd.read_csv('y_test_small.csv').values.ravel() 



smote = SMOTE(random_state=42,k_neighbors=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)




model = LogisticRegression(random_state=42,max_iter=1000)
model.fit(X_train_res, y_train_res)


y_pred = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

roc_score = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_score}')

joblib.dump(model,r'C:\Users\ASUS\Desktop\credit card fraud detection\Models\credit_card_fraud_model_small.pkl')

