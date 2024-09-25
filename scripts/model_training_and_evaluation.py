import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
X_train_tfidf, X_test_tfidf, y_train, y_test = joblib.load(r'C:\Users\ASUS\Desktop\fake news detection model\models\preprocessed_data.pkl')
scaler = StandardScaler(with_mean=False)
X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.transform(X_test_tfidf)
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, zero_division=0))
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
y_pred_nb = model_nb.predict(X_test_tfidf)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}")
joblib.dump(model,r'C:\Users\ASUS\Desktop\fake news detection model\models\fake_news_detection_model.pkl')
