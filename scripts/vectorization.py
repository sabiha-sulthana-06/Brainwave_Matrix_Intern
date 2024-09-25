import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
def split_data(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    X = df['Text']
    y = df['label']
    return train_test_split(X, y,test_size=0.2, random_state=42)
def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model_dir = r'C:\Users\ASUS\Desktop\fake news detection model\models'
    os.makedirs(model_dir,exist_ok=True)
    joblib.dump(vectorizer,os.path.join(model_dir,'tfidf_vectorizer.pkl'))
    return X_train_tfidf,X_test_tfidf
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\ASUS\Desktop\fake news detection model\data\cleaned_fake_and_real_news.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    preprocessed_data = (X_train_tfidf, X_test_tfidf, y_train.values, y_test.values)
    joblib.dump(preprocessed_data,r'C:\Users\ASUS\Desktop\fake news detection model\models\preprocessed_data.pkl')
