import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
df = pd.read_csv(r'C:\Users\ASUS\Desktop\fake news detection model\data\fake_and_real_news.csv')
print(df.head())
if 'Text' in df.columns and 'label' in df.columns:
    def clean_text(text):
        text = re.sub(r'\W', ' ',text)
        text = text.lower()
        text = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        return ' '.join(text)
    df['Text'] = df['Text'].apply(clean_text)
    cleaned_file_path = r'C:\Users\ASUS\Desktop\fake news detection model\data\cleaned_fake_and_real_news.csv'
    df.to_csv(cleaned_file_path,index=False)
