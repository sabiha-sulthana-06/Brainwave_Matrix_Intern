import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r'C:\Users\ASUS\Desktop\credit card fraud detection\Data\creditcard.csv')
df_sample = df.sample(frac=0.04,random_state=42)
df_sample.to_csv(r'C:\Users\ASUS\Desktop\credit card fraud detection\Data\creditcard_sample.csv', index=False)
print(df_sample.columns)
print(df_sample['Class'].value_counts())
print(df_sample.describe())

df_sample['Amount_scaled'] = StandardScaler().fit_transform(df_sample['Amount'].values.reshape(-1,1))
df_sample['Time_scaled'] = StandardScaler().fit_transform(df_sample['Time'].values.reshape(-1,1))

df_sample = df_sample.drop(['Amount', 'Time'], axis=1)

X = df_sample.drop('Class', axis=1)
y = df_sample['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


X_train_res.to_csv('X_train_small.csv', index=False)
X_test.to_csv('X_test_small.csv', index=False)
y_train_res.to_csv('y_train_small.csv', index=False)
y_test.to_csv('y_test_small.csv', index=False)
