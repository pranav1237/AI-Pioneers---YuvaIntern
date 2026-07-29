import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/Titanic-Dataset.csv")

df.head()

df.shape
df.info()
df.describe()
df.columns

df.isnull().sum()

df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)

df.select_dtypes(include='object').columns


encoder = LabelEncoder()

df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])

scaler = StandardScaler()

cols = ['Age','Fare']

df[cols] = scaler.fit_transform(df[cols])

sns.countplot(x='Survived', data=df)

plt.title("Survival Count")
plt.show()

sns.countplot(x='Sex', data=df)

plt.title("Gender Distribution")
plt.show()

sns.histplot(df['Fare'], bins=30)

plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df)
plt.show()

df.head()
df.info()
df.describe()

df.to_csv("output/cleaned_titanic.csv", index=False)
