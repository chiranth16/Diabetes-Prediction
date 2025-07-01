
# DIABETES PREDICTION with Deep Learning

# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load Dataset
df = pd.read_csv('diabetes.csv')

# Data Cleaning
print('Before dropping duplicates:', df.shape)
df = df.drop_duplicates()
print('After dropping duplicates:', df.shape)

# Check for zero/missing values
print('Missing Glucose:', df[df['Glucose'] == 0].shape[0])
print('Missing BloodPressure:', df[df['BloodPressure'] == 0].shape[0])
print('Missing SkinThickness:', df[df['SkinThickness'] == 0].shape[0])
print('Missing Insulin:', df[df['Insulin'] == 0].shape[0])
print('Missing BMI:', df[df['BMI'] == 0].shape[0])

# Replace zeros with mean
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

# Data Visualization
f, ax = plt.subplots(1, 2, figsize=(10, 5))
df['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')
sns.countplot(data=df, x='Outcome', ax=ax[1])
ax[1].set_title('Outcome')
plt.grid()
plt.show()

# Histograms
df.hist(bins=10, figsize=(10, 10))
plt.show()

# Correlation heatmap
corr_mat = df.corr()
top_corr_features = corr_mat.index
plt.figure(figsize=(10, 10))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()

# Feature & Label Split
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# ----------------- Machine Learning Models -----------------

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize and Train Models
lr_model = LogisticRegression(solver='liblinear', multi_class='ovr')
lr_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

svm_model = SVC()
svm_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(criterion='entropy')
rf_model.fit(X_train, y_train)

# Model Predictions
lr_preds = lr_model.predict(X_test)
knn_preds = knn_model.predict(X_test)
nb_preds = nb_model.predict(X_test)
svm_preds = svm_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Accuracy Scores
print('Logistic Regression:', round(accuracy_score(y_test, lr_preds) * 100, 2))
print('K-Nearest Neighbors:', round(accuracy_score(y_test, knn_preds) * 100, 2))
print('Naive Bayes:', round(accuracy_score(y_test, nb_preds) * 100, 2))
print('Support Vector Machine:', round(accuracy_score(y_test, svm_preds) * 100, 2))
print('Decision Tree:', round(accuracy_score(y_test, dt_preds) * 100, 2))
print('Random Forest:', round(accuracy_score(y_test, rf_preds) * 100, 2))

# Save the Best ML Model (example: SVM)
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# ----------------- Deep Learning Model -----------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Build the Deep Learning Model
dl_model = Sequential()
dl_model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(8, activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))

# Compile the model
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = dl_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = dl_model.evaluate(X_test, y_test)
print(f"\n✅ Deep Learning Model Accuracy: {round(accuracy * 100, 2)}%")

# Save Deep Learning Model
dl_model.save('deep_learning_diabetes_model.h5')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
