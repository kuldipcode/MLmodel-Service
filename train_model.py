import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load  data

df = pd.read_csv("pullback.csv")
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['population', 'transportation']], df['arcservice'],test_size=0.2,random_state=12)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)
print(X_test)
# Make prediction on the test set
y_predict = lr.predict(X_test)


# Save model
with open('model.pickle', 'wb') as f:
    pickle.dump(lr, f)