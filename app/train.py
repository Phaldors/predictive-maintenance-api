from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import generate_sensor_data
import joblib

X, y = generate_sensor_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
tahminler = model.predict(X_test)
accuracy_score(y_test, tahminler)
dogruluk = accuracy_score(y_test, tahminler)
print(dogruluk)
joblib.dump(model, "model.pkl")