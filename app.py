from fastapi import FastAPI, Query
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

# Train model at startup
iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)
class_names = ["setosa", "versicolor", "virginica"]

# ✅ Root (VERY IMPORTANT)
@app.get("/")
def root():
    return {"status": "ok"}

# ✅ Health (both versions)
@app.get("/health")
@app.get("/health/")
def health():
    return {"status": "ok"}

# ✅ Predict (strict params)
@app.get("/predict")
def predict(
    sl: float = Query(...),
    sw: float = Query(...),
    pl: float = Query(...),
    pw: float = Query(...)
):
    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    return {"prediction": pred, "class_name": class_names[pred]}
