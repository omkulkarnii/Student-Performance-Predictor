
import pandas as pd
import joblib

model = joblib.load('src/model.pkl')  # model.pkl should be saved during training

# Sample input
new_data = pd.DataFrame([{
    "study_time": 2,
    "failures": 0,
    "absences": 3,
    "gender_F": 1,
    "gender_M": 0
}])

prediction = model.predict(new_data)
print("Predicted Final Grade:", prediction[0])
