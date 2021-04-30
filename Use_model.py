import joblib
import numpy as np

from dataprocess.Train_model import get_data

model = joblib.load('./model/log_reg.pkl')
data = get_data('./data2/')
data = np.array(data[0]).reshape((1, -1))
res = model.predict(data)
predict = model.predict_proba(data)
print(predict)
