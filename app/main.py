from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow import keras
import numpy as np
import pickle
import joblib



app = FastAPI()

loaded_model = None
threshold = None
# MODEL_PATH = '/home/sudarsan/Python_Projects/Predictive_Maintence/app/my_model'
MODEL_PATH = '/app/my_model'


class Model_Inference(BaseModel):
    ME1: list


def read_threshold_value():
    global threshold
    with open('threshold_value', 'rb') as f:
        data = pickle.load(f)
    threshold = data['threshold']
    print("Loaded threshold from disk")


def load_Scaler():
    global loaded_scaler
    loaded_scaler = joblib.load('scaler_data')
    print("Loaded scaler from disk")


def load_model():
    global loaded_model
    loaded_model = keras.models.load_model(MODEL_PATH)
    print("Loaded model from disk")
    read_threshold_value()
    load_Scaler()


load_model()


@app.get("/")
def read_root():
    return {"Root Endpoint from Predictive Maintence"}


@app.post("/PDM_Model_Inference/")
def PDM_Model_Inference(model_inference: Model_Inference):
    data = np.array(model_inference.ME1)
    data = data.reshape(1, len(data))

    data_scaled = loaded_scaler.transform(data) #2dl uncomment this code


    # reshape inputs for LSTM [samples, timesteps, features]
    data_scaled = data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])


    inference_predicted = loaded_model.predict(data_scaled)
    inference_predicted = inference_predicted.reshape(inference_predicted.shape[0], inference_predicted.shape[2])

    data_scaled = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[2])

    loss_mae = np.mean(np.abs(inference_predicted - data_scaled), axis=1)

    if loss_mae[0] < threshold:
        return "Normal"
    else:
        return "Anomaly Detected: Abnormal Status"


# if __name__ == "__main__":
	# uvicorn.run(host="127.0.0.1", port=8000)
	#uvicorn.run(app,port=8000,host="0.0.0.0")

    # uvicorn main:app --reload
    # http://127.0.0.1:8000/docs
    # http://127.0.0.1:8000/redoc
