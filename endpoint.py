from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from keras.models import model_from_json
import numpy
import tensorflow as tf
import pickle

app = FastAPI()

loaded_model = None
threshold = None


class Model_Inference(BaseModel):
    ME1: list


def read_threshold_value():
    global threshold
    with open("threshold_value", 'rb') as f:
        data = pickle.load(f)
    threshold = data['threshold']
    print(data['threshold'])

def load_model():
    global loaded_model
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    read_threshold_value()


load_model()


@app.get("/")
def read_root():
    return {"Root Endpoint"}


@app.post("/PDM_Model_Inference/")
async def PDM_Model_Inference(model_inference: Model_Inference):


    data = numpy.array(model_inference.ME1)
    resconstruction = loaded_model.predict(data.reshape(1, 66))
    inference_loss = tf.keras.losses.mae(resconstruction, data)

    inference_res = tf.math.less(inference_loss[0], threshold)

    if inference_res:
        return "Normal"
    else:
        return "Anomaly Detected"








if __name__ == "__main__":

    main()

    uvicorn.run(host = "127.0.0.1", port = 5000)


    # uvicorn endpoint:app --reload
    # http://127.0.0.1:8000/docs
    # http://127.0.0.1:8000/redoc