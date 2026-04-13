import joblib
import json

def model_fn(model_dir):
    return joblib.load(f"{model_dir}/model.pkl")

def input_fn(request_body, request_content_type):
    return json.loads(request_body)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
