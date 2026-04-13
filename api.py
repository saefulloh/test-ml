from fastapi import FastAPI
import boto3
import json

app = FastAPI()

runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")

ENDPOINT_NAME = "demo-endpoint"

@app.post("/predict")
async def predict(data: dict):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(data["instances"])
    )

    result = response["Body"].read().decode()
    return {"prediction": json.loads(result)}
