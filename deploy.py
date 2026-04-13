import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
bucket = "your-bucket-name"

# upload model.tar.gz manually dulu ke S3:
model_data = f"s3://{bucket}/model.tar.gz"

model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1"
)

predictor = model.deploy(
    endpoint_name="demo-endpoint",
    instance_type="ml.m5.large",
    initial_instance_count=1
)

print("Endpoint deployed:", predictor.endpoint_name)
