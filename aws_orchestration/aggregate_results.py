import boto3
from tqdm import tqdm
from aws_orchestration.utils import parse_gguf_model_quant
import json

RESULTS_S3_BUCKET_NAME = ...  # Insert results


def get_json_files_from_s3(bucket_name):
    # Create an S3 client
    s3_client = boto3.client("s3", region_name='eu-west-1')

    json_files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    # Use paginator to iterate over all objects in the bucket
    for page in paginator.paginate(Bucket=bucket_name):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(".json"):
                    json_files.append(key)

    return json_files

def read_json_from_s3(bucket_name, file_key):
    s3_client = boto3.client('s3', region_name='eu-west-1')
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    return json.loads(content)


if __name__ == "__main__":
    json_files = get_json_files_from_s3(bucket_name=RESULTS_S3_BUCKET_NAME)
    results = {}

    results_dict = dict()

    for file_key in tqdm(json_files):
        task_name = file_key.split('-')[-1][:-5]
        model_name = "-".join(file_key.split('-')[:-1])
        json_content = read_json_from_s3(RESULTS_S3_BUCKET_NAME, file_key)
        results[task_name] = json_content

        # Quantized (GGUF)
        if len(model_name.split('/')) == 3:
            quant = parse_gguf_model_quant(model_name)
            model_name = "/".join(model_name.split('/')[:2])
        else:  # Non-quantized
            quant = None

        if 'scores' in json_content:
            results_dict[file_key] = {
                "metric": json_content['scores']['test'][0]['main_score'],
                "task_name":task_name,
                "model_name": model_name,
                "quant": quant,
            }
        else:
            raise Exception(f"Scores not found for task {task_name} in {json_files}")

    with open('results.json', 'w') as f:
        json.dump(results_dict, f)

    print(results)
