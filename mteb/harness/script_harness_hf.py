from lm_eval import simple_evaluate
import json
import os
import boto3


if __name__ == "__main__":
    SAVE_DIR = 'results'
    BUCKET_NAME = "bep-results"

    # Load environment
    MODEL_NAME = os.environ.get("MODEL_NAME")
    TASK_NAME = os.environ.get("TASK_NAME")

    # Run evals
    results = simple_evaluate(
        model='hf',
        model_args={
            "model_path": MODEL_NAME,
        },
        tasks=TASK_NAME,
        limit=10,
    )

    # Save results
    print("======= SAVING TO S3 ======")
    s3 = boto3.client('s3')
    save_path = f'{SAVE_DIR}/{MODEL_NAME.replace("/", "__")}-{TASK_NAME}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print('==== WRITING RESULTS =====')
    with open(save_path, 'w') as f:
        json.dump(results['results'], f)
    print('==== WRITTEN! =====')

    s3.upload_file(save_path, BUCKET_NAME, f'{MODEL_NAME}-{TASK_NAME}.json')
    print("===== SAVED! ======")
