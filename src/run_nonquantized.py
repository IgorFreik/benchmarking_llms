import mteb
import os
import json

def find_file(start_directory, file_name):
    for root, dirs, files in os.walk(start_directory):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


if __name__ == "__main__":
    SAVE_DIR = 'results'
    BUCKET_NAME = "bep-results"

    MODEL_NAME = os.environ.get("MODEL_NAME")
    TASK_NAME = os.environ.get("TASK_NAME")
    SAVE_TO_S3 = os.environ.get("SAVE_TO_S3")

    # Run mteb evals
    model = mteb.get_model(MODEL_NAME)
    task = mteb.get_task(task_name=TASK_NAME)
    task.n_experiments = 1
    evaluation = mteb.MTEB(tasks=[task])
    results = evaluation.run(model, output_folder=SAVE_DIR)

    result_path = find_file(SAVE_DIR, f'{TASK_NAME}.json')
    print(f'>>> RESULTS PATH:\n{result_path}')

    if result_path:
        with open(result_path, 'r') as f:
            res = json.load(f)
            print(f">>> RESULTS MAIN SCORE: {res['scores']['test'][0]['main_score']}")
            print(f">>> FULL RESULTS: {res['scores']['test'][0]}")

        if SAVE_TO_S3 and SAVE_TO_S3 == "True":
            import boto3

            s3 = boto3.client('s3')

            if result_path is not None:
                s3.upload_file(result_path, BUCKET_NAME, f'{MODEL_NAME}-{TASK_NAME}.json')
                print(">> SAVED RESULTS!")
    else:
        print(f"File '{TASK_NAME}.json' not found in '{SAVE_DIR}'.")
