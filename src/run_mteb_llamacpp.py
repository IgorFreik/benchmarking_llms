import mteb
from huggingface_hub import hf_hub_download
import os

def download_model_for_llama_cpp(model_name: str, save_directory: str, gguf_files: list):
    """
    Download a model in GGUF format from Hugging Face for use with llama.cpp.

    Parameters:
    - model_name (str): The Hugging Face model name (e.g., "TheBloke/Llama-2-7B-GGUF").
    - save_directory (str): The path to the local folder where the model will be saved.

    Returns:
    - None
    """
    os.makedirs(save_directory, exist_ok=True)

    for file in gguf_files:
        file_path = hf_hub_download(
            repo_id=model_name,
            filename=file,
            cache_dir=save_directory,
            local_dir=save_directory,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded '{file}' to '{file_path}'")


if __name__ == "__main__":
    SAVE_DIR = 'results'

    MODEL_NAME = os.environ.get("MODEL_NAME")
    TASK_NAME = os.environ.get("TASK_NAME")
    SAVE_TO_S3 = bool(os.environ.get("SAVE_TO_S3", "False"))

    HF_REPO = '/'.join(MODEL_NAME.split('/')[:-1])
    GGUF_FILE = MODEL_NAME.split('/')[-1]
    GGUF_FILE_BASE = GGUF_FILE[:-5].replace('_', '-')
    MODEL_PATH = f"./weights/{HF_REPO}/{GGUF_FILE}"

    # Download weights
    download_model_for_llama_cpp(HF_REPO, f"./weights/{HF_REPO}", [GGUF_FILE])

    # Run mteb evals
    model = mteb.get_model(MODEL_PATH)
    task = mteb.get_task(task_name=TASK_NAME)
    task.n_experiments = 1
    evaluation = mteb.MTEB(tasks=[task])
    results = evaluation.run(model, output_folder=SAVE_DIR)
    model_pth_formatted = MODEL_PATH.replace('/', '__')
    result_path = f'{SAVE_DIR}/{model_pth_formatted}/no_revision_available/{TASK_NAME}.json'
    print(f'RESULTS PATH:\n{result_path}')

    # Save results
    if SAVE_TO_S3:
        import boto3

        BUCKET_NAME = "bep-results"
        print("===== SAVING TO S3 =====")
        s3 = boto3.client('s3')

        s3.upload_file(result_path, BUCKET_NAME, f'{HF_REPO}/{GGUF_FILE_BASE}-{TASK_NAME}.json')
        print("===== SAVED! ======")
