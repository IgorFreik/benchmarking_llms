"""The data selection for this script is largely inspired by this source: https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/main/src/seb/registered_tasks/speed.py"""
"""The dataset used for speed testing is an original book by Hans Anderson."""

import time
import psutil
from llama_cpp import Llama
from huggingface_hub import list_repo_tree, hf_hub_download
import json
import os
import shutil

ALLOWED_QUANTS = [
    'q2k',
    'q3kl',
    'q3km',
    'q3ks',
    'q40',
    'q4km',
    'q4ks',
    'q50',
    'q5km',
    'q5ks',
    'q6k',
    'q80'
]
TOKENS_IN_UGLY_DUCKLING = 3591
MODELS = [
    'ChristianAzinn/e5-base-v2-gguf',
    'nomic-ai/nomic-embed-text-v1.5-GGUF',
    'second-state/E5-Mistral-7B-Instruct-Embedding-GGUF'
]


def save_results(results_dict, filename='metrics_results_llamacpp.json'):
    """Save results to the JSON file."""
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=4)

def load_existing_results(filename='metrics_results_llamacpp.json'):
    """Load existing results from the JSON file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

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

    return file_path

def empty_folder(folder_path):
    # Ensure the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through each file and folder in the directory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # If it's a file, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # If it's a directory, delete it and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")

def measure_metrics_llama_cpp(
        gguf_model_path: str,
        prompts: list[str],
        is_embedding: bool = False,

):
    """
    1) Measure memory usage before and after loading the model (via psutil).
    2) Calculate total latency, memory usage
    3) Return a summary of latency and memory usage.
    """
    def get_memory_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


    ### LOADING ###
    mem_before_load = get_memory_mb()
    load_start = time.time()

    llama = Llama(model_path=gguf_model_path, embedding=is_embedding)
    llama.embed(["encode this"])  # ensure model is loaded

    load_time = time.time() - load_start
    mem_after_load = get_memory_mb()

    ### EMBEDDING ###
    gen_start = time.time()
    for prompt in prompts:
        llama.embed(prompt)
    gen_total_time = time.time() - gen_start


    ### SUMMARIZE ###
    metrics = {
        # times
        "load_time_s": load_time,
        "total_latency_s": gen_total_time,
        "words / sec": TOKENS_IN_UGLY_DUCKLING / gen_total_time,

        # memory usage
        "mem_before_load_mb": mem_before_load,
        "mem_after_load_mb": mem_after_load,
        "load_memory_mb": mem_after_load - mem_before_load,
    }
    return metrics


def sample_prompts() -> list[str]:
    """A small list of prompts for testing based on the ugly duckling book."""
    with open("./the_ugly_duckling.txt", "rt") as f:
        text = f.read()
    return text.split("\n\n")


def main_llamacpp(repo_ids):
    prompts = sample_prompts()
    results_dict = load_existing_results()

    for repo_id, is_embedding in repo_ids:
        repo_tree = list_repo_tree(repo_id)
        quants = [file_info.path for file_info in repo_tree]
        quants = [q for q in quants if q.endswith('.gguf')]

        filtered_quants = set()
        for q in quants:
            for allowed_q in ALLOWED_QUANTS:
                if allowed_q in q.replace('-', '').replace('_', '').lower():
                    filtered_quants.add(q)

        quants = list(filtered_quants)



        for quant in quants:
            save_name = f'{repo_id}/{quant}'
            previous_results = results_dict.get(save_name, [])

            if len(previous_results) >= 5:
                print(f"Skipping {save_name}, 5 results already saved.")
                continue

            download_model_for_llama_cpp(repo_id, 'weights', [quant])

            results = measure_metrics_llama_cpp(
                gguf_model_path=os.path.join('weights', quant),
                prompts=prompts,
                is_embedding=is_embedding
            )

            previous_results.append(results)
            results_dict[save_name] = previous_results
            save_results(results_dict)

            print("=== Metrics ===")
            for k, v in results.items():
                print(f"{k}: {v}")

        empty_folder('weights')


if __name__ == "__main__":
    main_llamacpp(MODELS)
