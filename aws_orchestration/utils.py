from aws_orchestration.constants import COMPLETED_TASKS_FILE
from huggingface_hub import list_repo_tree
import json
import os
import time

AWS_ACCESS_KEY_ID = ...  # Replace with yours
AWS_SECRET_ACCESS_KEY = ...  # Replace with yours
AWS_DEFAULT_REGION = ...  # Replace with yours
SECURITY_GROUP = ...  # Replace with yours
SUBNET = ...  # Replace with yours
ECS_CONTAINER_NAME = ...  # Replace with yours
CLUSTER_NAME = ...  # Replace with yours


def trigger_task(ecs_client, task_definition: str, task_name: str, model_name: str, revision: str):
    print(f'TASK DEFINITION: {task_definition}')
    response = ecs_client.run_task(
        cluster=CLUSTER_NAME,
        launchType='FARGATE',
        taskDefinition=f'{task_definition}:{revision}',
        overrides={
            'containerOverrides': [
                {
                    'name': ECS_CONTAINER_NAME,
                    'environment': [
                        {'name': 'TASK_NAME', 'value': task_name},
                        {'name': 'MODEL_NAME', 'value': model_name},
                        {'name': 'AWS_ACCESS_KEY_ID', 'value': AWS_ACCESS_KEY_ID},
                        {'name': 'AWS_SECRET_ACCESS_KEY', 'value': AWS_SECRET_ACCESS_KEY},
                        {'name': 'AWS_DEFAULT_REGION', 'value': AWS_DEFAULT_REGION}
                    ]
                }
            ]
        },
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [SUBNET],
                'securityGroups': [SECURITY_GROUP],
                'assignPublicIp': 'ENABLED'
            }
        }
    )
    return response

def load_completed_tasks(task_definition):
    save_path = COMPLETED_TASKS_FILE.format(task_definition)

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            return json.load(f)
    return {}

def save_completed_tasks(completed_tasks, task_definition):
    with open(COMPLETED_TASKS_FILE.format(task_definition), 'w') as f:
        json.dump(completed_tasks, f, indent=4)

def is_task_completed(completed_tasks, task_name, model_name):
    return completed_tasks.get(task_name, {}).get(model_name, False)

def mark_task_as_completed(completed_tasks, task_name, model_name, task_definition):
    if task_name not in completed_tasks:
        completed_tasks[task_name] = {}
    completed_tasks[task_name][model_name] = True
    save_completed_tasks(completed_tasks, task_definition)

def trigger_task_with_retries(ecs_client, task_definition: str, task_name: str, model_name: str, revision: str):
    while True:
        response = trigger_task(ecs_client, task_definition, task_name, model_name, revision)

        if len(response['failures']) == 0:
            return response

        print(f'>>> Failed with error: {response["failures"]}\n')
        time.sleep(60)

def parse_gguf_model_quant(model_name: str) -> str:
    # quant_mapping = dict()
    quant_mappings = []

    for q_precision in [32, 16, 2, 3, 4, 5, 6, 8]:
        for q_type in ['xxs', 'xs', 'nl', 'ks', 'km', 'kl', 's', 'k', '0', '1', '']:
            for q_prefix in ['iq', 'f', 'fp', 'q']:
                quant_mappings.append(f'{q_prefix}{q_precision}{q_type}')

    # Normalize the model name for comparison (lowercase and remove spaces/underscores)
    normalized_name = model_name.lower().replace(" ", "").replace("_", "").replace("-", "")

    # Match the quantization type
    for quant in quant_mappings:
        if quant in normalized_name:
            return quant

def get_quant_required_revision(size_mb, revision_map):
    req_mem_mb = size_mb * 2
    for i in range(len(revision_map)):
        if revision_map[i][1] > req_mem_mb:
            return revision_map[i][0]
    return -1

def get_quant_file_sizes_hf(repo_id: str, revision_map: list) -> dict:
    file_sizes = {}

    # Fetch file metadata from the repository
    repo_tree = list_repo_tree(repo_id)

    for file_info in repo_tree:
        quant = parse_gguf_model_quant(file_info.path)

        if quant:
            # Fetch file size using the metadata API
            file_size_mb = int(file_info.size) / (1024 * 1024)
            file_sizes[file_info.path] = (
                quant,
                file_size_mb,
                get_quant_required_revision(file_size_mb, revision_map)
            )
        elif file_info.path.endswith('.gguf'):
            print(f'>>>>> FAILED TO PARSE: {file_info.path}!')

    return file_sizes
