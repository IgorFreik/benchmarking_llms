import boto3
from tqdm import tqdm
from aws_orchestration.utils import *
from constants import MODEL_CONFIGS, TASK_CONFIGS, TASK_NAMES, ALLOWED_QUANTS


TASK_DEFINITION = 'mteb-hf'  # SWITCH to 'mteb-llamacpp' to run quantized models


if __name__ == "__main__":
    # Initialize the ECS client
    ecs_client = boto3.client('ecs', region_name='eu-west-1')

    completed_tasks = load_completed_tasks(TASK_DEFINITION)

    # Trigger tasks
    for model_cfg in tqdm(MODEL_CONFIGS[TASK_DEFINITION]):
        if '-llamacpp' in TASK_DEFINITION:
            model_name = model_cfg['gguf']
            backend = 'llamacpp'
        elif '-hf' in TASK_DEFINITION:
            model_name = model_cfg['model']
            backend = 'hf'
        else:
            raise Exception('Invalid Task Definition!')

        print(f'>> Selected backend: {backend}')
        print(f'========= TRIGGERING FOR {model_name} ==========')

        revision_default = TASK_CONFIGS[TASK_DEFINITION]['revision_map'][model_cfg['revision']][0]

        for task_name in tqdm(TASK_NAMES[TASK_DEFINITION]):
            print(f'========= TRIGGERING FOR {task_name} ==========')

            if backend == 'llamacpp':
                revision_map = TASK_CONFIGS[TASK_DEFINITION]['revision_map']
                files = get_quant_file_sizes_hf(model_name, revision_map)

                for f_name, (q, size, revision) in files.items():
                    print(f'========= TRIGGERING FOR {f_name} ==========')
                    model_name_full = f'{model_name}/{f_name}'

                    # Skip if the task has already been completed
                    if is_task_completed(completed_tasks, task_name, model_name_full):
                        print(f'Task {task_name} for model {model_name_full} already completed. Skipping.')
                        continue

                    if q not in ALLOWED_QUANTS:
                        print(f'Quant {q} not relevant for the evaluation. Skipping!')
                        continue

                    response = trigger_task_with_retries(
                        ecs_client,
                        TASK_DEFINITION,
                        task_name,
                        model_name_full,
                        revision_default,
                    )
                    print(response)
                    mark_task_as_completed(completed_tasks, task_name, model_name_full, TASK_DEFINITION)
                    time.sleep(2)
            elif backend == 'hf':
                # Skip if the task has already been completed
                if is_task_completed(completed_tasks, task_name, model_name):
                    print(f'Task {task_name} for model {model_name} already completed. Skipping.')
                    continue

                response = trigger_task_with_retries(
                    ecs_client,
                    TASK_DEFINITION,
                    task_name,
                    model_name,
                    revision_default
                )
                print(response)

                mark_task_as_completed(completed_tasks, task_name, model_name, TASK_DEFINITION)
                time.sleep(3)
            else:
                raise Exception('Backend not implemented!')
