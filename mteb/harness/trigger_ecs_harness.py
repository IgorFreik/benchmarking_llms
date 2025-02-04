import boto3
import os
from tqdm import tqdm

from mteb.abstasks.TaskMetadata import TASK_CATEGORY
from test_parsing import get_quant_file_sizes_hf
from orchestration import *
from constants import TASK_CONFIGS, TASK_NAMES, MODEL_CONFIGS

# IMPORTANT
os.environ['AWS_PROFILE'] = 'igor'
TASK_DEFINITION = 'harness-hf'
# TASK_DEFINITION = 'harness-llamacp    "ChristianAzinn/e5-base-v2-gguf/e5-base-v2_fp32.gguf": {
#         "model_path": "weights/e5-base-v2_fp32.gguf",
#         "num_prompts": 3,
#         "n_predict": 16,
#         "load_time_s": 0.4361305236816406,
#         "average_ttft_s": 0.2612005869547526,
#         "average_tpot_s": 2.6463322513054785e-07,
#         "average_total_latency_s": 0.2614041169484456,
#         "mem_before_load_mb": 425.7265625,
#         "mem_after_load_mb": 841.36328125,
#         "mem_after_gen_mb": 841.36328125
#     },
#     "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q3_K_L.gguf": {
#         "model_path": "weights/nomic-embed-text-v1.5.Q3_K_L.gguf",
#         "num_prompts": 3,
#         "n_predict": 16,
#         "load_time_s": 0.43377232551574707,
#         "average_ttft_s": 0.18982497851053873,
#         "average_tpot_s": 2.907442559578376e-07,
#         "average_total_latency_s": 0.1900488535563151,
#         "mem_before_load_mb": 425.97265625,
#         "mem_after_load_mb": 493.5859375,
#         "mem_after_gen_mb": 493.5859375
#     },
#     "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q3_K_M.gguf": {
#         "model_path": "weights/nomic-embed-text-v1.5.Q3_K_M.gguf",
#         "num_prompts": 3,
#         "n_predict": 16,
#         "load_time_s": 0.32293224334716797,
#         "average_ttft_s": 0.297105073928833,
#         "average_tpot_s": 2.641151491220699e-07,
#         "average_total_latency_s": 0.2973085244496663,
#         "mem_before_load_mb": 426.07421875,
#         "mem_after_load_mb": 489.42578125,
#         "mem_after_gen_mb": 489.42578125
#     },p'
#



if __name__ == "__main__":
    ecs_client = boto3.client('ecs', region_name='eu-west-1')
    completed_tasks = load_completed_tasks(TASK_DEFINITION)

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

                    response = trigger_task_with_retries(
                        ecs_client,
                        TASK_DEFINITION,
                        task_name,
                        model_name_full,
                        # revision,
                        revision_default,
                    )
                    print(response)
                    # raise Exception("Testing")
                    mark_task_as_completed(completed_tasks, task_name, model_name_full, TASK_DEFINITION)
                    time.sleep(1)

            elif backend == 'hf':
                # Skip if the task has already been completed
                if is_task_completed(completed_tasks, task_name, model_name):
                    print(f'Task {task_name} for model {model_name} already completed. Skipping.')
                    continue

                response = trigger_task_with_retries(
                    ecs_client, TASK_DEFINITION, task_name, model_name, revision_default
                )
                print(response)
                time.sleep(10)
                # if len(response['failures']) > 0:
                #     exit()
                # raise Exception("Testing")
                mark_task_as_completed(completed_tasks, task_name, model_name, TASK_DEFINITION)
            else:
                raise Exception('Backend not implemented!')
