import os
import pandas as pd

RESULT_TASKS = ['RTE3', 'STSBenchmark']
RESULT_MODELS = ['intfloat/e5-base-v2', 'nomic-ai/nomic-embed-text-v1.5', 'intfloat/e5-mistral-7b-instruct']
MODEL_CONFIGS_MTEB = [
    {
        "model": 'intfloat/e5-base-v2',
        "gguf": "ChristianAzinn/e5-base-v2-gguf",
    },
    {
        "model": 'nomic-ai/nomic-embed-text-v1.5',
        "gguf": "nomic-ai/nomic-embed-text-v1.5-GGUF",
    },
    {
        "model": 'intfloat/e5-mistral-7b-instruct',
        "gguf": "second-state/E5-Mistral-7B-Instruct-Embedding-GGUF",
    },
]


def format_df(df):
    res_df = df.copy()
    res_df = res_df.rename(
        columns={
            'task_name': 'Dataset',
            'model_name': 'Model',
            'quant': "Quantization",
            'metric': 'Score',
        }
    )
    res_df['Quantization'] = res_df['Quantization'].map(
        {
            'q2k': 'Q2_K',
            'q3k': 'Q3_K',
            'q3kl': 'Q3_K_L',
            'q3km': 'Q3_K_M',
            'q3ks': 'Q3_K_S',
            'q40': 'Q4_0',
            'q4k': 'Q4_K',
            'q4kl': 'Q4_K_L',
            'q4km': 'Q4_K_M',
            'q4ks': 'Q4_K_S',
            'q50': 'Q5_0',
            'q5ks': 'Q5_K_S',
            'q5km': 'Q5_K_M',
            'q60': 'Q6_0',
            'q6k': 'Q6_K',
            'q80': 'Q8_0',
            None: 'Unquantized',
        }
    )
    return res_df

if __name__ == "__main__":
    RAW_RESULT_PATH = os.environ.get(
        'RESULTS_PATH',
        os.path.join(os.path.dirname(__file__), 'artifacts', 'example_result.json')
    )
    df = pd.read_json(RAW_RESULT_PATH).T

    # Filter valid models
    all_models = [cfg['model'] for cfg in MODEL_CONFIGS_MTEB] + [cfg['gguf'] for cfg in MODEL_CONFIGS_MTEB]
    df = df[df.model_name.isin(all_models)]
    df = df.reset_index()
    df = df.drop(columns=['index'])

    map_dict = {m['model']: m['gguf'] for m in MODEL_CONFIGS_MTEB}

    for key, value in map_dict.items():
        df.loc[df['model_name'] == value, 'model_name'] = key

    df.model_name.unique()


    result_df = df[df['task_name'].isin(RESULT_TASKS) & df['model_name'].isin(RESULT_MODELS)]

    res1 = format_df(result_df[result_df['task_name'] == RESULT_TASKS[0]])
    res2 = format_df(result_df[result_df['task_name'] == RESULT_TASKS[1]])

    res12 = pd.merge(
        res1.drop(columns='Dataset'),
        res2.drop(columns='Dataset'),
        on=['Model', 'Quantization'],
        how='outer',
        suffixes=(f'_{RESULT_TASKS[0]}', f'_{RESULT_TASKS[1]}')
    ).set_index(['Model', 'Quantization']).sort_index()

    # Print results
    print("Formatted Results Table:")
    print(res12)

    # Save results to CSV and LaTeX files
    OUTPUT_DIR = 'results_output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    res12_csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    res12_latex_path = os.path.join(OUTPUT_DIR, 'results.tex')

    res12.to_csv(res12_csv_path, index=False)
    with open(res12_latex_path, 'w') as f:
        f.write(res12.to_latex().replace('_', '\_'))

    print(f"Results saved to: {res12_csv_path} and {res12_latex_path}")


