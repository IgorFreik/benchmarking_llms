from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


DATASETS_TO_SUBSET = [  # Hugging Face dataset paths
    'mteb/stsbenchmark-sts',
    'maximoss/rte3-multi',
]
N_SAMPLES = 300


if __name__ == "__main__":
    login(token="YOUR TOKEN")

    for dataset_name in tqdm(DATASETS_TO_SUBSET):
        dataset = load_dataset(dataset_name)
        test_dataset_len = len(dataset['test'])

        sampled_test_data = (
            dataset['test']
            .shuffle(seed=42)
            .select([i for i in range(min(test_dataset_len, N_SAMPLES))])
        )
        dataset_dict = DatasetDict({
            'test': sampled_test_data
        })

        dataset_name = dataset_name.split('/')[-1]
        dataset_dict.push_to_hub(f"{dataset_name}_{N_SAMPLES}")
