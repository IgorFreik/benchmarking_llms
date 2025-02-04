COMPLETED_TASKS_FILE = "completed_tasks_{}.json"


TASK_CONFIGS = {  # Revision ID, Memory in MB
    "mteb-hf": {
        'revision_map': [
            ('1', None),
            ('3', None),
            ('5', None),
        ],
    },
    "mteb-llamacpp": {
        'revision_map': [
            ('2', 4 * 1024),
            ('16', 8 * 1024),
            ('13', 16 * 1024),
            ('14', 30 * 1024),
            ('15', 60 * 1024),
        ],
    },
}


MODEL_CONFIGS_MTEB = [
    {
        "model": 'intfloat/e5-mistral-7b-instruct',
        "gguf": "second-state/E5-Mistral-7B-Instruct-Embedding-GGUF",
        "revision": 2
    },

    {
        "model": 'intfloat/e5-base-v2',
        "gguf": "ChristianAzinn/e5-base-v2-gguf",
        'revision': 0,
    },

    {
        "model": 'nomic-ai/nomic-embed-text-v1.5',
        "gguf": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        'revision': 0,
    },
]

MODEL_CONFIGS = {
    'mteb-hf': MODEL_CONFIGS_MTEB,
    'mteb-llamacpp': MODEL_CONFIGS_MTEB,
}

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

TASK_NAMES_MTEB = [
    'RTE3',
    'STSBenchmark',
]

TASK_NAMES = {
    'mteb-hf': TASK_NAMES_MTEB,
    'mteb-llamacpp': TASK_NAMES_MTEB,
}
