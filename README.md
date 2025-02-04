## How to run

### Creating dataset subsets

1. First, install dependencies:
```commandline
pip install datasets==3.2.0 huggingface-hub==0.28.1 tqdm==4.67.1
```
2. Select the datasets to take subsets from by changing the `DATASETS_TO_SUBSET` varible in the `create_datasets.py` file. The default list is `["RTE3" and "STSBenchmark"]`, which aligns with the methodology of the paper.  
3. Proved a valid HF token in the same script. This is required to enable saving the subset to Hugging Face. Unfortunately, the whole MTEB package is built around Hugging Face dataset repository, therefore this approach was utilized throughout the experiments and enabling local datasets was not prioritized. 
4. Run the subset creation:
```commandline
python create_datasets.py
```

### Running evaluations
[Optional] It is possible to change the datasets to your subset created with `create_dataset.py` script. This can be done by changing "path" variable in the task source to a new valid path of a Hugging Face dataset in the source. For example, for `STSBenchmark`, make the following adjustment in the `src/mteb/tasks/STS/eng/STSBenchmarkSTS.py`. For `RTE3` -- `src/mteb/tasks/PairClassification/multilingual/RTE3.py`
```
dataset={
    "path": <NEW_HF_PATH>,
    "revision": None,
}
```
Unfortunately, the whole MTEB package is built around Hugging Face dataset repository, therefore this approach was utilized throughout the experiments and enabling local datasets was not prioritized. 

To reproduce the results of this study, no changes are required -- all corresponding subsets are already set up.

#### Nonquantized models

1. First, install dependencies:
```commandline
pip install ./mteb
pip install -r requirements_nonquantized.txt
```
2. Specify the task and model to evaluate in the environment variables `TASK_NAME` and `MODEL_NAME`, for example:
```commandline
export TASK_NAME=RTE3
export MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
```
The `MODEL_NAME` here is a valid model name to be pulled from Hugging Face.
3. Run the models:
```commandline
python src/run_nonquantized.py
```
4. Done, the main score, full results and saved results path are printed by the script. 

#### Quantized models

1. First, install dependencies:
```commandline
pip install ./mteb
pip install -r requirements_llamacpp.txt
```
2. Specify the task and model to evaluate in the environment variables `TASK_NAME` and `MODEL_NAME`, for example:
```commandline
export TASK_NAME=RTE3
export MODEL_NAME=nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf
```
The `MODEL_NAME` here should be in format `HF_MODEL/FILE_NAME.gguf`. HF_MODEL -- a valid HF model repository, `FILE_NAME` -- a file name within that repository to use. It is assumed that each represents a single separate quantized model at a specific quantization level.
3. Run the models:
```commandline
python src/run_llamacpp.py
```
4. Done, the main score, full results and saved results path are printed by the script.

### [Optional] Creating aggregated tables tables

The script `create_result_tables.py` script processes and formats results from a JSON file and saves the output as a CSV and LaTeX file.
Here is how to run it:
1. Specify the JSON results path as environment variable: 
```commandline
export RESULTS_PATH=path/to/file.json
```
The following format for results is expected:
```JSON
{
  "ChristianAzinn/e5-base-v2-gguf/e5-base-v2.Q4-0-RTE3.json": {
    "metric": 0.956394,
    "task_name": "RTE3",
    "model_name": "ChristianAzinn/e5-base-v2-gguf",
    "quant": "q40"
  },
  "intfloat/e5-base-v2-RTE3.json": {
    "metric": 0.957161,
    "task_name": "RTE3",
    "model_name": "intfloat/e5-base-v2",
    "quant": null
  },
  ...
}
```

### [Optional] AWS orchestration

Given the large number of independent model runs required for this study and the highly parallelizable nature of the process, AWS orchestration was used to accelerate and structure the workflow.

The implementation can be found in the aws_orchestration folder. The `main.py` file is the entry point. However, due to the substantial AWS account requirements and potential high costs, running this setup is not expected within the scope of this project. Instead, it serves as a transparent description of the approach used to obtain the reported results.

## Acknowledgments

This project makes use of the [MTEB framework](https://github.com/embeddings-benchmark/mteb) and the following datasets:

### RTE3
- **Description:** Recognising Textual Entailment Challenge (RTE-3)
- **Reference:** [ACL Anthology - W07-1401](https://aclanthology.org/W07-1401/)
- **BibTeX Citation:**
  ```bibtex
  @inproceedings{giampiccolo-etal-2007-third,
      title = "The Third {PASCAL} Recognizing Textual Entailment Challenge",
      author = "Giampiccolo, Danilo  and
      Magnini, Bernardo  and
      Dagan, Ido  and
      Dolan, Bill",
      booktitle = "Proceedings of the {ACL}-{PASCAL} Workshop on Textual Entailment and Paraphrasing",
      month = jun,
      year = "2007",
      address = "Prague",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/W07-1401",
      pages = "1--9",
  }
### STSBenchmark
- **Description:** Semantic Textual Similarity Benchmark (STSbenchmark) dataset.
- **Reference:** [GitHub - PhilipMay/stsb-multi-mt]https://github.com/PhilipMay/stsb-multi-mt/)
- **BibTeX Citation:**
  ```bibtex
  @InProceedings{huggingface:dataset:stsb_multi_mt,
    title = {Machine translated multilingual STS benchmark dataset.},
    author={Philip May},
    year={2021},
    url={https://github.com/PhilipMay/stsb-multi-mt}
    }

