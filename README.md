## How to run

### Creating dataset subsets

1. First, install dependencies:
```commandline
pip install datasets==3.2.0 huggingface-hub==0.28.1 tqdm==4.67.1
```
2. [Optional] Select the datasets to take subsets by changing the `DATASETS_TO_SUBSET` varible.
3. Proved your HF token in the same script to enable saving the subset to hugging face.
4. Run the subset creation:
```commandline
python create_datasets.py
```

### Running models on mteb
[Optional] It is possible to change the datasets to your subset created with `create_dataset.py` script. This can be done by changing "path" variable in the task source to the new dataset in the source. For example, for `STSBenchmark`, make the following adjustment in the `src/mteb/tasks/STS/eng/STSBenchmarkSTS.py`. For `RTE3` -- `src/mteb/tasks/PairClassification/multilingual/RTE3.py`
```
dataset={
    "path": <NEW_HF_PATH>,
    "revision": None,
}
```

#### Nonquantized models

1. First, install dependencies:
```commandline
pip install -r requirements_sentence_transformers.txt
```
2. Specify the task and model to evaluate in the environment variables `TASK_NAME` and `MODEL_NAME`, for example:
```commandline
export TASK_NAME=STSBenchmark
export MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
```
3. Run the models:
```commandline
python src/run_sentence_transformers.py
```
4. Done, the results can be found in the path printed by the script. Full path is printed by the script.

#### Quantized models

1. First, install dependencies:
```commandline
pip install -r requirements_sentence_transformers.txt
```
2. Specify the task and model to evaluate in the environment variables `TASK_NAME` and `MODEL_NAME`, for example:
```commandline
export TASK_NAME=STSBenchmark
export MODEL_NAME=nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf
```
3. Run the models:
```commandline
python src/run_llamacpp.py
```
4. Done, the results can be found in the path printed by the script. Full path is printed by the script.


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

