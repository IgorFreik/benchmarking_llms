import os
import json
import boto3
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
from lm_eval.api.model import TemplateLM
from lm_eval import simple_evaluate
from lm_eval.api.registry import register_model




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

@register_model("llamacpp_final")
class LLAMACPP(TemplateLM):
    def __init__(self, model_path, embedding: bool=False, max_length=2048, temperature=0.0, **kwargs):
        super().__init__()
        if not embedding:
            self.model = Llama(model_path=model_path, n_ctx=max_length, logits_all=True)
        else:
            self.model = Llama(model_path=model_path, n_ctx=max_length, embedding=True)

        self.logprobs = 10
        self.temperature = temperature
        self.max_length = max_length



    def _call_model(self, ctx, cnt):

        ctx_tokens = self.model.tokenize(ctx.encode('UTF-8'))
        cnt_tokens = self.model.tokenize(cnt.encode('UTF-8'))[1:]

        log_preds = []
        current_ctx_tokens = ctx_tokens

        for token in cnt_tokens:
            probs = self.model(
                self.model.detokenize(current_ctx_tokens).decode('UTF-8'),
                max_tokens=1,
                temperature=0.0,
                logprobs=self.model.n_vocab()
            )
            str_token = self.model.detokenize([token]).decode('UTF-8')

            if 'logprobs' not in probs['choices'][0]:
                raise ValueError("The model output does not include token logprobs.")

            # Get the log probability for the current token
            token_logprob = probs['choices'][0]["logprobs"]['top_logprobs'][0].get(str_token, 1e-10)  # Avoid log(0) by using a small probability
            top_token_str = next(iter(probs['choices'][0]["logprobs"]['top_logprobs'][0]))

            # Add the log of the probability to the log-likelihood
            log_preds.append((token_logprob, top_token_str))

            # Update the context by adding the current token
            current_ctx_tokens.append(token)

        return log_preds

    def _loglikelihood_tokens(
            self,
            requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
            disable_tqdm: bool = False,
        ):

        logpreds = []

        for (context, continuation), context_enc, continuation_enc in tqdm(requests, disable=disable_tqdm):
            response = self._call_model(context, continuation)
            print(f'Response: {response}')

            # Calculate log-likelihood of continuation
            # continuation_logprob = sum(token_logprobs[len(context_enc):len(context_enc) + len(continuation_enc)])
            token_logprobs = [resp[0] for resp in response]
            greedy_tokens = [resp[1] for resp in response]
            continuation_logprob = sum(token_logprobs)

            # Determine greedy match
            # greedy_tokens = choice['logprobs']['tokens'][len(context_enc):len(context_enc) + len(continuation_enc)]
            is_greedy = greedy_tokens == continuation_enc

            logpreds.append((continuation_logprob, is_greedy))

        return logpreds


    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        raise Exception("Not implemented")
        continuations = []

        for request in tqdm(requests, disable=disable_tqdm):
            context, gen_kwargs = request.args

            response = self.model(context, **gen_kwargs)
            choice = response['choices'][0]
            generated_text = choice['text']

            stop_sequence = gen_kwargs.get('until')
            if stop_sequence:
                generated_text = generated_text.split(stop_sequence)[0] if stop_sequence in generated_text else generated_text

            continuations.append(generated_text)

        return continuations

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise Exception("Not implemented")
        results = []

        for request in tqdm(requests, disable=disable_tqdm):
            context = request.args[0]
            total_logprob = 0.0

            tokens = self.model.tokenize(context)
            for i in range(0, len(tokens), self.max_length):
                chunk_tokens = tokens[i : i + self.max_length]

                with torch.no_grad():
                    logits = self.model(chunk_tokens)
                    log_probs = F.log_softmax(logits, dim=-1)

                total_logprob += log_probs.sum().item()

            results.append(total_logprob)

        return results

    @property
    def eot_token_id(self):
        return self.model.token_eos()


    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        return self.model.tokenize(string.encode('UTF-8'))



if __name__ == "__main__":
    SAVE_DIR = 'results'
    BUCKET_NAME = "bep-results"

    MODEL_NAME = os.environ.get("MODEL_NAME")
    TASK_NAME = os.environ.get("TASK_NAME")

    HF_REPO = '/'.join(MODEL_NAME.split('/')[:-1])
    GGUF_FILE = MODEL_NAME.split('/')[-1]
    GGUF_FILE_BASE = GGUF_FILE[:-5].replace('_', '-')
    MODEL_PATH = f"./weights/{HF_REPO}/{GGUF_FILE}"

    # Download weights
    download_model_for_llama_cpp(HF_REPO, f"./weights/{HF_REPO}", [GGUF_FILE])

    # Run evals
    results = simple_evaluate(
        model='llamacpp_final',
        model_args={
            "model_path": MODEL_PATH
        },
        tasks=TASK_NAME,
        limit=10,
    )

    # Save results
    print("======= SAVING TO S3 ======")
    s3 = boto3.client('s3')
    save_path = f'{SAVE_DIR}/{MODEL_NAME.replace("/", "__")}-{TASK_NAME}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results['results'], f)

    s3.upload_file(save_path, BUCKET_NAME, f'{MODEL_NAME}-{TASK_NAME}.json')
    print("===== SAVED! ======")
