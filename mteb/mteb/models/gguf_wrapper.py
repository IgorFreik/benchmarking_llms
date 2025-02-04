from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from mteb.encoder_interface import PromptType

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


def gguf_wrapper(
    model_path: str,
    model_prompts: dict[str, str] | None = None,
    **kwargs,
):
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            f"Please install `pip install llama-cpp-python` to use {model_path}."
        )

    class GGUFWrapper(Wrapper):
        def __init__(
            self,
            model_path: str,
            model_prompts: dict[str, str] | None = None,
            **kwargs,
        ) -> None:
            """Wrapper for GGUF models.

            Args:
                model: The Llama model to use. Should be a string path pointing to the local gguf file to use.
                model_prompts: A dictionary mapping task names to prompt names.
                    First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                    then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                    and finally to the specific prompt type.
                **kwargs: Additional arguments to pass to the Llama model.
            """
            self.model = Llama(model_path=model_path, embedding=True, **kwargs)

            if (
                model_prompts is None
                and hasattr(self.model, "prompts")
                and len(self.model.prompts) > 0
            ):
                try:
                    model_prompts = self.validate_task_to_prompt_name(self.model.prompts)
                except ValueError:
                    model_prompts = None
            elif model_prompts is not None and hasattr(self.model, "prompts"):
                logger.info(f"Model prompts will be overwritten with {model_prompts}")
                self.model.prompts = model_prompts
            self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        def encode(
            self,
            sentences: Sequence[str],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            """Encodes the given sentences using the encoder.

            Args:
                sentences: The sentences to encode.
                task_name: The name of the task. Sentence-transformers uses this to
                    determine which prompt to use from a specified dictionary.
                prompt_type: The name type of prompt. (query or passage)
                **kwargs: Additional arguments to pass to the encoder.

                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)


            Returns:
                The encoded sentences.
            """
            prompt_name = None
            if self.model_prompts is not None:
                prompt_name = self.get_prompt_name(
                    self.model_prompts, task_name, prompt_type
                )
            if prompt_name:
                logger.info(
                    f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
                )
            else:
                logger.info(
                    f"No model prompts found for task={task_name} prompt_type={prompt_type}"
                )
            logger.info(f"Encoding {len(sentences)} sentences.")

            embeddings = []
            if 'prompt' in kwargs:
                prompt = kwargs['prompt']
                logger.info(f'Found prompt: {prompt}')
            else:
                prompt = None

            for sentence in tqdm(sentences):
                if 'instruct' in self.model.model_path.lower():
                    logger.info("Using instruct mode")
                    instruction = self.get_instruction(task_name, prompt_type)

                    if 'qwen2' in self.model.model_path.lower():
                        instruction_template = "Instruct: {instruction}\nQuery: "
                        instruction = instruction_template.format(instruction=instruction)
                        sentence = instruction + sentence
                        logger.info(f'Using instruction: {instruction}')

                    elif 'e5' in self.model.model_path.lower():
                        instruction_template = "Instruct: {instruction}\nQuery: "
                        instruction = instruction_template.format(instruction=instruction)
                        sentence = instruction + sentence
                        logger.info(f'Using instruction: {instruction}')
                elif prompt:
                    sentence = prompt + sentence
                    logger.info(f'Using prompts: {prompt}')
                else:
                    logger.warning('No instructs / prompts found.')

                output = self.model.embed(sentence)

                if len(np.array(output).shape) > 1:
                    logger.info('Taking last embedding.')
                    if 'instruct' not in self.model.model_path.lower():
                        logger.info(f'Shape: {np.array(output).shape}')
                        logger.error("Non instruct model giving non token-level preds.")
                    output = self.model.embed(sentence)[-1]
                else:
                    output = self.model.embed(sentence)
                    logger.info("Output is 1d array.")
                embeddings.append(output)

            if isinstance(embeddings, torch.Tensor):
                # sometimes in kwargs can be return_tensors=True
                embeddings = embeddings.cpu().detach().float().numpy()
            return embeddings

    return GGUFWrapper(model_path, model_prompts, **kwargs)
