from typing import List, Tuple
from torch import NumberType, Tensor
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig
import torch


class BeamSearchSampler:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def select_next_tokens(self, current_sequence: str) -> str:
        input_ids = self.tokenizer.encode(
            current_sequence, return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        beam_text = ""

        with torch.no_grad():
            config = GenerationConfig(num_beams=1, max_new_tokens=10)
            beam_outputs = self.model.generate(input_ids, generation_config=config)
            print(f"beam count: {len(beam_outputs)}")
            for i, beam_output in enumerate(beam_outputs):
                beam_text = self.tokenizer.decode(
                    beam_output, skip_special_tokens=False
                )
                print("{}: {}".format(i, beam_text))

        return beam_text
