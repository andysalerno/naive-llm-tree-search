from torch import NumberType
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM


class GreedySampler:
    def select_next_token(self, token_score_mapping: dict[str, NumberType]) -> str:
        return max(token_score_mapping, key=token_score_mapping.get)
