from torch import NumberType


class GreedySampler:
    def select_next_tokens(
        self, token_score_mapping: dict[str, NumberType], current_sequence: str
    ) -> str:
        return max(token_score_mapping, key=token_score_mapping.get)
