from torch import NumberType, Tensor
from transformers import PreTrainedTokenizer
from awq.models.base import BaseAWQForCausalLM
import torch


class _Node:
    def __init__(self, text=None, score=None, parent=None):
        self.text: str = text
        self.score: float = score
        self.children: [_Node] = []
        self.parent: _Node = parent
        self.tokens: [str] = []


class LookAheadSampler:
    def __init__(self, model: BaseAWQForCausalLM, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # how many top tokens to evaluate
        self.test_top_n_tokens = 3

        # for each top token, how deep should it be evaluated
        self.test_depth = 2

    def select_next_tokens(
        self,
        current_sequence: str,
    ) -> str:
        root = _Node()
        root.text = current_sequence
        root.score = 0

        (best_score, best_sequence) = self._evaluate(
            depth=0, max_depth=self.test_depth, node=root
        )

        print(f'selecting sequence: "{best_sequence}"')

        return best_sequence.removeprefix(current_sequence)

    def _evaluate(self, depth: int, max_depth: int, node: _Node):
        if depth > max_depth:
            return (node.score, node.text)

        # give more weight to later states:
        # modifier = math.log(depth + 2)
        modifier = 1

        # get the mappings of token: value for the top N next tokens
        next_token_scores = self._get_next_token_scores(node.text)

        max_child_score = -float("inf")
        max_child_sequence = None

        for k, v in next_token_scores.items():
            text = node.text + k
            score = node.score + (modifier * v)
            child = _Node(text, score=score, parent=node)
            node.children.append(child)

            (child_best_score, child_best_sequence) = self._evaluate(
                depth + 1, max_depth, child
            )

            if child_best_score > max_child_score:
                max_child_score = child_best_score
                max_child_sequence = child_best_sequence

        return (max_child_score, max_child_sequence)

    def _get_next_token_scores(self, text: str) -> dict[str, NumberType]:
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        ).to("cuda")

        topk_values, topk_indices = self._get_score_for_tokenized_input(
            input_ids, self.test_top_n_tokens
        )

        topk_tokens = self.tokenizer.convert_ids_to_tokens(
            topk_indices, skip_special_tokens=False
        )

        # combine tokens and scores into a mapping { token: score }
        token_score_mapping = {
            token.replace("▁", " "): score.item()
            for token, score in zip(topk_tokens, topk_values)
        }

        return token_score_mapping

    def _get_score_for_tokenized_input(self, input_ids, n: int) -> (Tensor, Tensor):
        """Given tokenized text, returns a tuple of (topk_values, topk_indices)"""
        with torch.no_grad():
            outputs = self.model.forward(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        topk_values, topk_indices = torch.topk(next_token_logits, n)
        topk_values = topk_values[0]
        topk_indices = topk_indices[0]

        return (topk_values, topk_indices)
