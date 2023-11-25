from torch import NumberType
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import torch


class _Node:
    def __init__(self, text=None, score=None, parent=None):
        self.text: str = text
        self.score: float = score
        self.children: [_Node] = []
        self.parent: _Node = parent


class LookAheadSampler:
    def __init__(self, model: AutoAWQForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # how many top tokens to evaluate
        self.test_top_n_tokens = 3

        # for each top token, how deep should it be evaluated
        self.test_depth = 2

    def select_next_tokens(
        self,
        token_score_mapping: dict[str, NumberType],
        current_sequence: str,
    ) -> str:
        root = _Node()
        root.text = current_sequence
        root.score = 0

        (best_score, best_sequence) = self._evaluate(
            depth=0, max_depth=self.test_depth, node=root
        )

        return best_sequence

    def _evaluate(self, depth: int, max_depth: int, node: _Node):
        if depth > max_depth:
            return (node.score, node.text)

        current_sequence = []
        cur_node = node
        while cur_node is not None:
            current_sequence.append(cur_node.text)
            cur_node = cur_node.parent

        current_sequence.reverse()
        current_sequence = "".join(current_sequence)

        print(f"evaluating sequence: {current_sequence}")

        # get the mappings of token: value for the top N next tokens
        next_token_scores = self._get_next_token_scores(current_sequence)

        max_child_score = -99999
        max_child_sequence = None

        for k, v in next_token_scores.items():
            text = current_sequence + k
            score = node.score + v
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

        outputs = self.model.forward(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        topk_values, topk_indices = torch.topk(
            next_token_logits, self.test_top_n_tokens
        )
        topk_values = topk_values[0]
        topk_indices = topk_indices[0]

        topk_tokens = self.tokenizer.convert_ids_to_tokens(
            topk_indices, skip_special_tokens=False
        )
        token_score_mapping = {
            token.replace("▁", " "): score.item()
            for token, score in zip(topk_tokens, topk_values)
        }

        return token_score_mapping

    def _get_top_n(
        token_score_mapping: dict[str, NumberType], n: int
    ) -> dict[str, NumberType]:
        """Returns a dictionary of the top N token: score mappings"""
        # Sort the dictionary by its values in decreasing order and convert it back to a dictionary
        sorted_dict = dict(
            sorted(token_score_mapping.items(), key=lambda item: item[1], reverse=True)
        )
        # Use list slicing to get the first 'n' items from the sorted dictionary
        top_n_dict = dict(list(sorted_dict.items())[:n])

        return top_n_dict
