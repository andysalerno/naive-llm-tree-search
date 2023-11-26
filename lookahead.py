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
        token_score_mapping: dict[str, NumberType],
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
        spaces = " " * depth
        print(
            f"    {spaces}evaluating sequence: '{node.text}' with score: {node.score}"
        )

        if depth > max_depth:
            print(f"    {spaces}(end of sequence)")
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

    def _get_score_for_tokenized_input(self, input_ids, n: int) -> (Tensor, Tensor):
        """Given tokenized text, returns a tuple of (topk_values, topk_indices)"""
        print(f"testing ids: {input_ids}")
        with torch.no_grad():
            outputs = self.model.forward(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        comma_token = (
            self.tokenizer.encode(",", return_tensors="pt", add_special_tokens=False)
            .squeeze()
            .item()
        )
        print(f"comman token id: {comma_token}")
        comma_token_score = next_token_logits[:, 28725].item()
        print(f"comma token id: {28725}, value: {comma_token_score}")

        correct_next_token = self.tokenizer.convert_ids_to_tokens(
            28725, skip_special_tokens=False
        )
        print(f'correct value: "{correct_next_token}"')

        topk_values, topk_indices = torch.topk(next_token_logits, n)
        topk_values = topk_values[0]
        topk_indices = topk_indices[0]

        return (topk_values, topk_indices)

    def _get_global_token_mapping(self):
        if self._global_token_mapping is None:
            # todo: determine range
            self._global_token_mapping = self.tokenizer.convert_ids_to_tokens(
                range(32002), skip_special_tokens=False
            )

        return self._global_token_mapping

    def _get_next_token_scores(self, text: str) -> dict[str, NumberType]:
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        ).to("cuda")

        topk_values, topk_indices = self._get_score_for_tokenized_input(input_ids, 1)
        print(
            f"(1 only) topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )
        topk_values, topk_indices = self._get_score_for_tokenized_input(input_ids, 1)
        print(
            f"(1 only) topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )
        topk_values, topk_indices = self._get_score_for_tokenized_input(input_ids, 1)
        print(
            f"(1 only) topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )
        topk_values, topk_indices = self._get_score_for_tokenized_input(
            input_ids, self.test_top_n_tokens
        )
        print(
            f"topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )
        topk_values, topk_indices = self._get_score_for_tokenized_input(
            input_ids, self.test_top_n_tokens
        )
        print(
            f"topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )
        topk_values, topk_indices = self._get_score_for_tokenized_input(
            input_ids, self.test_top_n_tokens
        )
        print(
            f"topk_values: {topk_values} topk_indices: {topk_indices} for text '{text}'"
        )

        topk_tokens = self.tokenizer.convert_ids_to_tokens(
            topk_indices, skip_special_tokens=False
        )

        # combine tokens and scores into a mapping { token: score }
        token_score_mapping = {
            token.replace("▁", " "): score.item()
            for token, score in zip(topk_tokens, topk_values)
        }

        print(f"    found mapping: {token_score_mapping} for text: '{text}'")

        return token_score_mapping

        # token healing: back up by one token, get scores again, but this time for tokens that have the final token as a prefix
        # back up by one:
        last_id = input_ids[-1]
        input_ids = input_ids[:, :-1]

        # get scores again:
        # todo: these can be cached from previous executions
        topk_values, topk_indices = self._get_score_for_tokenized_input(
            input_ids, n=32002
        )

        # limit the output to only tokens that have some overlap with the final token
        # ... first, get a mapping of { token: score } for the backed up position:
        topk_tokens = self.tokenizer.convert_ids_to_tokens(
            topk_indices, skip_special_tokens=False
        )
        token_score_mapping_prefix = {
            token.replace("▁", " "): score.item()
            for token, score in zip(topk_tokens, topk_values)
        }

        # prune the mapping to only tokens that have the previous final token as a prefix:
        # todo: is only a 'full overlap' possible, or can 'partial overlap' be possible too?
        last_token_text = self.tokenizer.convert_ids_to_tokens(
            last_id, skip_special_tokens=False
        )[-1].replace("▁", " ")
        print(f"        last token from input is: '{last_token_text}'")

        # todo: how do we differentiate between these two scenarios:
        # 1. the next token is correctly identical to the previous token, e.x. 196 -> 1966
        # 2. the next token was simply re-selecting the previous token, i.e. notorious -> ious
        tokens_with_prefix = {
            k: v
            for (k, v) in token_score_mapping_prefix.items()
            if k.startswith(last_token_text)
            and k
            != last_token_text  # uh oh, this is wrong.. e.x. won't select from 196 -> 1966.
        }
        print(f"        matching tokens with prefix: {tokens_with_prefix}")

        # combine the overlap tokens into the existing mapping:
        token_score_mapping.update(tokens_with_prefix)

        # trim back down to the top N again:
        # todo: is it valid to compare scores from backup with scores from non-backup?
        sorted_items = sorted(
            token_score_mapping.items(), key=lambda x: x[1], reverse=True
        )
        top_entries = dict(sorted_items[: self.test_top_n_tokens])

        # bug: if the value select is from the 'healing' section, it can't be simply appended.
        # it must be merged.
        return top_entries

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
