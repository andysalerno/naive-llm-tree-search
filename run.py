from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from beam_search import BeamSearchSampler
from greedy import GreedySampler
import torch

from lookahead import LookAheadSampler

quant_path = "TheBloke/openchat_3.5-AWQ"

# Load model
# model = AutoAWQForCausalLM.from_quantized(
model = AutoModelForCausalLM.from_pretrained(
    quant_path,
    return_dict_in_generate=True,
    use_flash_attention_2=True,
    device_map="cuda:0",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)


print(f"tokenizer eos: {tokenizer.eos_token}")

test_conversation = [
    {
        "role": "system",
        "content": "You are a highly capable assistant.",
    },
    {
        "role": "user",
        "content": "List the ablums released by The Beatles",
    },
]


TEMPLATE = "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}"
chat_text = tokenizer.apply_chat_template(
    test_conversation,
    add_generation_prompt=True,
    tokenize=False,
    chat_template=TEMPLATE,
)


def test_with_greedy(input: str, max_new_tokens: int):
    # inputs = "A list of colors: red, blue"
    strategy = GreedySampler()

    inputs = [input]

    for _ in range(max_new_tokens):
        current_sequence = "".join(inputs)

        if current_sequence.endswith(tokenizer.eos_token):
            break

        input_ids = tokenizer.encode(
            current_sequence, return_tensors="pt", add_special_tokens=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.forward(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            # TOP = len(next_token_logits[0])
            TOP = 3

            topk_values, topk_indices = torch.topk(next_token_logits, TOP)
            topk_values = topk_values[0]
            topk_indices = topk_indices[0]

            topk_tokens = tokenizer.convert_ids_to_tokens(
                topk_indices, skip_special_tokens=False
            )
            token_score_mapping = {
                token: score.item() for token, score in zip(topk_tokens, topk_values)
            }

            next_tokens = strategy.select_next_tokens(
                token_score_mapping, current_sequence
            )

            next_tokens = "".join(next_tokens)

            # hack: why is this necessary?
            next_tokens = next_tokens.replace("▁", " ")

            print(f"selected token: '{next_tokens}'")

            inputs.append(next_tokens)
            next_inputs = "".join(inputs)
            print(f"next inputs: '{next_inputs}'")

    return "".join(inputs)


def test_with_lookahead(input: str, max_new_tokens: int):
    inputs = [input]

    strategy = LookAheadSampler(model, tokenizer)

    for _ in range(max_new_tokens):
        current_sequence = "".join(inputs)

        if current_sequence.endswith(tokenizer.eos_token):
            break

        print("generating next logits...")
        with torch.no_grad():
            next_tokens = strategy.select_next_tokens(current_sequence)

            # hack: why is this necessary?
            next_tokens = next_tokens.replace("▁", " ")

            print(f"selected token: '{next_tokens}'")

            inputs.append(next_tokens)

            print(f"next sequence: '{inputs}'")

    return "".join(inputs)


def test_with_beam_search(input: str, max_new_tokens: int):
    strategy = BeamSearchSampler(model, tokenizer)

    for _ in range(max_new_tokens):
        current_sequence = input

        if current_sequence.endswith(tokenizer.eos_token):
            break

        print("generating next logits...")
        with torch.no_grad():
            next_tokens = strategy.select_next_tokens(current_sequence)

            # hack: why is this necessary?
            next_tokens = next_tokens.replace("▁", " ")

            print(f"selected token: '{next_tokens}'")

            input = next_tokens

            print(f"next sequence: '{input}'")

    return input


def main():
    SEED = 42
    MAX_NEW_TOKENS = 128
    INPUT = "The Beatles were"
    set_seed(SEED)
    print(f"set seed to: {SEED}")

    print(f"chat text: {chat_text}")

    # print("Testing with: GreedySampler")
    # greedy_result = test_with_greedy(chat_text, MAX_NEW_TOKENS)
    # print(f"greedy result: {greedy_result}")

    # print("Testing with: LookAheadSampler")
    # lookahead_result = test_with_lookahead(chat_text, MAX_NEW_TOKENS)
    # print(f'lookahead result: "{lookahead_result}"')

    print("Testing with: BeamSearch")
    beam_search_result = test_with_beam_search(chat_text, MAX_NEW_TOKENS)
    print(f'beam search result: "{beam_search_result}"')

    print("final results:")
    # print(f"greedy result: '{greedy_result}'")
    # print(f"lookahead result: '{lookahead_result}'")
    print(f"beam search result: '{beam_search_result}'")


if __name__ == "__main__":
    main()
