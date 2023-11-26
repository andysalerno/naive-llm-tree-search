from transformers import (
    set_seed,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)
from awq import AutoAWQForCausalLM
from greedy import GreedySampler
import torch

from lookahead import LookAheadSampler

quant_path = "TheBloke/openchat_3.5-AWQ"

# Load model
# model = AutoAWQForCausalLM.from_quantized(
model = AutoModelForCausalLM.from_pretrained(
    quant_path,
    return_dict_in_generate=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)


print(f"tokenizer eos: {tokenizer.eos_token}")

test_conversation = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a thug",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]


def test_with_strategy(strategy, max_new_tokens: int):
    # inputs = "A list of colors: red, blue"
    inputs = ["The Beatles were"]

    for _ in range(max_new_tokens):
        print("tokenizing inputs...")
        current_sequence = "".join(inputs)
        # input_ids = tokenizer.encode(
        #     current_sequence, return_tensors="pt", add_special_tokens=True
        # ).to("cuda")

        print("generating next logits...")
        with torch.no_grad():
            # print(f"testing ids: {input_ids}")
            # outputs = model.forward(input_ids)
            # next_token_logits = outputs.logits[:, -1, :]

            # # TOP = len(next_token_logits[0])
            # TOP = 3

            # topk_values, topk_indices = torch.topk(next_token_logits, TOP)
            # topk_values = topk_values[0]
            # topk_indices = topk_indices[0]
            # print(f"topk_values: {topk_values} topk_indices: {topk_indices}")

            # topk_tokens = tokenizer.convert_ids_to_tokens(
            #     topk_indices, skip_special_tokens=False
            # )
            # token_score_mapping = {
            #     token: score.item() for token, score in zip(topk_tokens, topk_values)
            # }

            # print(f"original token_score_mapping: {token_score_mapping}")

            next_tokens = strategy.select_next_tokens(None, current_sequence)

            # next_tokens = "".join(next_tokens)
            # hack: why is this necessary?
            next_tokens = next_tokens.replace("▁", " ")

            print(f"selected token: '{next_tokens}'")

            inputs.append(next_tokens)

            print(f"next sequence: '{inputs}'")

        if next_tokens is []:
            return


def main():
    SEED = 42
    MAX_NEW_TOKENS = 128
    set_seed(SEED)
    print(f"set seed to: {SEED}")

    # print("Testing with: GreedySampler")
    # test_with_strategy(GreedySampler(), MAX_NEW_TOKENS)

    print("Testing with: LookAheadSampler")
    test_with_strategy(LookAheadSampler(model, tokenizer), MAX_NEW_TOKENS)


if __name__ == "__main__":
    main()
