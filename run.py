from transformers import set_seed, AutoTokenizer
from awq import AutoAWQForCausalLM
from greedy import GreedySampler
import torch

quant_path = "TheBloke/openchat_3.5-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=True, return_dict_in_generate=True
)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)


print(f"tokenizer eos: {tokenizer.eos_token}")

test_conversation = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a thug",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]


def test_with_strategy(strategy):
    # inputs = "A list of colors: red, blue"
    inputs = ["The Beatles were"]

    model.eval()

    while True:
        print("tokenizing inputs...")
        input_ids = tokenizer.encode(
            "".join(inputs), return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        print("generating next logits...")
        with torch.no_grad():
            outputs = model.forward(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            TOP = 100
            # TOP = len(next_token_logits[0])

            topk_values, topk_indices = torch.topk(next_token_logits, TOP)
            topk_values = topk_values[0]
            topk_indices = topk_indices[0]

            topk_tokens = tokenizer.convert_ids_to_tokens(
                topk_indices, skip_special_tokens=True
            )
            token_score_mapping = {
                token: score.item() for token, score in zip(topk_tokens, topk_values)
            }

            next_token = strategy.select_next_token(token_score_mapping)

            # hack: why is this necessary?
            next_token = next_token.replace("_", " ")

            print(f"selected token: {next_token}")

            inputs.append(next_token)

            # even with 'skip_special_tokens=True' it is emitting an underscore instead of spaces?
            print(f"next sequence: {inputs}")

        if next_token is False:
            return


def main():
    SEED = 42
    set_seed(SEED)
    print(f"set seed to: {SEED}")

    test_with_strategy(GreedySampler(model, tokenizer))


if __name__ == "__main__":
    main()
