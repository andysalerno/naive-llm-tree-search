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
    inputs = "A list of colors: red, blue"

    model.eval()

    while True:
        print("tokenizing inputs...")
        input_ids = tokenizer.encode(inputs, return_tensors="pt").to("cuda")

        print("generating next logits...")
        with torch.no_grad():
            outputs = model.forward(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            TOP = 100
            # TOP = len(next_token_logits[0])

            topk_values, topk_indices = torch.topk(next_token_logits, TOP)
            topk_values = topk_values[0]
            topk_indices = topk_indices[0]

            topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)
            token_score_mapping = {
                token: score.item() for token, score in zip(topk_tokens, topk_values)
            }
            print(f"mapping: {token_score_mapping}")

            next_token = strategy.select_next_token(token_score_mapping)

            print(f"selected token: {next_token}")

            return

        return

        next_token = strategy.get_next_token()

        if next_token is False:
            return


def main():
    SEED = 42
    set_seed(SEED)
    print(f"set seed to: {SEED}")

    test_with_strategy(GreedySampler(model, tokenizer))


if __name__ == "__main__":
    main()
