# Experimenting with simple tree-search techniques for LLM token sampling

At its core, an LLM is a value function.

Given a state (i.e. the current context, or input text), it scores all possible next actions (i.e. tokens).

Therefore, it's pretty simple to imagine the task of sampling tokens as a state-space exploration problem, where we use a tree data structure to map out the state space and explore it to find high-scoring states.

For LLMs, the most common sampling technique is a naive greedy approach - simply take the next token with the highest score, every time.

Well, that's not entirely true - there are sampling techniques like top_k, top_p, etc, which are strategies used to probe the LLM down paths that would not otherwise be taken in a purely greedy approach. top_k will pick the top K scoring tokens, and sample from them using their scores as a probability distribution. top_p will sample from the top N tokens, where N is the smallest amount of tokens that reach a certain probability threshhold, and then pick from that set.

And yet, neither top_k nor top_p explore any further than the very next token. You could say they have "depth" value of 1, since they never look further down the state-space tree than one step.

What are some strategies for exploring further down the tree? Can we explore a bit deeper, and possibly find a future state that has a higher total score? A higher total score across all tokens should imply a better resonse from the LLM, right?

## Beam Search

Beam search is a strategy that is already supported by the Huggingface Transformers library.

Beam search will fire off multiple concurrent traversals down the tree, and when all N traversals reach a terminal state (i.e., the EOS, or end-of-stream token), it selects the beam that resulted in the highest score.

There are some more advanced flavors of beam search, which you can read about here.

There's one major downside to beam search, however:

It uses **far** more memory, which is a problem when using consumer GPUs with paltry vram. To put this into perspective, when I use beam search with 3 beams on a 7B Mistral model, quantized to 4bits with AWQ, I've seen my VRAM balloon to 18GB (out of the 24GB on my 3090).

## A Naive Tree Search Approach

There's an alternative to beam search, which uses almost no extra memory, at the expense of a bit more latency.

The concept is almost exactly the same as beam search, except instead of running the beams concurrently, we have a only single worker exploring the state space. This means we use almost no additional memory (again, at the expense of latency).

The implementation looks like something you'd see out of your traditional programming interview, if the question was: "Given a root node of a tree, find the subtree of length N that has the highest total value".

The implementation is so simple that I can show almost the entire thing right here:

```python
def _evaluate(self, depth: int, max_depth: int, node: _Node) -> Tuple[float, str]:
    if depth > max_depth:
        return (node.score, node.text)

    # give more weight to later states:
    # modifier = math.log(depth + 2)
    modifier = 1

    # get the mappings of token: value for the top N next tokens
    # this is where we actually get the token scores from the LLM
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
```

In this approach, instead of picking the highest-scoring token and calling it a day, we explore the tree with a depth of N (configurable) and find the next N tokens that have the total highest score.

### Monte Carlo Tree Search

Another interesting tree search strategy is Monte Carlo Tree Search.


## Results

If you're curious, here are some outputs, all for the exact same input prompt, but using different strategies.

**Prompt:**

*sort the list [4, 5, 28, 12, 343, 29, 199, 404, 3, 101, 73] in ascending numeric order. Your answer must only include the sorted list, no additional text.*

Note: the model used is openchat3.5, AWQ quantized from TheBloke. The openchat3.5 chat template is used to convert the above prompt into the model's expected chat format.

**Expected result:**

[3, 4, 5, 12, 28, 29, 73, 101, 199, 343, 404]

### Greedy Search Result
[3, 4, 5, 12, 28, 29, 343, 101, 199, 404, 343]

max vram usage: 4.6GB

### Beam Search (3 beams) Result
[3, 4, 5, 12, 28, 29, 101, 199, 343, 404]

max vram usage: 5.1GB

### Tree search (Depth=3, topK=3) Result
[3, 4, 5, 12, 28, 29, 73, 101, 199, 343, 404]

max vram usage: 4.6GB


## sources
- https://huggingface.co/blog/how-to-generate
    - covers greedy, beam, top_k, top_p sampling
- https://huggingface.co/docs/transformers/llm_tutorial


## ordering of thoughts
- tried max sum, weird results
- tried log, better results
- suspect problems with token healing because of '1' not becoming '1960'