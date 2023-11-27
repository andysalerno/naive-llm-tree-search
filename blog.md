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

## Monte Carlo Tree Search

There 

## sources
- https://huggingface.co/blog/how-to-generate
    - covers greedy, beam, top_k, top_p sampling
- https://huggingface.co/docs/transformers/llm_tutorial


## ordering of thoughts
- tried max sum, weird results
- tried log, better results
- suspect problems with token healing because of '1' not becoming '1960'