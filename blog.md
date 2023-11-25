## sources
- https://huggingface.co/blog/how-to-generate
    - covers greedy, beam, top_k, top_p sampling
- https://huggingface.co/docs/transformers/llm_tutorial


## ordering of thoughts
- tried max sum, weird results
- tried log, better results
- suspect problems with token healing because of '1' not becoming '1960'