# Learning GPT
 - Generative Pre-trained Transformer.
 - A language model trained with both supervised and reinforcement learning. 
 
# Key Concept
## tokeniser
 - tiktoken: openAI [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser: https://github.com/openai/tiktoken
 - sentencepiece: google's unsupervised text tokenizer and detokenizer : https://github.com/google/sentencepiece
 - string <-> int
   - string could be char, word, sub-word.
   - int: [0, vocabulary]
   - example: openAI gpt2: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json

# batch/mini-batch
utilize the vector computation, optimize the CPU/GPU/Memory usage, and through put.

## context
 - option1: fixed size rolling window, partial at the beginning
 - option2: dynamic range w/ max(fixed) size - GPT
  - samples

# tensor
multi-dimentional data object.
 - shape. e.g [B, T, C]

# embedding
 - tensor look up table by row index.

# self-attention
 - attention is the communication mechanism;
 - compare with convolution nn, it preseved the `space` info, filter is sequencial moved through the `space`, and result is put back to the `same space`.
   Attention has no `space` info, have to manually encode the position info.
 - attention is within the `batch`, no cross the batches.
 - each token has an embedding, it `combines` token self-embedding + position embedding. [B, T, C].
 - key: `what do I have`. Computed via key-head([C, head_size]) applied to token embedding, [B, T, C] => [B, T, head_size]
 - query: `what am I looking for`. Computed via query-head([C, head_size]) applied to token embedding, [B, T, C] => [B, T, head_size]
 - weight: query @ key, [B, T, head_size]@[B, head_size, T].transpose(-2, -1) => [B, T, T]. 
 - value: 


# References
- nanoGPT: https://github.com/karpathy/nanoGPT
