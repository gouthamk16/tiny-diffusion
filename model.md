## TON-v1 - Score Entropy Discrete Diffusion Model for Text Generation

Usual transformer models are autoregressive, i.e., it generates one output tokens given a set of input tokens. They make use of causal attention, which masks the sucessive tokens and only considers the tokens (context) until the current token being processed. A text diffusion model works differently, it is not an autoregressive model, it generates a set of output tokens given a set of input tokens. We add an amount of noise to the input tokens and the model predicts the original tokens in that position (similar to how ddpm works). 

We did some experiments with both SEDD and MDLM. MDLM turned out to be better in terms of perplexity and semantic contextual quality compared to SEDD. Obviously this is a very small model and there might be other ways to implement SEDD that might turn out to be better than MDLM. 

My aim was to try and implement a small text diffusion model and run performance and quality benchmarks on it with minimal training on a mediocre mobile GPU.