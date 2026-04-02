# ECE9660 Mixtral Of Experts

## Models Used

### Mixtral 8x7B API

The Mixtral 8x7B API was used because there was a limiting in our own computer hardware for us to run the full Mixtral 8x7B model locally. The API provides a free version of their model, so we can run a more faithfully.

There is one major difference in our replication, which is the hellaswag result as it is significantly lower than the paper, 24.37% vs 81.9%. The reason for the difference could be due to the API having some instruction tuning applied, causing some difference between its response compared to a plain text prompt.

### mlx-community/Mistral-7B-v0.1-q

https://huggingface.co/mlx-community/Mistral-7B-v0.1-q

Hugging face was used to obtain a quantized version of the model for apple hardware as inference for these tests were ran on an apple device (16 GB RAM). 

Runs `mlx-community/Mistral-7B-v0.1-q` via HuggingFace and evaluates using logit-based scoring to match the paper's methodology

This portion is a replication of the Mistral 7B row from Table 2 of [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024).

One script per benchmark and a script to control them (run_evals.py) with results in the readme and the results folder. 
