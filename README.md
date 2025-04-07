# DI 725: Transformers and Attention-Based Deep Networks

## An Assignment for Implementing Transformers in PyTorch

This repo is follows the baseline prepared by Andrej Karpathy, (and Ümit Mert Çağlar), customized for sentiment analysis, prepared for DI725 course.

### Author:

* Nisan Yıldız

## Requirements

Install requirements for your environment, comment out for later uses.

Dependencies:

- [pytorch](https://pytorch.org)  
- [numpy](https://numpy.org/install/)  
- `transformers` for huggingface transformers (to load GPT-2 checkpoints)  
- `datasets` for huggingface datasets (to download \+ preprocess datasets)  
- `tiktoken` for OpenAI's fast BPE code  
- `wandb` for optional logging  
- `tqdm` for progress bars

pip install torch numpy transformers datasets tiktoken wandb tqdm

## Quick Start

The dataset we will be using for training and testing consists of customer service conversations and features associated with given conversations such as issue category, issue complexity and experience level of the agent responding to the customer, as well as a human-curated feature of sentiments of the whole conversation, categorized into three classes: negative, neutral, and positive; totaling 11 features in total, including the conversation

Prepare the dataset using  the following:

python data/customer\_service/prepare\_char.py

This creates a `train_sentiment.pkl` and `val_sentiment.pkl` in that data directory. Now it is time to train our own GPT. The size of the GPT model depends on the computational resources. It is advised to have a GPU for heavy works, and to train lightweight and evaluate and infer models with a CPU.

Small scale GPT with the settings provided in the [config/train\_sent\_char.py](https://colab.research.google.com/drive/config/train_sent_char.py) config file will be trained with the following code:

python train.py config/train\_sent\_char.py

We are training a small scaled GPT with a context size of up to 256 characters, 384 feature channels, 6 layers of transformer with 6 attention heads. On one GTX 3070 GPU this training run takes about 10 minutes and the best validation loss is 1.1620. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-sent-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

## Finetuning

Finetuning or transfer learning is a precious method of achieving better models thanks to pre-trained models. Finetuning GPT models is just as simple as training from scratch\! We will use the same dataset but this time we will define it with tokens (using OpenAI's BPE tokenizer) instead of characters.

Run an example finetuning like:

python data/customer\_service\_gpt2/prepare.py

