# The code has been taken from Sebastian Raschka's LLM from scratch book's repo.
# But it's been modified to work with jax/flax.
# It downloads pretrained gpt2 weights from open, loads them into out gpt2 model
# and generates text using a sample prompt.
# That being said, Please go buy his book, it's a masterpiece!
# Also, below is the original license. please don't sue me @rasbt

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
from flax import nnx
from gpt2 import GPT2
import tiktoken
import os
import urllib.request
# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    # Send a GET request to download the file

    try:
        with urllib.request.urlopen(url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # Define the block size for reading the file
            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(url)  # Extract filename from URL
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar
    except urllib.error.HTTPError:
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    return params



def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Shape of left and right must be equal. "
                         f"Left: {left.shape}, Right: {right.shape}")
    return nnx.Param(jnp.asarray(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.embedding = assign(gpt.pos_emb.embedding, params['wpe'])
    gpt.token_emb.embedding = assign(gpt.token_emb.embedding, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transf_blocks[b].attn.query_proj.kernel = assign(
            gpt.transf_blocks[b].attn.query_proj.kernel, q_w)
        gpt.transf_blocks[b].attn.key_proj.kernel = assign(
            gpt.transf_blocks[b].attn.key_proj.kernel, k_w)
        gpt.transf_blocks[b].attn.value_proj.kernel = assign(
            gpt.transf_blocks[b].attn.value_proj.kernel, v_w)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transf_blocks[b].attn.query_proj.bias = assign(
            gpt.transf_blocks[b].attn.query_proj.bias, q_b)
        gpt.transf_blocks[b].attn.key_proj.bias = assign(
            gpt.transf_blocks[b].attn.key_proj.bias, k_b)
        gpt.transf_blocks[b].attn.value_proj.bias = assign(
            gpt.transf_blocks[b].attn.value_proj.bias, v_b)

        gpt.transf_blocks[b].attn.out_proj.kernel = assign(
            gpt.transf_blocks[b].attn.out_proj.kernel,
            params["blocks"][b]["attn"]["c_proj"]["w"])
        gpt.transf_blocks[b].attn.out_proj.bias = assign(
            gpt.transf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transf_blocks[b].ff.l1.kernel = assign(
            gpt.transf_blocks[b].ff.l1.kernel,
            params["blocks"][b]["mlp"]["c_fc"]["w"])
        gpt.transf_blocks[b].ff.l1.bias = assign(
            gpt.transf_blocks[b].ff.l1.bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transf_blocks[b].ff.l2.kernel = assign(
            gpt.transf_blocks[b].ff.l2.kernel,
            params["blocks"][b]["mlp"]["c_proj"]["w"])
        gpt.transf_blocks[b].ff.l2.bias = assign(
            gpt.transf_blocks[b].ff.l2.bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transf_blocks[b].norm1.scale = assign(
            gpt.transf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transf_blocks[b].norm1.bias = assign(
            gpt.transf_blocks[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transf_blocks[b].norm2.scale = assign(
            gpt.transf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transf_blocks[b].norm2.bias = assign(
            gpt.transf_blocks[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"])

    gpt.last_norm.scale = assign(gpt.last_norm.scale, params["g"])
    gpt.last_norm.bias = assign(gpt.last_norm.bias, params["b"])
    gpt.out_linear.kernel = assign(gpt.out_linear.kernel, params["wte"].T)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = jnp.expand_dims(jnp.asarray(encoded), 0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(key, model, idx, max_new_tokens, vocab_size, context_size, temperature=0.0, top_k=None, eos_id=None):
    key, subkey = random.split(key, 2)
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = lax.top_k(logits, top_k)
            min_val = top_logits[:, -1]
            logits = jnp.where(logits < min_val, -jnp.inf, logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = jax.nn.softmax(logits, axis=-1)  # (batch_size, context_len)

            # Sample from the distribution
            key, subkey = random.split(subkey, 2)
            # idx_next = random.choice(key, a=context_size,
            #                          p=probs, shape=(logits.shape[0], 1))  # (batch_size, 1)
            idx_next = jax.vmap(
                lambda key, p: random.choice(key, a=vocab_size, p=p),
                in_axes=(0, 0),
            )(random.split(key, logits.shape[0]), probs)
            idx_next = jnp.expand_dims(idx_next, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = jnp.argmax(logits, axis=-1, keepdims=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = jnp.concat((idx, idx_next), axis=1)  # (batch_size, num_tokens+1)

    return idx


def main(input_prompt, model_size="124M"):
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPT2(rngs=nnx.Rngs(1999))
    load_weights_into_gpt(gpt, params)

    tokenizer = tiktoken.get_encoding("gpt2")
    key = random.key(2005)
    token_ids = generate(
        key,
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer),
        max_new_tokens=25,
        vocab_size=50257,
        context_size=1024,
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

main("Every effort moves you")