import os
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

def load_dataset(file_path, tokenizer, block_size=128):

    with open(file_path, "r") as file:
        lines = file.readlines()

    inputs = tokenizer(lines, return_tensors="tf", max_length=block_size, truncation=True, padding="max_length")

    dataset = tf.data.Dataset.from_tensor_slices((inputs["input_ids"], inputs["attention_mask"]))
    return dataset

def train_model(train_file, model_output_dir):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    train_dataset = load_dataset(train_file, tokenizer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss)

    model.fit(train_dataset.batch(4), epochs=3)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

if __name__ == "__main__":

    train_file = "../data/interactions.txt"
    model_output_dir = "../models/trained_model"

    train_model(train_file, model_output_dir)
