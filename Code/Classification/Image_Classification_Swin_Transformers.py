
import os
import pandas as pd
import torch

from datasets import load_dataset
from datasets import load_metric 

from transformers import AutoImageProcessor

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


model_checkpoint = "/home/chayan/MIMIC-Dataset/Codes/swin-base-patch4-window12-384-finetuned-mimic-filtered-linear-v2/" # pre-trained model from which to fine-tune ##microsoft/swinv2-base-patch4-window8-256 ## microsoft/swin-base-patch4-window12-384


dataset = load_dataset("imagefolder", data_dir="/mnt/data/chayan/Mini-MIMIC/Train")

print('Total Training Data: ', len(dataset))

metric = load_metric("accuracy")

labels = dataset["train"].features["label"].names

label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
  label2id[label] = i
  id2label[i] = label
print(id2label[0])


image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")
    
print("Image_resolution:",crop_size)

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    
val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
    
    
    
# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)




model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

model_name = model_checkpoint.split("/")[-1]


# Define your training arguments
num_epochs = 25  # Set the total number of epochs
warmup_proportion = 0.1  # Set the proportion of warmup steps relative to total steps
train_batch_size = 16  # Set your training batch size
test_batch_size = 16
gradient_accumulation_steps=8
# Calculate the total number of training steps
total_training_steps = len(train_ds) // train_batch_size * num_epochs
global_steps = total_training_steps // gradient_accumulation_steps


# Calculate the number of warmup steps
num_warmup_steps = int(warmup_proportion * global_steps)

args = TrainingArguments(
    f"{model_name}-finetuned-mimic-filtered-Linear-v3",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    #lr_scheduler_type = "cosine_with_restarts",
    #lr_scheduler_kwargs = { "num_cycles": 2 },
    #warmup_steps = num_warmup_steps,
    #optim = "adamw_hf",
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=test_batch_size,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    #report_to="wandb",
    # push_to_hub=True,
)

import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
## define a collate_fn, which will be used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


train_results = trainer.train()


trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)