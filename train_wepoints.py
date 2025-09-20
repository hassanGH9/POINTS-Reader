#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A concise and correct script to fine-tune the POINTSV15 Chat Model using TRL SFTTrainer.

This script implements a feature-extraction approach:
1.  **Freezing Vision Components**: The vision_encoder and vision_projector are frozen by default.
2.  **Efficient Data Preprocessing**: A single `.map()` call prepares the entire dataset, creating
    masked labels and preparing image tensors.
3.  **Essential Forward Pass Override**: A custom SFTTrainer modifies the forward pass to correctly
    inject image features into the text embeddings before calculating the loss. This is a critical
    step for any VLM training.

Example Usage (Single GPU with LoRA):
python train_wepoints.py \
    --model_name_or_path "models/POINTS-Reader" \
    --dataset_name "axolotl-ai-co/llava-instruct-mix-vsft-small" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --output_dir ./pointsv15-sft-lora-output \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --do_eval True \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --save_strategy "steps" \
    --save_steps 30 \
    --bf16 True \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,vision_projector.mlp.0,vision_projector.mlp.2" \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --gradient_checkpointing True \
    --only_one_turn False

# With quantization support (add these flags for 4-bit quantization):
#    --load_in_4bit True \
#    --bnb_4bit_quant_type nf4 \
#    --bnb_4bit_compute_dtype bfloat16
"""
import sys
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                          PreTrainedTokenizer, TrainerCallback,
                          TrainingArguments)
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    ScriptArguments,
    get_kbit_device_map,
    get_quantization_config,
)

@dataclass
class CustomScriptArguments(ScriptArguments):
    """Custom script arguments with additional parameters."""
    only_one_turn: bool = field(
        default=True,
        metadata={"help": "Whether to only keep the first turn of conversation in multi-turn datasets"}
    )

# Import model definition files from the correct path
from configuration_pointsv15_chat import POINTSV15ChatConfig
from modeling_pointsv15_chat import POINTSV15ChatModel

try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
except ImportError:
    print("Please upgrade `transformers` to version >= 4.46.3.")
    sys.exit(1)

# Import WePOINTS dependencies
try:
    from wepoints.models import Qwen2VisionTransformerForNavitPOINTS
except ImportError:
    print("Please install WePOINTS, and refer to https://github.com/WePOINTS/WePOINTS")
    sys.exit(1)


class CustomSFTTrainer(SFTTrainer):
    """
    A custom SFTTrainer that overrides the forward pass to handle VLM inputs.
    """

    def has_valid_pixels(self, pixel_values):
        """Check if pixel_values contain valid image data."""
        if pixel_values is None:
            return False
        if hasattr(pixel_values, 'numel'):  # PyTorch tensor
            return pixel_values.numel() > 0
        elif hasattr(pixel_values, 'size'):  # NumPy array
            return pixel_values.size > 0
        elif hasattr(pixel_values, '__len__'):  # List
            return len(pixel_values) > 0
        return False

    def inject_image_features(self, input_ids, pixel_values, image_grid_thws, model):
        """Unified image feature injection logic for both training and evaluation.

        Args:
            input_ids: Token IDs with image placeholder tokens
            pixel_values: Processed image pixel values
            image_grid_thws: Image grid dimensions
            model: The VLM model

        Returns:
            inputs_embeds: Text embeddings with image features injected
        """
        # Get base text embeddings
        inputs_embeds = model.llm.get_input_embeddings()(input_ids)

        # Check if we have valid pixel data
        if not self.has_valid_pixels(pixel_values):
            return inputs_embeds

        # Convert to tensors if needed
        import numpy as np
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values).to(model.device)
        if isinstance(image_grid_thws, np.ndarray):
            image_grid_thws = torch.from_numpy(image_grid_thws).to(model.device)

        # Fix shape: from [1, 1, 3] to [1, 3]
        if image_grid_thws.dim() == 3 and image_grid_thws.shape[1] == 1:
            image_grid_thws = image_grid_thws.squeeze(1)

        # Process images through vision components
        image_features = model.vision_encoder(pixel_values, grid_thw=image_grid_thws)
        image_features = model.vision_projector(image_features.to(torch.bfloat16))

        # Find and replace image token embeddings
        image_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        batch_size = input_ids.shape[0]

        # Clone to avoid in-place modification issues
        inputs_embeds = inputs_embeds.clone()

        # Track current position in image_features
        image_feature_idx = 0

        for i in range(batch_size):
            image_token_indices = torch.where(input_ids[i] == image_pad_token_id)[0]
            if image_token_indices.numel() > 0:
                num_features = len(image_token_indices)
                if image_feature_idx + num_features <= image_features.shape[0]:
                    inputs_embeds[i, image_token_indices] = image_features[image_feature_idx:image_feature_idx+num_features].to(inputs_embeds.dtype)
                    image_feature_idx += num_features

        return inputs_embeds

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using unified image feature injection."""
        # Pop image-related data from inputs
        pixel_values = inputs.pop("pixel_values", None)
        image_grid_thws = inputs.pop("image_grid_thws", None)
        input_ids = inputs.get("input_ids")

        # Use unified image feature injection
        inputs_embeds = self.inject_image_features(input_ids, pixel_values, image_grid_thws, model)

        # Forward pass
        inputs["input_ids"] = None
        inputs["inputs_embeds"] = inputs_embeds
        outputs = model.llm(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """Prediction step - 完全复用compute_loss的核心逻辑。"""
        with torch.no_grad():
            if prediction_loss_only:
                # 只需要损失值
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            else:
                # 需要完整的预测结果
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs.logits if hasattr(outputs, 'logits') else None
                labels = inputs.get("labels")
                return (loss, logits, labels)


def process_single_sample(sample: Dict, tokenizer: PreTrainedTokenizer, image_processor: Qwen2VLImageProcessor, is_first_sample: bool = False) -> Dict:
    """Process a single sample with images and text."""
    messages = sample["messages"]
    images = sample["images"]

    # Process messages and inject image tokens
    processed_messages = []
    pixel_values_list = []
    grid_thws_list = []

    for msg in messages:
        text_content = ""
        for item in msg["content"]:
            if item["type"] == "text" and item.get("text"):
                text_content += item["text"]
            elif item["type"] == "image" and item.get("index") is not None:
                # Process image
                img_idx = item["index"]
                if img_idx < len(images):
                    image = images[img_idx].convert("RGB")
                    # Follow modeling_pointsv15_chat.py logic exactly
                    img_data = image_processor(images=image)
                    pixel_values = img_data['pixel_values']
                    image_grid_thw = img_data['image_grid_thw']

                    # Extend pixel_values to match modeling file logic
                    pixel_values_list.extend(pixel_values)
                    grid_thws_list.append(image_grid_thw)

                    # Calculate sequence length and add image tokens
                    seq_len = int(image_grid_thw[0][1] * image_grid_thw[0][2] / 4)
                    text_content += f"<|vision_start|>{'<|image_pad|>' * seq_len}<|vision_end|>\n"

        processed_messages.append({"role": msg["role"], "content": text_content})

    # Build conversation manually without chat template
    all_tokens = []
    labels = []

    for msg in processed_messages:
        # Manually construct message format
        if msg["role"] == "user":
            msg_text = f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            msg_text = f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        else:  # system
            msg_text = f"<|im_start|>system\n{msg['content']}<|im_end|>\n"

        msg_tokens = tokenizer(msg_text, add_special_tokens=False)['input_ids']

        # Add tokens to sequence
        all_tokens.extend(msg_tokens)

        # Add labels: -100 for user/system, actual tokens for assistant
        if msg["role"] == "assistant":
            labels.extend(msg_tokens)  # Keep assistant tokens
        else:
            labels.extend([-100] * len(msg_tokens))  # Mask user/system tokens

    # Mask image pad tokens
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    labels = [-100 if token_id == image_pad_id else label for token_id, label in zip(all_tokens, labels)]

    # Process pixel_values and image_grid_thws like in modeling_pointsv15_chat.py
    if pixel_values_list:
        # Convert to numpy array like in modeling file
        import numpy as np
        pixel_values_array = np.array(pixel_values_list)
        concatenated_grids = np.concatenate(grid_thws_list, axis=0)
    else:
        pixel_values_array = np.empty((0, 3, 0, 0))
        concatenated_grids = np.empty((0, 3))

    result = {
        "input_ids": all_tokens,
        "labels": labels,
        "pixel_values": pixel_values_array,
        "image_grid_thws": concatenated_grids
    }

    # Debug first sample processing only
    if is_first_sample and len(pixel_values_list) > 0:
        print(f"\n=== DEBUG PROCESS_SINGLE_SAMPLE ===")
        print(f"pixel_values_list[0] shape: {pixel_values_list[0].shape}")
        print(f"grid_thws_list[0] original: {grid_thws_list[0]}")
        print(f"grid_thws_list[0] shape: {grid_thws_list[0].shape}")
        print(f"grid_thws_list[0].tolist(): {grid_thws_list[0].tolist()}")
        print("===================================\n")

    return result

def preprocess_batch(examples: Dict, tokenizer: PreTrainedTokenizer, image_processor: Qwen2VLImageProcessor, first_batch: bool = False) -> Dict:
    """High-performance batch processing."""
    batch_size = len(examples["messages"])

    # Process each sample
    processed = [
        process_single_sample(
            {"messages": examples["messages"][i], "images": examples["images"][i]},
            tokenizer,
            image_processor,
            is_first_sample=(first_batch and i == 0)
        )
        for i in range(batch_size)
    ]

    # Collate results
    return {
        "input_ids": [p["input_ids"] for p in processed],
        "labels": [p["labels"] for p in processed],
        "pixel_values": [p["pixel_values"] for p in processed],
        "image_grid_thws": [p["image_grid_thws"] for p in processed]
    }

def main():
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Get the only_one_turn parameter
    only_one_turn = script_args.only_one_turn

    # Set up gradient checkpointing and max_length
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None

    ################
    # Quantization and Model Configuration
    ################
    # Use bf16 for FlashAttention compatibility
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    if model_args.dtype and model_args.dtype not in ["auto", None]:
        dtype = getattr(torch, model_args.dtype)

    # Use eager attention by default since POINTSV15ChatModel doesn't support FlashAttention 2.0
    # attn_implementation = model_args.attn_implementation or "eager"

    model_kwargs = dict(
        revision=model_args.model_revision,
        # attn_implementation=attn_implementation,
        torch_dtype=dtype,  # Use torch_dtype instead of dtype for consistency
        trust_remote_code=True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # 1. Load Model, Tokenizer, and Image Processor
    model = POINTSV15ChatModel.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_args.model_name_or_path)

    # 2. Freeze vision components by default
    print("Freezing vision encoder and projector...")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # 3. Load and prepare dataset
    # This formatting function is specific to the LLaVA dataset structure
    def format_sample(sample: Dict) -> Dict:
        """Simple and efficient sample formatting."""
        messages = sample["messages"]
        images = sample.get("images", [])

        # Keep only first turn if enabled
        if only_one_turn and len(messages) > 2:
            messages = messages[:2]

        # Filter out incomplete samples
        if len(messages) < 2:
            return {"messages": []}

        return {"messages": messages, "images": images}

    try:
        # Load both train and validation splits
        raw_train_dataset = load_dataset(script_args.dataset_name, split="train")
        raw_val_dataset = load_dataset(script_args.dataset_name, split="test")
        print(f"Train dataset loaded successfully: {len(raw_train_dataset)} examples")
        print(f"Validation dataset loaded successfully: {len(raw_val_dataset)} examples")

    except Exception as e:
        print(f"Error loading dataset {script_args.dataset_name}: {e}")
        sys.exit(1)

    print(f"Data format: {list(raw_train_dataset[0].keys())}")

    # Process train dataset
    train_dataset = raw_train_dataset.map(
        format_sample,
        num_proc=4,
        desc="Formatting train samples"
    )
    train_dataset = train_dataset.filter(lambda x: len(x['messages']) > 0, num_proc=4)
    print(f"Final train dataset size: {len(train_dataset)}")

    # Process validation dataset
    val_dataset = raw_val_dataset.map(
        format_sample,
        num_proc=4,
        desc="Formatting validation samples"
    )
    val_dataset = val_dataset.filter(lambda x: len(x['messages']) > 0, num_proc=4)
    print(f"Final validation dataset size: {len(val_dataset)}")

    # High-performance preprocessing for train dataset
    batch_counter = 0
    def preprocess_with_counter(x):
        nonlocal batch_counter
        is_first_batch = batch_counter == 0
        batch_counter += 1
        return preprocess_batch(x, tokenizer, image_processor, first_batch=is_first_batch)

    processed_train_dataset = train_dataset.map(
        preprocess_with_counter,
        batched=True,
        batch_size=16,  # Larger batch size for better performance
        num_proc=1,     # Use single process to maintain counter
        remove_columns=train_dataset.column_names,
        desc="Processing VLM train data"
    )

    # Process validation dataset
    processed_val_dataset = val_dataset.map(
        lambda x: preprocess_batch(x, tokenizer, image_processor, first_batch=False),
        batched=True,
        batch_size=16,
        num_proc=1,
        remove_columns=val_dataset.column_names,
        desc="Processing VLM validation data"
    )

    # Print first sample to verify data processing
    print("\n" + "="*80)
    print("FIRST TRAINING SAMPLE VERIFICATION")
    print(f"Only one turn mode: {only_one_turn}")
    print("="*80)

    first_sample = processed_train_dataset[0]

    # Handle both tensor and list formats
    input_ids = first_sample['input_ids']
    labels = first_sample['labels']
    pixel_values = first_sample['pixel_values']
    image_grid_thws = first_sample['image_grid_thws']

    print(f"Input IDs length: {len(input_ids)}")
    print(f"Labels length: {len(labels)}")
    print(f"Pixel values shape: {pixel_values.shape if hasattr(pixel_values, 'shape') else f'tensor/list with length {len(pixel_values) if hasattr(pixel_values, '__len__') else type(pixel_values)}'}")
    print(f"Image grid thws shape: {image_grid_thws.shape if hasattr(image_grid_thws, 'shape') else f'tensor/list with length {len(image_grid_thws) if hasattr(image_grid_thws, '__len__') else type(image_grid_thws)}'}")

    # Convert to list if tensor for decoding
    if hasattr(input_ids, 'tolist'):
        input_ids_list = input_ids.tolist()
    else:
        input_ids_list = input_ids

    # Convert labels to list if tensor
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    else:
        labels_list = labels

    # Create text with color highlighting - green for training tokens
    print(f"\nText with highlighting (\\033[32mgreen\\033[0m = training tokens):")
    print("-" * 60)

    # Decode token by token and highlight training ones
    for i, (token_id, label) in enumerate(zip(input_ids_list, labels_list)):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if label != -100:  # Training token
            print(f"\033[32m{token_text}\033[0m", end="")  # Green color
        else:  # Masked token
            print(token_text, end="")  # Normal color
    print()  # New line at the end

    # Show which tokens are masked
    masked_count = sum(1 for label in labels_list if label == -100)
    total_count = len(labels_list)
    print(f"\nMasking info:")
    print(f"  Total tokens: {total_count}")
    print(f"  Masked tokens: {masked_count}")
    print(f"  Training tokens: {total_count - masked_count}")
    print(f"  Masking ratio: {masked_count/total_count:.2%}")

    # Check for image pad tokens
    image_pad_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_pad_count = sum(1 for token_id in input_ids_list if token_id == image_pad_token_id)
    print(f"  Image pad tokens: {image_pad_count}")

    print("="*80 + "\n")

    # 4. Initialize Trainer
    training_args.remove_unused_columns = False # MUST be False to keep pixel_values etc.
    training_args.dataset_text_field = "input_ids"  # Set in config instead of trainer

    # Configure evaluation settings
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 10
    training_args.do_eval = True

    # Get and fix PEFT config
    peft_config = get_peft_config(model_args)
    if peft_config and hasattr(peft_config, 'target_modules') and isinstance(peft_config.target_modules, str):
        # Convert comma-separated string to list
        peft_config.target_modules = [module.strip() for module in peft_config.target_modules.split(',')]

    # Simple data collator that handles image fields
    def custom_data_collator(features):
        """Simple data collator for VLM training."""
        # Extract text fields for padding
        text_features = []
        pixel_values_list = []
        image_grid_thws_list = []

        for feature in features:
            text_features.append({
                "input_ids": feature["input_ids"],
                "labels": feature["labels"],
                "attention_mask": [1] * len(feature["input_ids"])
            })
            pixel_values_list.append(feature["pixel_values"])
            image_grid_thws_list.append(feature["image_grid_thws"])

        # Pad text sequences
        text_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
        batch = text_collator(text_features)

        # Handle image data - concatenate instead of stack for variable sizes
        import numpy as np

        if pixel_values_list:
            # Concatenate all pixel values along batch dimension
            batch["pixel_values"] = np.concatenate([np.array(pv) for pv in pixel_values_list], axis=0)
        else:
            batch["pixel_values"] = np.empty((0, 3, 0, 0))

        if image_grid_thws_list:
            # Concatenate all grid info along batch dimension
            batch["image_grid_thws"] = np.concatenate([np.array(grid) for grid in image_grid_thws_list], axis=0)
        else:
            batch["image_grid_thws"] = np.empty((0, 3))

        return batch

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        peft_config=peft_config,
        data_collator=custom_data_collator,
    )

    # Manually set the tokenizer - this is required for our custom compute_loss
    trainer.tokenizer = tokenizer

    # 5. Train
    trainer.train()

    # 6. Save
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Training complete. Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()