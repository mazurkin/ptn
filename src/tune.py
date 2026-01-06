"""Fine-tune a decoder model on Q&A pairs from chat JSON files."""
import enum
import json
import logging
import os
import dataclasses
import torch
import semchunk
import abc

from enum import StrEnum
from pathlib import Path
from dataclasses import field, dataclass
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BatchEncoding,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@enum.unique
class SpecialToken(StrEnum):
    """Special tokens for Q&A format - these are added to tokenizer vocabulary."""

    QUESTION = "<|question|>"

    ANSWER = "<|answer|>"


@dataclass(frozen=True)
class ModelDescriptorItem:
    """
    single model descriptor
    """

    # HF model name (repository)
    repository: str

    # inference pattern for QA
    chat_infer_pattern: str

    # training pattern for QA
    chat_train_pattern: str


@enum.unique
class ModelDescriptor(enum.Enum):
    """
    enumeration of the base model to be used for fine-tuning

    choices are (you will need a model with the good support of russian language):

    # the best modern model so far (LLama architecture, first-class support of russian language)
    yandex/YandexGPT-5-Lite-8B-pretrain

    # I don't like some grammar of the resulting texts in russian
    # russian language is second-class in the tokenizer (generates many tokens even for simple words)
    Qwen/Qwen3-32B
    Qwen/Qwen3-14B
    Qwen/Qwen3-8B
    Qwen/Qwen3-4B
    Qwen/Qwen3-1.7B

    # I don't like some grammar of the resulting texts in russian
    # russian language is second-class in the tokenizer (generates many tokens even for simple words)
    Qwen/Qwen2.5-32B
    Qwen/Qwen2.5-14B
    Qwen/Qwen2.5-7B
    Qwen/Qwen2.5-3B
    Qwen/Qwen2.5-0.5B

    # Sberbank models, bases on the previous GPT architecture
    ai-forever/mGPT-13B
    ai-forever/mGPT
    ai-forever/rugpt3large_based_on_gpt2
    """

    # Yandex GPT 5

    YANDEX_GPT_8B = ModelDescriptorItem(
        repository='yandex/YandexGPT-5-Lite-8B-pretrain',
        chat_infer_pattern="<|question|>{prompt}<|answer|>",
        chat_train_pattern="<|question|>{prompt}<|answer|>{answer}{eos}",
    )

    # QWEN 3
    # /no_think disables Qwen3 thinking mode for direct responses

    QWEN_3_2B = ModelDescriptorItem(
        repository='Qwen/Qwen3-1.7B',
        chat_infer_pattern="/no_think <|question|>{prompt}<|answer|>",
        chat_train_pattern="<|question|>{prompt}<|answer|>{answer}{eos}",
    )

    QWEN_3_14B = ModelDescriptorItem(
        repository='Qwen/Qwen3-14B',
        chat_infer_pattern="/no_think <|question|>{prompt}<|answer|>",
        chat_train_pattern="<|question|>{prompt}<|answer|>{answer}{eos}",
    )

    # QWEN 2.5

    QWEN_25_1B = ModelDescriptorItem(
        repository='Qwen/Qwen2.5-0.5B',
        chat_infer_pattern="<|question|>{prompt}<|answer|>",
        chat_train_pattern="<|question|>{prompt}<|answer|>{answer}{eos}",
    )

    QWEN_25_14B = ModelDescriptorItem(
        repository='Qwen/Qwen2.5-14B',
        chat_infer_pattern="<|question|>{prompt}<|answer|>",
        chat_train_pattern="<|question|>{prompt}<|answer|>{answer}{eos}",
    )

    @property
    def repository(self) -> str:
        return self.value.repository

    @property
    def chat_infer_pattern(self) -> str:
        return self.value.chat_infer_pattern

    @property
    def chat_train_pattern(self) -> str:
        return self.value.chat_train_pattern


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    descriptor: ModelDescriptor = field(
        default=ModelDescriptor.YANDEX_GPT_8B,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype for model (auto, float16, bfloat16, float32)"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA (Low-Rank Adaptation) configuration."""

    # Rank of the low-rank matrices. Higher = more capacity but more parameters.
    # Typical values: 8, 16, 32, 64. Use 32+ for better quality, especially to prevent repetition.
    lora_r: int = field(
        default=48,
        metadata={"help": "LoRA rank (dimension of low-rank matrices)"}
    )

    # Scaling factor for LoRA layers. Usually set to 2x the rank.
    # Higher alpha = stronger LoRA effect relative to base model.
    lora_alpha: int = field(
        default=96,
        metadata={"help": "LoRA alpha (scaling factor)"}
    )

    # Dropout probability for LoRA layers. Helps prevent overfitting.
    # Typical values: 0.05-0.1 for small datasets.
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers"}
    )

    # Which modules to apply LoRA to. Target attention projections for best results.
    # Common choices: q_proj, k_proj, v_proj, o_proj (attention), gate_proj, up_proj, down_proj (MLP)
    # Including MLP layers (gate/up/down) gives more capacity and helps prevent repetition.
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to"}
    )

    # Directory where merged model will be saved after training.
    # The merged model combines base model + LoRA weights for easy deployment.
    merged_output_dir: str = field(
        default="../target/model-merged",
        metadata={"help": "Directory to save the merged model (base + LoRA)"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    data_dir: str = field(
        default="../data",
        metadata={"help": "Directory containing JSON and TXT files"}
    )

    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization (model context length)"}
    )

    speech_chunk_tokens: int = field(
        default=1536,
        metadata={"help": "Target size of speech chunks in tokens (uses semantic chunking)"}
    )

    speech_chunk_overlap: float = field(
        default=0.1,
        metadata={"help": "Overlap ratio (0-1) or absolute token count (>=1) for speech chunks"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments with custom defaults for fine-tuning."""

    # Directory where model checkpoints and final model will be saved.
    # Checkpoints include model weights, optimizer state, and training metadata.
    output_dir: str = field(default="../target/model")

    # Directory for keeping the tensorflow logs
    logging_dir: str = field(default="../target/tensorboard")

    # Number of complete passes through the entire training dataset.
    # This is set dynamically before each training phase from num_pretrain_epochs or num_qa_epochs.
    num_train_epochs: int = field(default=0)

    # Number of epochs for pretraining on extras (domain adaptation).
    num_extras_epochs: int = field(default=2)

    # Number of epochs for pretraining on speeches (domain adaptation).
    num_pretrain_epochs: int = field(default=5)

    # More epochs = more training but risk of overfitting on small datasets.
    num_qa_epochs: int = field(default=5)

    # Number of training samples processed per GPU in each forward pass.
    # Limited by GPU memory; smaller values use less memory but train slower.
    per_device_train_batch_size: int = field(default=4)

    # Number of evaluation samples processed per GPU in each forward pass.
    # Can be larger than train batch size since no gradients are computed.
    per_device_eval_batch_size: int = field(default=4)

    # Number of forward passes to accumulate before performing a backward pass.
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps.
    # Allows larger effective batches without increasing memory usage.
    gradient_accumulation_steps: int = field(default=4)

    # Initial learning rate for AdamW optimizer.
    # Controls step size during gradient descent; 2e-5 is typical for fine-tuning.
    learning_rate: float = field(default=2e-5)

    # L2 regularization coefficient applied to all weights except biases and LayerNorm.
    # Helps prevent overfitting by penalizing large weight values.
    weight_decay: float = field(default=0.1)

    # Fraction of total training steps used for linear learning rate warmup.
    # Gradually increases LR from 0 to prevent early training instability.
    warmup_ratio: float = field(default=0.1)

    # Log training metrics (loss, learning rate, etc.) every N steps.
    # Lower values give more granular monitoring but slightly more overhead.
    logging_steps: int = field(default=10)

    # Save a model checkpoint every N steps.
    # Allows resuming training and provides intermediate model snapshots.
    save_steps: int = field(default=1000)

    # Maximum number of checkpoints to keep on disk.
    # Older checkpoints are deleted to save disk space; best model is always kept.
    save_total_limit: int = field(default=1)

    # When to run evaluation: "no", "steps", "epoch".
    # "steps" runs evaluation every eval_steps; "epoch" runs after each epoch.
    eval_strategy: str = field(default="steps")

    # Run evaluation every N training steps (when eval_strategy="steps").
    # More frequent evaluation helps monitor overfitting but adds overhead.
    eval_steps: int = field(default=500)

    # After training, load the checkpoint with the best metric value.
    # Ensures the final saved model is the best performing one, not the last.
    load_best_model_at_end: bool = field(default=True)

    # Which metric to use for determining the "best" model checkpoint.
    # Common choices: "loss", "eval_loss", "accuracy", "perplexity".
    metric_for_best_model: str = field(default="loss")

    # Whether higher metric values are better (True) or lower (False).
    # For loss/perplexity: False. For accuracy/F1: True.
    greater_is_better: bool = field(default=False)

    # Use 16-bit floating point precision (FP16) for training.
    # Reduces memory usage and speeds up training on compatible GPUs (NVIDIA).
    fp16: bool = field(default=False)

    # Use bfloat16 precision for training.
    # Better numerical stability than FP16; supported on Ampere+ GPUs and TPUs.
    bf16: bool = field(default=True)

    # Enable gradient checkpointing to trade compute for memory.
    # Recomputes activations during backward pass instead of storing them.
    gradient_checkpointing: bool = field(default=True)

    # gradient clipping (already built into Trainer)
    max_grad_norm: float = field(default=1.0)

    # Pin data in CPU memory for faster GPU transfer.
    # Speeds up CPU-to-GPU data transfer by using page-locked memory.
    dataloader_pin_memory: bool = field(default=True)

    # Number of subprocesses for data loading.
    # More workers can speed up data preprocessing but use more CPU/memory.
    dataloader_num_workers: int = field(default=0)

    # Integration for logging metrics: "none", "wandb", "tensorboard", etc.
    # "none" disables external logging; use "wandb" or "tensorboard" for experiment tracking.
    report_to: str = field(default="tensorboard")


class PtnTrainerDatasets(abc.ABC):

    def __init__(self, name: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.name = name
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def create_texts(self) -> list[str]:
        """Load texts from files and split into chunks."""
        pass


    def create_dataset(self, max_length: int, test_size: float = 0.1) -> tuple[Dataset, Dataset]:
        """Create train and eval datasets from formatted texts."""
        texts: list[str] = self.create_texts()

        # Create dataset
        dataset = Dataset.from_dict({"text": texts}).shuffle(seed=42)

        # Split into train/eval
        split = dataset.train_test_split(test_size=test_size, seed=42)

        # Tokenizer processor
        def tokenize_function(examples) -> BatchEncoding:
            # Tokenize and add EOS token
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
            )

            # For causal LM, labels are the same as input_ids
            outputs["labels"] = outputs["input_ids"].copy()

            return outputs

        train_dataset = split["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing train dataset"
        )

        eval_dataset = split["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval dataset"
        )

        logger.info("Train dataset '%s' size: %s", self.name, len(train_dataset))
        logger.info("Eval dataset '%s' size: %s", self.name,  len(eval_dataset))

        return train_dataset, eval_dataset


class PtnTrainerJsonDatasets(PtnTrainerDatasets):

    def __init__(self, name: str, tokenizer: PreTrainedTokenizerBase, folder: Path, pattern: str):
        super().__init__(name=name, tokenizer=tokenizer)

        self.folder = folder
        self.pattern = pattern

    def create_texts(self) -> list[str]:
        pairs: list[dict] = self.load_qa_pairs()
        if not pairs:
            raise ValueError("No JSON Q/A found in %s", self.folder)

        return self.format_qa_pairs(pairs)

    def format_qa_pairs(self, qa_pairs: list[dict]) -> list[str]:
        """Format Q&A pairs using the template with special tokens.

        Each example ends with EOS token so the model learns when to stop generating.
        """
        formatted = []

        for pair in qa_pairs:
            text = self.pattern.format(
                prompt=str(pair['questions']),
                answer=str(pair['answer']),
                eos=self.tokenizer.eos_token,
            )

            formatted.append(text)

        return formatted

    def load_qa_pairs(self) -> list[dict]:
        """Load all Q&A pairs from JSON files in the data directory."""
        all_pairs: list[dict] = list()

        json_files = sorted(self.folder.rglob("*.json"))
        logger.info("Found %s JSON files in %s", len(json_files), self.folder)

        for json_file in json_files:
            if not json_file.is_file():
                continue

            with open(json_file, 'rt', encoding='utf-8') as f:
                pairs = json.load(f)
                all_pairs.extend(pairs)
                logger.info(f"  Loaded %s pairs from %s", len(pairs), json_file.name)

        logger.info("Total Q&A pairs loaded: %s", len(all_pairs))

        return all_pairs


class PtnTrainerTextDatasets(PtnTrainerDatasets):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        folder: Path,
        chunk_tokens: int,
        chunk_overlap: float | int | None = None,
    ):
        super().__init__(name=name, tokenizer=tokenizer)

        self.folder = folder
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    def create_texts(self) -> list[str]:
        # noinspection PyTypeChecker
        chunker = semchunk.chunkerify(self.tokenizer, chunk_size=self.chunk_tokens)

        chunks: list[str] = list()

        txt_files = sorted(self.folder.rglob("*.txt"))
        logger.info("Found %s TXT files in %s", len(txt_files), self.folder)

        for txt_file in txt_files:
            if not txt_file.is_file():
                continue

            with open(txt_file, 'rt', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                continue

            file_chunks = chunker(text, overlap=self.chunk_overlap)
            chunks.extend(file_chunks)

            logger.info("  Loaded %s chunks from %s", len(file_chunks), txt_file.name)

        if not chunks:
            raise ValueError("No text chunks found int %s", self.folder)

        logger.info("Total text chunks loaded: %s", len(chunks))

        # Format speech chunks with EOS token and create dataset
        texts = [chunk + self.tokenizer.eos_token for chunk in chunks]

        return texts


class PtnTraining:

    def __init__(
        self,
        name: str,
        datasets: PtnTrainerDatasets,
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel | PeftModel,
        data_collator: DataCollatorForSeq2Seq,
        num_epochs: int,
        max_length: int,
        test_size: float = 0.1,
    ):
        self.name = name

        training_args_copy = dataclasses.replace(
            training_args,
            num_train_epochs=num_epochs,
        )

        train_dataset, eval_dataset = datasets.create_dataset(
            max_length=max_length,
            test_size=test_size,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args_copy,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

    def run(self):
        logger.info("Training %s in %s epochs", self.name, self.trainer.args.num_train_epochs)

        train_result = self.trainer.train()

        train_metrics = train_result.metrics
        self.trainer.log_metrics(self.name + "_train", train_metrics)
        self.trainer.save_metrics(self.name + "_train", train_metrics)

        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics(self.name + "_eval", eval_metrics)
        self.trainer.save_metrics(self.name + "_eval", eval_metrics)

        train_loss = train_metrics.get('train_loss', 'N/A')
        logger.info("Training of %s is complete. Loss: %s", self.name, train_loss)

    def save_model(self):
        logger.info("Saving the model %s to %s", self.name, self.trainer.args.output_dir)
        self.trainer.save_model()


# noinspection DuplicatedCode
def main():
    # ---------------------------------------------------------------
    # configuration
    # ---------------------------------------------------------------

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, CustomTrainingArguments))

    model_args: ModelArguments
    lora_args: LoraArguments
    data_args: DataArguments
    training_args: CustomTrainingArguments

    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent

    data_args.data_dir = str((script_dir / data_args.data_dir).resolve())
    logger.info(f"Data directory: {data_args.data_dir}")

    training_args.output_dir = str((script_dir / training_args.output_dir).resolve())
    logger.info(f"Output directory: {training_args.output_dir}")
    training_args.logging_dir = str((script_dir / training_args.logging_dir).resolve())
    logger.info(f"Logging directory: {training_args.logging_dir}")

    lora_args.merged_output_dir = str((script_dir / lora_args.merged_output_dir).resolve())
    logger.info(f"Merge directory: {lora_args.merged_output_dir}")

    # ---------------------------------------------------------------
    # base model tokenizer
    # ---------------------------------------------------------------

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_args.descriptor.repository}")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_args.descriptor.repository,
        trust_remote_code=True,
    )

    # Add special Q&A tokens to vocabulary
    special_tokens_list = list(str(token.value) for token in SpecialToken)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})
    logger.info(f"Added {num_added} special tokens to tokenizer: {special_tokens_list}")

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"

    # Log original chat template (if any) before we override it
    if tokenizer.chat_template:
        logger.info(f"Original chat template:\n{tokenizer.chat_template}")
    else:
        logger.info("No chat template defined in base tokenizer")

    # Add chat template for llama.cpp/Ollama compatibility (Jinja2 format)
    # Only uses the last user message since model was trained on single Q&A pairs, not multi-turn
    tokenizer.chat_template = model_args.descriptor.chat_infer_pattern.format(
        prompt="{{ messages[-1]['content'] }}",
    )

    # ---------------------------------------------------------------
    # base model configuration
    # ---------------------------------------------------------------

    # Determine torch dtype
    if model_args.torch_dtype == "auto":
        torch_dtype = "auto"
    elif model_args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif model_args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    config = AutoConfig.from_pretrained(
        model_args.descriptor.repository,
        trust_remote_code=True,
    )

    if training_args.gradient_checkpointing:
        # fixes https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/trainer.py#L1991
        config.use_cache = False

    # ---------------------------------------------------------------
    # load base model
    # ---------------------------------------------------------------

    # Load base model
    logger.info(f"Loading base model: {model_args.descriptor.repository}")

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_args.descriptor.repository,
        config=config,
        dtype=torch_dtype,
        trust_remote_code=True,
    )

    if not hasattr(model, 'loss_type') or model.loss_type is None:
        # fixes https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/modeling_utils.py#L4256
        model.loss_type = "ForCausalLM"

    # Resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # Log base model parameters
    logger.info(f"Base model parameters: {model.num_parameters():,}")

    # ---------------------------------------------------------------
    # LoRA adapter
    # ---------------------------------------------------------------

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model: PeftModel = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for LoRA - must use peft's method
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ---------------------------------------------------------------
    # log the model's parameters
    # ---------------------------------------------------------------

    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(f"LoRA trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    model.print_trainable_parameters()

    # Log actual dtype used
    actual_dtype = next(model.parameters()).dtype
    logger.info("Model dtype: %s", actual_dtype)

    # ---------------------------------------------------------------
    # data collator
    # ---------------------------------------------------------------

    # Data collator that handles padding properly for causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # For efficiency
    )

    # ---------------------------------------------------------------
    # Phase 1: Warming up on extras (domain adaptation)
    # ---------------------------------------------------------------

    logger.info("Phase 1: Warming up on extras (domain adaptation)")

    extras_datasets = PtnTrainerTextDatasets(
        name='extras',
        tokenizer=tokenizer,
        folder=Path(data_args.data_dir) / 'extras',
        chunk_tokens=data_args.speech_chunk_tokens,
        chunk_overlap=data_args.speech_chunk_overlap,
    )

    extras_training = PtnTraining(
        name='extras',
        datasets=extras_datasets,
        training_args=training_args,
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        num_epochs=training_args.num_extras_epochs,
        max_length=data_args.max_length,
        test_size=0.1,
    )

    extras_training.run()

    # ---------------------------------------------------------------
    # Phase 2: Pretraining on speeches (domain adaptation)
    # ---------------------------------------------------------------

    logger.info("Phase 2: Pretraining on speeches (domain adaptation)")

    speeches_datasets = PtnTrainerTextDatasets(
        name='speeches',
        tokenizer=tokenizer,
        folder=Path(data_args.data_dir) / 'speeches',
        chunk_tokens=data_args.speech_chunk_tokens,
        chunk_overlap=data_args.speech_chunk_overlap,
    )

    speeches_training = PtnTraining(
        name='speeches',
        datasets=speeches_datasets,
        training_args=training_args,
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        num_epochs=training_args.num_pretrain_epochs,
        max_length=data_args.max_length,
        test_size=0.1,
    )

    speeches_training.run()

    # ---------------------------------------------------------------
    # Phase 3: Fine-tuning on Q&A pairs (instruction tuning)
    # ---------------------------------------------------------------

    logger.info("Phase 3: Fine-tuning on Q&A pairs (instruction tuning)")

    chats_datasets = PtnTrainerJsonDatasets(
        name='chats',
        tokenizer=tokenizer,
        folder=Path(data_args.data_dir) / 'chats',
        pattern=model_args.descriptor.chat_train_pattern,
    )

    chats_training = PtnTraining(
        name='chats',
        datasets=chats_datasets,
        training_args=training_args,
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        num_epochs=training_args.num_qa_epochs,
        max_length=data_args.max_length,
        test_size=0.1,
    )

    chats_training.run()
    chats_training.save_model()

    # ---------------------------------------------------------------
    # save the tokenizer
    # ---------------------------------------------------------------

    logger.info(f"Saving tokenizer to {training_args.output_dir}")
    tokenizer.save_pretrained(training_args.output_dir)

    # ---------------------------------------------------------------
    # merge the base model and LoRA and save the merged model
    # ---------------------------------------------------------------

    # Merge LoRA weights into base model for easy deployment
    logger.info("Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving the merged model to {lora_args.merged_output_dir}")
    os.makedirs(lora_args.merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(lora_args.merged_output_dir)
    tokenizer.save_pretrained(lora_args.merged_output_dir)

    # ---------------------------------------------------------------
    # adjust the parameters in the resulting config for inference
    # ---------------------------------------------------------------

    # Update generation config with recommended defaults (preserves model's existing settings)
    gen_config = GenerationConfig.from_pretrained(lora_args.merged_output_dir)

    # noinspection PyTypeChecker
    gen_config.update(
        max_length=2048,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=True,
    )

    gen_config.save_pretrained(lora_args.merged_output_dir)
    logger.info("Merged configuration saved successfully to %s", lora_args.merged_output_dir)


if __name__ == "__main__":
    main()
