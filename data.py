"""
Data loading and processing for training

Supports:
- Streaming datasets from HuggingFace
- On-the-fly tokenization
- Variable-length sequence packing
"""
import os
from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class DataConfig:
    """Configuration for data loading"""
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    tokenizer_path: str = "Qwen/Qwen3-0.6B"
    max_seq_length: int = 32768
    batch_size: int = 1
    num_workers: int = 0  # Set to 0 to avoid multi-process tokenizer loading overhead
    prefetch_factor: int = 2
    streaming: bool = True
    seed: int = 42


class TokenizedDataset(IterableDataset):
    """
    Iterable dataset that tokenizes on-the-fly and packs sequences
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str,
        tokenizer,
        max_seq_length: int,
        streaming: bool = True,
        seed: int = 42,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.streaming = streaming
        self.seed = seed
        self.split = split
        
        self._dataset = None

    def _load_dataset(self):
        """Lazy load dataset"""
        if self._dataset is not None:
            return
        
        from datasets import load_dataset
        
        if self.streaming:
            self._dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.split,
                streaming=True,
            )
            self._dataset = self._dataset.shuffle(seed=self.seed, buffer_size=10000)
        else:
            self._dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.split,
            )
            self._dataset = self._dataset.shuffle(seed=self.seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._load_dataset()
        
        buffer = []
        buffer_len = 0
        
        for example in self._dataset:
            # Tokenize
            text = example.get("text", example.get("content", ""))
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Add BOS token at the start
            if self.tokenizer.bos_token_id is not None:
                tokens = [self.tokenizer.bos_token_id] + tokens
            
            # Add to buffer
            buffer.extend(tokens)
            buffer_len += len(tokens)
            
            # Yield packed sequences
            while buffer_len >= self.max_seq_length:
                seq = buffer[:self.max_seq_length]
                buffer = buffer[self.max_seq_length:]
                buffer_len -= self.max_seq_length
                
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = input_ids.clone()

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                }


class PackedDataset(IterableDataset):
    """
    Pre-packed dataset for efficient training
    Packs multiple documents into single sequences with proper attention masking
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str,
        tokenizer,
        max_seq_length: int,
        streaming: bool = True,
        seed: int = 42,
        split: str = "train",
        pack_sequences: bool = True,
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.streaming = streaming
        self.seed = seed
        self.split = split
        self.pack_sequences = pack_sequences

        self._dataset = None

        # Get special token ids
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id or self.eos_token_id

        # Validate that we have valid token IDs
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer does not have pad_token_id, eos_token_id, or bos_token_id set. "
                "Please ensure the tokenizer has at least one special token defined."
            )

    def _load_dataset(self):
        if self._dataset is not None:
            return

        from datasets import load_dataset

        if self.streaming:
            self._dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.split,
                streaming=True,
            )
            self._dataset = self._dataset.shuffle(seed=self.seed, buffer_size=10000)
        else:
            self._dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.split,
            )
            self._dataset = self._dataset.shuffle(seed=self.seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._load_dataset()
        
        if self.pack_sequences:
            yield from self._packed_iterator()
        else:
            yield from self._simple_iterator()

    def _simple_iterator(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Simple iteration without packing"""
        for example in self._dataset:
            text = example.get("text", example.get("content", ""))
            
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                add_special_tokens=True,
            )
            
            # Pad if necessary
            if len(tokens) < self.max_seq_length:
                padding_length = self.max_seq_length - len(tokens)
                tokens = tokens + [self.pad_token_id] * padding_length
                attention_mask = [1] * (self.max_seq_length - padding_length) + [0] * padding_length
            else:
                attention_mask = [1] * self.max_seq_length
            
            input_ids = torch.tensor(tokens[:self.max_seq_length], dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
            labels = input_ids.clone()
            labels[~attention_mask] = -100  # Ignore padding in loss
            
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    def _packed_iterator(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Pack multiple sequences with document boundaries"""
        buffer = []
        doc_boundaries = []  # Track where documents start
        
        for example in self._dataset:
            text = example.get("text", example.get("content", ""))
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Add document markers (only if token IDs are not None)
            doc_tokens = []
            if self.bos_token_id is not None:
                doc_tokens.append(self.bos_token_id)
            doc_tokens.extend(tokens)
            if self.eos_token_id is not None:
                doc_tokens.append(self.eos_token_id)
            
            # Skip very long documents
            if len(doc_tokens) > self.max_seq_length:
                doc_tokens = doc_tokens[:self.max_seq_length]
            
            # Check if we can add to buffer
            if len(buffer) + len(doc_tokens) <= self.max_seq_length:
                doc_boundaries.append(len(buffer))
                buffer.extend(doc_tokens)
            else:
                # Yield current buffer (pad if not empty)
                if buffer:
                    yield self._create_packed_batch(buffer, doc_boundaries, pad=True)

                # Start new buffer
                buffer = doc_tokens
                doc_boundaries = [0]
            
            # Yield complete sequences
            while len(buffer) >= self.max_seq_length:
                yield self._create_packed_batch(buffer[:self.max_seq_length], 
                                                [b for b in doc_boundaries if b < self.max_seq_length])
                
                # Update buffer and boundaries
                remaining_start = self.max_seq_length
                buffer = buffer[remaining_start:]
                doc_boundaries = [b - remaining_start for b in doc_boundaries if b >= remaining_start]
        
        # Yield final partial buffer
        if buffer:
            yield self._create_packed_batch(buffer, doc_boundaries, pad=True)

    def _create_packed_batch(
        self,
        tokens: List[int],
        doc_boundaries: List[int],
        pad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Create batch from packed tokens"""
        seq_len = len(tokens)

        if pad and seq_len < self.max_seq_length:
            padding_length = self.max_seq_length - seq_len
            tokens = tokens + [self.pad_token_id] * padding_length
            attention_mask = [1] * seq_len + [0] * padding_length
        else:
            attention_mask = [1] * len(tokens)

        input_ids = torch.tensor(tokens[:self.max_seq_length], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:self.max_seq_length], dtype=torch.bool)
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "doc_boundaries": torch.tensor(doc_boundaries, dtype=torch.long),
        }


def create_dataloader(
    config: DataConfig,
    tokenizer,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create dataloader for training"""
    
    dataset = PackedDataset(
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        streaming=config.streaming,
        seed=config.seed + rank,  # Different seed per rank
        pack_sequences=True,
    )
    
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


def get_tokenizer(tokenizer_path: str):
    """Load tokenizer"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    # Ensure special tokens are set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.bos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.bos_token_id
        else:
            # Fallback to vocab size - 1 if no special tokens exist
            tokenizer.pad_token_id = tokenizer.vocab_size - 1

    return tokenizer
