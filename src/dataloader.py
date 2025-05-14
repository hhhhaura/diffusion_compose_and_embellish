import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


import utils

LOGGER = utils.get_logger(__name__)

# yunchenc: new group text function
def _group_texts(examples, block_size, cond_size):
  num = len(examples['input_ids'])
  _values = []
  _conds = []  # For condition IDs

  for i in range(num):
    _values.append(
        examples['input_ids'][i]
    )
    if 'cond_ids' in examples:
      _conds.append(
          examples['cond_ids'][i]
      )
  result = {}
  result['input_ids'] = _values
  result['cond_ids'] = _conds

  return result

# yunchenc: TODO is to fix test logic
def load_condition_target_dataset(
    cond_file, target_file, tokenizer, max_length_target, max_length_cond,
    num_proc=len(os.sched_getaffinity(0))):

  """
  Load conditions and targets into a dataset and preprocess them.
  """
  # Read the files
  with open(cond_file, "r") as f:
      conditions = f.readlines()
  if target_file is not None: 
    with open(target_file, "r") as f:
        targets = f.readlines()
    assert len(conditions) == len(targets), "Conditions and targets must have the same number of lines."

  # Create a raw dataset
  data = {"conditions": [line.strip() for line in conditions], 
          "targets": [line.strip() for line in targets] if target_file is not None else ["" for line in conditions]}
  dataset = datasets.Dataset.from_dict(data)

  # Tokenization
  def tokenize_function(example):
      cond_encodings = tokenizer(
          example["conditions"], 
          max_length=max_length_cond,
          padding="max_length", 
          truncation=True,
          add_special_tokens=False,
          return_attention_mask=True,
          return_token_type_ids=False)

      # yunchenc: is the BOS and EOS still there
      cond_tokens = {'cond_ids': cond_encodings['input_ids']}

      target_encodings = tokenizer(
          example["targets"], 
          max_length=max_length_target, 
          padding="max_length", 
          truncation=True,
          add_special_tokens=False,
          return_attention_mask=True,
          return_token_type_ids=False)

      target_tokens = {'input_ids': target_encodings['input_ids']}

      return {
          "cond_ids": cond_tokens['cond_ids'],
          "input_ids": target_tokens['input_ids'],
          "attention_mask": target_encodings['attention_mask'],
      }

  tokenized_dataset = dataset.map(
    tokenize_function, batched=True, num_proc=num_proc, 
    load_from_cache_file=True,
    desc='Tokenizing')

  return tokenized_dataset



def get_cond_dataset(config, tokenizer, wrap, mode, cache_dir, num_proc=len(os.sched_getaffinity(0)), streaming=False):

  # Tokenizer settings
  if wrap:
    filename = f'{config.data.name}_{mode}_bs{config.data.tgt_length}_wrapped.dat'
  else:
    filename = f'{config.data.name}_{mode}_bs{config.data.tgt_length}_unwrapped.dat'

  _path = None
  if cache_dir != None: 
    _path = os.path.join(cache_dir, filename)

  if not _path is None and utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')

  if not _path is None: 
    LOGGER.info(f'Generating new data at: {_path}')

  if mode == 'train':
    tokenized_dataset = load_condition_target_dataset(
      config.data.train_cond, config.data.train_tgt,
      tokenizer, config.data.tgt_length, config.data.cond_length,
      num_proc=num_proc)
    tokenized_dataset = tokenized_dataset.remove_columns(['conditions', 'targets'])
  elif mode == 'valid':
    tokenized_dataset = load_condition_target_dataset(
      config.data.valid_cond, config.data.valid_tgt,
      tokenizer, config.data.tgt_length, config.data.cond_length,
      num_proc=num_proc)
    tokenized_dataset = tokenized_dataset.remove_columns(['conditions', 'targets'])
  elif mode == 'test':
    tokenized_dataset = load_condition_target_dataset(
      config.data.cond, None,
      tokenizer, config.data.tgt_length, config.data.cond_length,
      num_proc=num_proc)
    tokenized_dataset = tokenized_dataset.remove_columns(['conditions', 'targets'])
  else:
    raise ValueError(f"Invalid type: {type}. Must be 'train', 'valid', or 'test'.")

  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  tokenizer.padding_side = 'right'
  tokenizer.truncation_side = 'right'

  group_texts = functools.partial(
    _group_texts, block_size=config.data.tgt_length, cond_size=config.data.cond_length)

  chunked_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=num_proc,
    load_from_cache_file=True,
    desc='Grouping')

  if not _path is None: 
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')

  return chunked_dataset


def get_tokenizer(config):
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):

  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))

  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')

  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')

  if skip_train:
    train_set = None
  else:
    train_set = get_cond_dataset(config, tokenizer, mode='train',
        wrap=config.data.wrap,
        cache_dir=config.data.cache_dir)

  if skip_valid:
    valid_set = None
  else:
      valid_set = get_cond_dataset(config, tokenizer, mode='valid',
        wrap=config.data.wrap,
        cache_dir=config.data.cache_dir,
        streaming=False)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer

  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)

    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)

    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader

# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0