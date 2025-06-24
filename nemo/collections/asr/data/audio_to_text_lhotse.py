# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json # Add this import
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class TextOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset that loads tokenized text from a JSON file.
    Each line in the JSON file is expected to be a list of token IDs.
    """

    def __init__(self, tokenized_text_filepath: str, tokenizer: TokenizerSpec):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer) # Though tokenizer might not be directly used if text is pre-tokenized
        self.token_sequences = []
        with open(tokenized_text_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming each line is a JSON list of integers (token IDs)
                self.token_sequences.append(torch.tensor(json.loads(line), dtype=torch.long))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.token_sequences[index]

    def __len__(self) -> int:
        return len(self.token_sequences)


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        # Add unpaired_tokens and unpaired_token_lens to output_types
        output_types = {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }
        if hasattr(self, 'text_only_dataset') and self.text_only_dataset is not None:
            output_types['unpaired_tokens'] = NeuralType(('B', 'T'), LabelsType())
            output_types['unpaired_token_lens'] = NeuralType(tuple('B'), LengthsType())
        return output_types

    def __init__(self, tokenizer: TokenizerSpec, return_cuts: bool = False, tokenized_text_filepath: Optional[str] = None): # Add tokenized_text_filepath
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.return_cuts = return_cuts
        # Initialize TextOnlyDataset if tokenized_text_filepath is provided
        self.text_only_dataset = None
        if tokenized_text_filepath is not None:
            # It's assumed that the BPE dataset is used for training, so we check for 'train_txt_ds'
            # This is a bit of a hack, ideally the config would pass a flag.
            # We check if the cuts object (which is a CutSet in Lhotse) has a name attribute and if it's 'train_txt_ds'.
            # This check might need to be more robust depending on how Lhotse is used.
            # For now, we'll assume if tokenized_text_filepath is given, we load TextOnlyDataset.
            self.text_only_dataset = TextOnlyDataset(tokenized_text_filepath, tokenizer)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts_with_audio = self.load_audio(cuts) # Rename cuts to cuts_with_audio for clarity
        tokens = [
            torch.cat(
                [
                    torch.as_tensor(s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language))
                    for s in c.supervisions
                ],
                dim=0,
            )
            for c in cuts_with_audio # Use cuts_with_audio here
        ]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

        # If text_only_dataset exists, get unpaired text data
        if self.text_only_dataset is not None:
            # We need to sample from text_only_dataset. The number of samples should match the audio batch size.
            # This ensures that each batch item from the audio dataset has a corresponding text-only sample.
            # Note: This might not be the most efficient way if len(self.text_only_dataset) is very different from len(cuts_with_audio)
            # or if specific pairing strategies are needed. This is a simple random sampling approach.
            num_audio_samples = audio.size(0)
            text_indices = torch.randint(0, len(self.text_only_dataset), (num_audio_samples,)).tolist()
            unpaired_tokens_list = [self.text_only_dataset[i] for i in text_indices] # This now returns tensors directly
            unpaired_token_lens = torch.tensor([t.size(0) for t in unpaired_tokens_list], dtype=torch.long)
            unpaired_tokens = collate_vectors(unpaired_tokens_list, padding_value=self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') and self.tokenizer.pad_id is not None else 0)


            if self.return_cuts:
                return audio, audio_lens, tokens, token_lens, unpaired_tokens, unpaired_token_lens, cuts_with_audio.drop_in_memory_data()
            return audio, audio_lens, tokens, token_lens, unpaired_tokens, unpaired_token_lens

        if self.return_cuts:
            return audio, audio_lens, tokens, token_lens, cuts_with_audio.drop_in_memory_data()
        return audio, audio_lens, tokens, token_lens
