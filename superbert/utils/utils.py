import numpy as np
import math
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch


def torch_mask_tokens(inputs: Any,mlm_probability=0.15, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        labels = torch.ones((inputs.shape[0], inputs.shape[1])).to(inputs.device)#inputs).clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # if special_tokens_mask is None:
        #     special_tokens_mask = [
        #         self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        #     ]
        #     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # else:
        #     special_tokens_mask = special_tokens_mask.bool()

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = 0  # We only compute loss on masked tokens

        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = torch.zeros(inputs.shape[2]) #self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randn((torch.sum(indices_random==True), inputs.shape[2]))#torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        
        # inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return labels

