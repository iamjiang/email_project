import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer , AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import os

class ModelForSequenceClassification(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_checkpoint)
        self.config=AutoConfig.from_pretrained(model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,model_max_length=self.config.max_position_embeddings)
        self.model=AutoModel.from_pretrained(model_name)
        
        if args.frozen_layers!=0:
            modules = [self.model.base_model.embeddings, *self.model.base_model.encoder.layer[:args.frozen_layers]] 
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            
        if args.frozen_all:
            for param in self.model.parameters():
                param.requires_grad = False
            
            
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_labels)

        # self.init_weights()
        
        model_param = 0
        for name, param in self.model.named_parameters():
            model_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - model_param
        # print('total param is {}'.format(total_param))
        print()
        print(f"The maximal # input tokens : {self.tokenizer.model_max_length:,}")
        print(f"Vocabulary size : {self.tokenizer.vocab_size:,}")
        if args.frozen_all:
            print(f"The # of parameters to be updated : {total_param:,}")
        else:
            print(f"The # of parameters to be updated : {all_param:,}")
        print()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if token_type_ids is not None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask
            )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )