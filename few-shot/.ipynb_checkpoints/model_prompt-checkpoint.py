from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoConfig
import torch
import wrapper

class Prompting(object):
    """ doc string 
    This class helps us to implement
    Prompt-based Learning Model
    """
    def __init__(self, **kwargs):
        """ constructor 
        parameter:
        ----------
           model: AutoModelForMaskedLM
                path to a Pre-trained language model form HuggingFace Hub
           tokenizer: AutoTokenizer
                path to tokenizer if different tokenizer is used, 
                otherwise leave it empty
        """
        model_path=kwargs['model']
        
        if "tokenizer" in kwargs.keys():
            tokenizer_path= kwargs['tokenizer']
        else:
            tokenizer_path= kwargs['model']
            
        self.config=AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,model_max_length=self.config.max_position_embeddings-2)
        
        
    def compute_max_seq_len(self) -> int:
        """Compute the maximum sequence length of an input sequence.
        Large input sequences need to be truncated while keeping the
        prompt template untruncated.
        Returns:
            The maximum sequence length in number of tokens.
        """
        
        template_len=len(self.tokenizer(self.prompt,add_special_tokens=False).input_ids)
        
        n_special_chars = self.tokenizer.num_special_tokens_to_add(False)

        max_seq_len = self.tokenizer.model_max_length \
            - (template_len + n_special_chars)

        assert max_seq_len > 0

        return max_seq_len
    
    def prompt_pred(self,text,prompt,token_list1, token_list2, device):
        """
          Predict MASK token by listing the probability of candidate tokens 
          where the first token is the most likely
          Parameters:
          ----------
          text: str 
              The text including [MASK] token.
              It only supports single MASK token. If more [MASK]ed tokens 
              are given, it takes the first one.
          Returns:
          --------
          list of (token, prob)
             The return is a list of all token in LM Vocab along with 
             their prob score, sort by score in descending order 
        """
        threshold=self.benchmark_prompt(prompt, token_list1, token_list2, device)
        pred_out=[]
        
        wrapped_text=wrapper.prompt_wrapper(text, prompt, self.tokenizer)
        wrapped_text=wrapped_text.map(lambda x:self.tokenizer(x['wrapped_email'],return_tensors="pt"), batched=True)
        
        wrapped_text.set_format("pandas")
        df=wrapped_text[:]
        
        for index,row in tqdm(df.iterrows(), total=df.shape[0]):
            input={"input_ids":row["input_ids"],'attention_mask':row['attention_mask']}
            input={a:torch.from_numpy(b).unsqueeze(0).to(device) for a,b in input.items()}
            mask_pos=row["input_ids"].tolist().index(self.tokenizer.mask_token_id)
            model.eval()
            with torch.no_grad():
                self.model=self.model.to(device)
                outputs = self.model(**input)
                predictions = outputs[0]
            values, indices=torch.sort(predictions[0, mask_pos],  descending=True)
            result=list(zip(tokenizer.convert_ids_to_tokens(indices), values))
            scores_dict={a:b for a,b in result}
            
            softmax_rt=self.compute_tokens_prob(scores_dict, token_list1, token_list2)
            
            if softmax_rt[1]>threshold[1]:
                pred_out.append((row["target"],1))
            else:
                pred_out.append((row["target"],0))
            
        return pred_out

    def benchmark_prompt(self,prompt, token_list1, token_list2, device):

        if len(prompt)==1:
            indexed_prompt=self.tokenizer(prompt[0], return_tensors="pt",add_special_tokens=False).input_ids
        elif len(prompt)>1:
            x_v1=self.tokenizer(prompt[0], return_tensors="pt",add_special_tokens=False).input_ids
            x_v2=self.tokenizer(prompt[1], return_tensors="pt",add_special_tokens=False).input_ids
            indexed_prompt=torch.cat([x_v1, x_v2],dim=1)
            
        tokenized_text= self.tokenizer.decode(indexed_prompt[0].to(torch.long))
        indexed_tokens=self.tokenizer(tokenized_text, return_tensors="pt",add_special_tokens=True).input_ids   ### add bos and eos tokens in the text
        
        # take the first masked token
        mask_pos=indexed_tokens[0].tolist().index(self.tokenizer.mask_token_id)
        self.model.eval()
        with torch.no_grad():
            self.model=self.model.to(device)
            indexed_tokens=indexed_tokens.to(device)
            outputs = self.model(indexed_tokens)
            predictions = outputs[0]
        values, indices=torch.sort(predictions[0, mask_pos],  descending=True)
        #values=torch.nn.functional.softmax(values, dim=0)
        result=list(zip(self.tokenizer.convert_ids_to_tokens(indices), values))
        scores_dict={a:b for a,b in result}        
        threshold=self.compute_tokens_prob(scores_dict, token_list1, token_list2)
        return threshold
                         
    def compute_tokens_prob(self, scores_dict, token_list1, token_list2):
        """
        Compute the activations for given two token list, 
        Parameters:
        ---------
        token_list1: List(str)
         it is a list for positive polarity tokens such as good, great. 
        token_list2: List(str)
         it is a list for negative polarity tokens such as bad, terrible.      
        Returns:
        --------
        Tuple (
           the probability for first token list,
           the probability of the second token list,
           the ratio score1/ (score1+score2)
           The softmax returns
           )
        """
        score1=[scores_dict[token1] if token1 in scores_dict.keys() else 0 for token1 in token_list1]
        score1= sum(score1)/len(token_list1)
        
        score2=[scores_dict[token2] if token2 in scores_dict.keys() else 0 for token2 in token_list2]
        score2= sum(score2)/len(token_list2)
        
        softmax_rt=torch.nn.functional.softmax(torch.Tensor([score1,score2]), dim=0)
        return softmax_rt        
    
    def fine_tune(self, sentences, labels, prompt=["[CLS] email :",". this email was [MASK] sentiment [SEP]"],goodToken="positive",badToken="negative"):
        """  
          Fine tune the model
        """
        good=tokenizer.convert_tokens_to_ids(goodToken)
        bad=tokenizer.convert_tokens_to_ids(badToken)

        from transformers import AdamW
        optimizer = AdamW(self.model.parameters(),lr=1e-3)

        for sen, label in zip(sentences, labels):
            if len(prompt)==1:
                new_sen=sen+prompt[0]
            elif len(prompt)>1:
                new_sen=prompt[0]+sen+prompt[1]
            tokenized_text = self.tokenizer.tokenize(new_sen)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            # take the first masked token
            mask_pos=tokenized_text.index(self.tokenizer.mask_token)
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            pred=predictions[0, mask_pos][[good,bad]]
            prob=torch.nn.functional.softmax(pred, dim=0)
            lossFunc = torch.nn.CrossEntropyLoss()
            loss=lossFunc(prob.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            optimizer.step()
        print("done!")
