#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import random
import time 
import torch
from IPython.display import HTML
import re
import textwrap

from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoConfig
import torch
from NLP_prompt import Prompting

import warnings
warnings.filterwarnings("ignore")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


model_name="roberta-large"
model_path=model_name.split("-")[0]+"_"+model_name.split("-")[1]+"_"+"repo"
model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/few-shot/prompt-learning/fine-tune-LM",model_path)
config=AutoConfig.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=config.max_position_embeddings-2)
print(tokenizer.mask_token, tokenizer.model_max_length,config.max_position_embeddings)        

prompting= Prompting(model=model_path)


# In[3]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# In[4]:


prefix_prompt="email review:"
post_prompt="The emotion of this email is " + f" {tokenizer.mask_token} ."
# prompt=[prefix_prompt,post_prompt]


# In[5]:


from IPython.core.display import HTML

keyword = r"The emotion of this email is <mask>"
color = "red"
style = "font-weight:bold;"

highlighted_text = f"The emotion of this email is <span style='color:{color};{style}'> &lt;mask&gt; </span>"
HTML(highlighted_text)


# In[ ]:


from IPython.core.display import HTML

keyword = "The emotion of this email is &lt;mask&gt;"
color = "red"
style = "font-weight:bold;"

highlighted_text = f"<span style='color:{color};{style}'>{keyword}</span>"
HTML(highlighted_text)


# In[6]:


prompt=[post_prompt]
threshold=prompting.compute_tokens_prob("",prompt=prompt, token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:





# In[29]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="Last Wednesday you indicated Lauren would be in touch with me and I  still  not hear from her. We have lost 5 weeks on this already, which is very disappointing. "+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:


# post_prompt=f"This email was  {tokenizer.mask_token}"
# threshold=prompting.compute_tokens_prob("",prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
# threshold


# In[10]:


text="""
Last Wednesday you indicated Lauren would be in touch with me and I  still  not hear from her. We have lost 5 weeks on this already, which is very disappointing. 
"""

threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive","neutral"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:


# threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
# print()
# print(threshold)
# print()
# print(prompting.prompt_text)


# In[13]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="I am not getting my excess retirement checks and the person and place they tell me to contact never responds to my numerous phone calls. "+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[30]:


text="I am not getting my excess retirement checks and the person and place they tell me to contact never responds to my numerous phone calls."
threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[15]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="Dear Ms. Tyler, Several employees at JP Morgan have concerns about inappropriate behavior and comments by Anthony Furia. He has made sexist and racist comments. He works on the commercial mortgage lending team in New York as a client associate. He also has many client complaints and acts in an unprofessional manner. We hope you will address this. Thank you for your time. "+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[17]:


text="Dear Ms. Tyler, Several employees at JP Morgan have concerns about inappropriate behavior and comments by Anthony Furia. He has made sexist and racist comments. He works on the commercial mortgage lending team in New York as a client associate. He also has many client complaints and acts in an unprofessional manner. We hope you will address this. Thank you for your time. "
threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:





# In[18]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="Hey John, Thanks for forwarding. The property has 4 residential units. We need a minimum of 5 to be able to quote – even if there’s commercial. So we wouldn’t be the best fit for this one. Appreciate you thinking of us and hope you’re enjoying the end of summer. Nate Yaghoubi's Recent Transactions Linkedin Profile Subject: 20 Bleecker St Nate This borrower has COVID problems and has a very bad VOM. Would you be able to quote on this, or is that too much hair. Please let me know. John Meagher"+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[19]:


text="Hey John, Thanks for forwarding. The property has 4 residential units. We need a minimum of 5 to be able to quote – even if there’s commercial. So we wouldn’t be the best fit for this one. Appreciate you thinking of us and hope you’re enjoying the end of summer. Nate Yaghoubi's Recent Transactions Linkedin Profile Subject: 20 Bleecker St Nate This borrower has COVID problems and has a very bad VOM. Would you be able to quote on this, or is that too much hair. Please let me know. John Meagher"
threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:





# In[21]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="The service is awesome and fatanstic."+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[26]:


text="""
The service is awesome and fatanstic. 
"""
threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:





# In[33]:


keyword=r"The emotion of this email is  &lt;mask&gt; ."
color = "red"
style="font-weight:bold;"

text="""
Hi Mark, I just returned from lunch. My phone log is showing 2 missed calls from CHASE yet no one left a message. Not sure who called and for what
reason. Trying to settle many open issues with CHASE as we have opened 5 new accounts with different entities within the last 2 mos. This is becoming
frustrating - between no one leaving messages, unsecured email  Can you please help me to determine who called and for what reason? Sharon Sharon K
Mitchell Corporate Controller Hostmark Hospitality Group “Unity is strength. Where there is teamwork and collaboration, wonderful things can be
achieved.” –Mattie JT Stepanek 1300 E. Woodfield Road, Suite 400 · Schaumburg, IL 60173 Office:  847-517-9100  ·  Direct:  847-915-4539  [] []
"""+keyword 
wrapper = textwrap.TextWrapper(width=80) 
highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[32]:


text="""
Hi Mark, I just returned from lunch. My phone log is showing 2 missed calls from CHASE yet no one left a message. Not sure who called and for what
reason. Trying to settle many open issues with CHASE as we have opened 5 new accounts with different entities within the last 2 mos. This is becoming
frustrating - between no one leaving messages, unsecured email  Can you please help me to determine who called and for what reason? Sharon Sharon K
Mitchell Corporate Controller Hostmark Hospitality Group “Unity is strength. Where there is teamwork and collaboration, wonderful things can be
achieved.” –Mattie JT Stepanek 1300 E. Woodfield Road, Suite 400 · Schaumburg, IL 60173 Office:  847-517-9100  ·  Direct:  847-915-4539  [] []
"""
threshold=prompting.compute_tokens_prob(text,prompt=[post_prompt], token_list1=["positive"], token_list2= ["negative"],device=device)
prob=torch.nn.functional.softmax(threshold.unsqueeze(0), dim=1)
print("{:<60}{:<20.0%}".format("The probability of negative sentiment for prompt only text: ",prob.squeeze()[1].item()))
print("{:<60}{:<20.0%}".format("The probability of positive sentiment for prompt only text: ",prob.squeeze()[0].item()))


# In[ ]:





# In[ ]:




