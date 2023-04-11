#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML
import re
import textwrap
import random


# In[2]:


df_train=pd.read_pickle(os.path.join(os.getcwd(),"df_train"))
df_val=pd.read_pickle(os.path.join(os.getcwd(),"df_val"))
df_test=pd.read_pickle(os.path.join(os.getcwd(),"df_test"))


# In[3]:


negative_word=[]
with open("negative-words.txt") as f:
    for curline in f:
        if curline.startswith(";"):
            continue
        if curline.strip():
            negative_word.append(curline.strip())
            
print()
print("There are {:,} negative words externally".format(len(negative_word)))
print()


# In[4]:


df_train['negative_word_set']=df_train["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))
df_val['negative_word_set']=df_val["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))
df_test['negative_word_set']=df_test["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))


train_complaint,  train_no_complaint=df_train[df_train['target']==1], df_train[df_train['target']==0]
val_complaint,  val_no_complaint=df_val[df_val['target']==1], df_val[df_val['target']==0]
test_complaint,  test_no_complaint=df_test[df_test['target']==1], df_test[df_test['target']==0]

def most_common_word(df,feature):
    word_count=Counter()
    for index,row in tqdm(df.iterrows(), total=df.shape[0]):
        if isinstance(row[feature],list):
            word_count.update(set(row[feature].split()))
        elif isinstance(row[feature],set):
            word_count.update(row[feature])
    word,freq=zip(*word_count.most_common())
    return word,freq

word_train_complaint, freq_train_complaint = most_common_word(train_complaint, feature="negative_word_set")
word_val_complaint, freq_val_complaint = most_common_word(val_complaint, feature="negative_word_set")
word_test_complaint, freq_test_complaint = most_common_word(test_complaint, feature="negative_word_set")

word_train_no_ccomplaint, freq_train_no_complaint = most_common_word(train_no_complaint, feature="negative_word_set")
word_val_no_ccomplaint, freq_val_no_complaint = most_common_word(val_no_complaint, feature="negative_word_set")
word_test_no_ccomplaint, freq_test_no_complaint = most_common_word(test_no_complaint, feature="negative_word_set")

# keyword_training=[w for w in word_train_complaint if w not in word_train_no_churn]
# keyword_test=[w for w in word_test_complaint if w not in word_test_no_churn]

dict_data={}
dict_data["training"]=word_train_complaint[0:50]
dict_data["validation"]=word_val_complaint[0:50]
dict_data["test"]=word_test_complaint[0:50]
pd.DataFrame(dict_data).style.format().set_caption("Most common negative sentiment word in complaint==1").set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}]) 


# In[55]:


# dict_data_transpose=pd.DataFrame(dict_data).transpose()
pd.DataFrame(dict_data_transpose).style.format().set_caption("Most common negative sentiment word in complaint==1").set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}]) 


# In[5]:


# word=["frustration","frustrated","frustrate","unacceptable","disappointed","disappointing","unhappy","misunderstanding"]
word=set(word_train_complaint[0:20]).difference(set(word_train_no_ccomplaint[0:50]))
print()
print(word)
print()
# tempt["negative_word"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(word).intersection(set(x.split())))!=0 else 0 )
df_train["negative_word"]=df_train["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
df_val["negative_word"]=df_val["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
df_test["negative_word"]=df_test["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )

tempt1=df_train.copy()
tempt1["data_type"]=["training_set"]*len(tempt1)
tempt2=df_val.copy()
tempt2["data_type"]=["validation_set"]*len(tempt2)
tempt3=df_test.copy()
tempt3["data_type"]=["test_set"]*len(tempt3)
tempt=pd.concat([tempt1,tempt2,tempt3],axis=0)

plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
ax = sns.barplot(data = tempt, x='negative_word',y='target',hue="data_type")
ax.set_title(" Negative word {'disappointed', 'ridiculous', 'unacceptable'} in email", fontsize=16)
ax.set_xticklabels(["no negative word", "negative word exist"])
ax.set_ylabel("complaint rate", fontsize=16)
ax.set_xlabel("")
plt.legend(loc="best")


# In[57]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

keyword='upset'
# keyword=['disappointed', 'ridiculous', 'unacceptable','upset','frustration','frustrated']
color = "red"
style="font-weight:bold;"
# df_all = pd.concat([df_train, df_val, df_test], axis=0)
df_all = df_train.copy()

tempt=df_all[df_all["target"]==0]
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(('disappointed', 'unacceptable')).intersection(x))>0 else 0)
# tempt["not_complaint_negative"].value_counts(dropna=False)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if keyword in x.split() else 0)
tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in x.split()  for item in keyword)) else 0)

tempt=tempt[tempt["not_complaint_negative"]==1]
# tempt[tempt.text_length<200].shape
tempt=tempt[tempt.text_length<300]

import textwrap
import random

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
for i in range(100):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
 
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
    print()
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+"not complaint"+bcolors.ENDC))
    print()
    text=exam_text.loc[j,"preprocessed_email"]
    # highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
    # display(HTML(wrapper.fill(highlighted_text)))
    pattern = "|".join(keyword)
    highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
    display(HTML(wrapper.fill(highlighted_text)))


# In[52]:


import warnings
warnings.simplefilter("ignore")
keyword=['disappointed', 'ridiculous', 'unacceptable','upset','frustration','frustrated']
tempt=df_all[df_all["target"]==0]
tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in x.split()  for item in keyword)) else 0)
tempt=tempt[tempt["not_complaint_negative"]==1]
# print("{:<80}{:<30,}".format("Total # of non-complaint emails", df_all[df_all["target"]==0].shape[0]))
print("{:<80}{:<30,}".format("The # of emails containing negative keyword but aren't labelled as complaint", tempt.shape[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# keyword='upset'
keyword=['frustration',  'unacceptable']
color = "red"
style="font-weight:bold;"

# df_all = pd.concat([df_train, df_val, df_test], axis=0)
df_all=df_train.copy()

tempt=df_all[df_all["target"]==0]
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(('disappointed', 'unacceptable')).intersection(x))>0 else 0)
# tempt["not_complaint_negative"].value_counts(dropna=False)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if keyword in x.split() else 0)
tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and all(item in x.split()  for item in keyword)) else 0)

tempt=tempt[tempt["not_complaint_negative"]==1]
# tempt[tempt.text_length<200].shape
tempt=tempt[tempt.text_length<200]

import textwrap
import random

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
if exam_text.shape[0]>10:
    for i in range(10):
        random.seed(101+i)
        j = random.choice(exam_text.index)

        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+"not complaint"+bcolors.ENDC))
        print()
        text=exam_text.loc[j,"preprocessed_email"]

        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))
else:
    for i in range(exam_text.shape[0]):
        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+"not complaint"+bcolors.ENDC))
        print()
        text=exam_text.loc[i,"preprocessed_email"]

        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))


# In[38]:


tempt.shape, exam_text.shape, i


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

keyword=['disappointed', 'ridiculous', 'unacceptable']
color = "red"
style="font-weight:bold;"

tempt=df_train[df_train["target"]==1]
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(('disappointed', 'unacceptable')).intersection(x))>0 else 0)
# tempt["not_complaint_negative"].value_counts(dropna=False)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if keyword in x.split() else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and all(item in x.split()  for item in keyword)) else 0)
tempt["complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in keyword for item in x.split())) else 0)
tempt=tempt[tempt["complaint_negative"]==1]
# tempt[tempt.text_length<200].shape
tempt=tempt[tempt.text_length<300]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
if exam_text.shape[0]>10:
    for i in range(10):
        random.seed(101+i)
        j = random.choice(exam_text.index)

        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"is_complaint"]+bcolors.ENDC))
        print()

        text=exam_text.loc[j,"preprocessed_email"]
        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))
else:
    for i in range(exam_text.shape[0]):
        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"is_complaint"]+bcolors.ENDC))
        print()

        text=exam_text.loc[j,"preprocessed_email"]
        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))


# In[60]:


exam_text.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


text="""
Punit, My manager, Maddie Hamilton, and I are available at 9 AM tomorrow. I will circulate an invite shortly. 
Subject: RE: Follow up Good Morning Punit, I certainly understand your frustration here. We are working expeditiously to settle this dispute 
within the confines of the bank’s policies and procedures related to fraudulent claims filed. Let me coordinate with the 
internal team here to see which of these times work best. Subject: Follow up Hi Chad – Lisa brought to my attention that the matter 
of unauthorized ACH withdrawals from Candid’s account has not been resolved and that Chase representatives have our accounting team 
following up on this and spending hours on the phone chasing the trail of these withdrawals and deposits. 
This is unacceptable. We were assured that this matter would be resolved when we met on August 4 as were within the 72 hour window of 
these transactions. I would like to meet with you and your superiors who can resolve this for us and not have my team running around in 
circles. I am available on Wednesday, August 17 from 4-5pm and on Thursday, August 18 from 9-10am. Please let me know which of these times 
is convenient for you and your manager to meet with us and I shall set up a Zoom meeting. “My work hours may not be yours. 
Please do not feel obligated to respond outside of your normal work hours.” Punit Kohli Vice President of Finance
"""


# In[22]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

wrapper = textwrap.TextWrapper(width=150) 
    
keyword=['disappointed', 'frustration', 'unacceptable', 'withdrawals', 'dispute', 'resolved','resolve', 'fraudulent','settle','obligated']
color = "red"
style="font-weight:bold;"

pattern = "|".join(keyword)
highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[44]:


from IPython.display import HTML
import re
import textwrap

text="""
Punit, My manager, Maddie Hamilton, and I are available at 9 AM tomorrow. I will circulate an invite shortly. 
Subject: RE: Follow up Good Morning Punit, I certainly understand your frustration here. We are working expeditiously to settle this dispute 
within the confines of the bank’s policies and procedures related to fraudulent claims filed. Let me coordinate with the 
internal team here to see which of these times work best. Subject: Follow up Hi Chad – Lisa brought to my attention that the matter 
of unauthorized ACH withdrawals from Candid’s account has not been resolved and that Chase representatives have our accounting team 
following up on this and spending hours on the phone chasing the trail of these withdrawals and deposits. 
This is unacceptable. We were assured that this matter would be resolved when we met on August 4 as were within the 72 hour window of 
these transactions. I would like to meet with you and your superiors who can resolve this for us and not have my team running around in 
circles. I am available on Wednesday, August 17 from 4-5pm and on Thursday, August 18 from 9-10am. Please let me know which of these times 
is convenient for you and your manager to meet with us and I shall set up a Zoom meeting. “My work hours may not be yours. 
Please do not feel obligated to respond outside of your normal work hours.” Punit Kohli Vice President of Finance
"""

keyword=['disappointed', 'frustration', 'unacceptable', 'withdrawals', 'dispute', 'resolved','resolve', 'fraudulent','settle','obligated']
for v in keyword:
    text=text.replace(v,'[MASK]')
    
wrapper = textwrap.TextWrapper(width=150)
keyword='[MASK]'
color = "green"
style="font-weight:bold;"

highlighted_text = re.sub(r'\[MASK\]', f"<span style='color:{color};{style}'>{keyword}</span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:





# In[ ]:





# In[6]:


#### complaint email demo


# In[15]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
tempt=df_train[df_train["target"]==1]
tempt=tempt[tempt.text_length<200]

import textwrap
import random

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
for i in range(10):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
 
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
    print()
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"is_complaint"]+bcolors.ENDC))
    print()
    text=exam_text.loc[j,"preprocessed_email"]
    # highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
    display(HTML(wrapper.fill(text)))


# In[ ]:





# In[ ]:


tempt_df=df_train[df_train.snapshot_id=="journal-18f263dbc0f5014cb56e3afce559a386-1669934861000"]
tempt_df


# In[14]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
keyword=['concerns', 'inappropriate', 'sexist','racist','complaints','unprofessional']
color = "red"
style="font-weight:bold;"

tempt_df=df_train[df_train.snapshot_id=="journal-18f263dbc0f5014cb56e3afce559a386-1669934861000"]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt_df.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

print('')
print("*"*50)
print('*********  preprocessed_email ********')
print("*"*50)

print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"snapshot_id"]+bcolors.ENDC))
print()
print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"is_complaint"]+bcolors.ENDC))
print()

text=exam_text.loc[0,"preprocessed_email"]
pattern = "|".join(keyword)
highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:





# In[20]:


keyword=['resent', 'not', 'happy']
color = "red"
style="font-weight:bold;"

tempt_df=df_train[df_train.snapshot_id=="journal-c984fdc30ba23559421d2828bcfaee16-1660009984000"]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt_df.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

print('')
print("*"*50)
print('*********  preprocessed_email ********')
print("*"*50)

print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"snapshot_id"]+bcolors.ENDC))
print()
print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"is_complaint"]+bcolors.ENDC))
print()

text=exam_text.loc[0,"preprocessed_email"]
pattern = "|".join(keyword)
highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:





# In[23]:


keyword=['can\'t','see', 'why', 'unhappy', 'ASAP' ]
color = "red"
style="font-weight:bold;"

tempt_df=df_train[df_train.snapshot_id=="journal-f49f31c1e7befb1dbaace6e9fc8a1728-1670433300000"]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt_df.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

print('')
print("*"*50)
print('*********  preprocessed_email ********')
print("*"*50)

print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"snapshot_id"]+bcolors.ENDC))
print()
print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[0,"is_complaint"]+bcolors.ENDC))
print()

text=exam_text.loc[0,"preprocessed_email"]
pattern = "|".join(keyword)
highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#### non-complaint email demo


# In[21]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
tempt=df_train[df_train["target"]==0]
tempt=tempt[tempt.text_length<200]

import textwrap
import random

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
for i in range(20):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
 
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
    print()
    print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"is_complaint"]+bcolors.ENDC))
    print()
    text=exam_text.loc[j,"preprocessed_email"]
    # highlighted_text = re.sub(keyword, f"<span style='color:{color};{style}'>{keyword}</span>", text)
    display(HTML(wrapper.fill(text)))


# In[ ]:





# In[24]:


## complaint email containing positive word


# In[30]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

keyword=['delighted', 'happy', 'satisfied','awesome' ,'fatanstic','great']
color = "red"
style="font-weight:bold;"

tempt=df_train[df_train["target"]==1]
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(('disappointed', 'unacceptable')).intersection(x))>0 else 0)
# tempt["not_complaint_negative"].value_counts(dropna=False)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if keyword in x.split() else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and all(item in x.split()  for item in keyword)) else 0)
tempt["complaint_positive"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in keyword for item in x.split())) else 0)
tempt=tempt[tempt["complaint_positive"]==1]
# tempt[tempt.text_length<200].shape
tempt=tempt[tempt.text_length<300]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]
exam_text.reset_index(drop=True,inplace=True)

exam_text.reset_index(drop=True,inplace=True)

# Randomly choose some examples.
if exam_text.shape[0]>10:
    for i in range(10):
        random.seed(101+i)
        j = random.choice(exam_text.index)

        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[j,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"is_complaint"]+bcolors.ENDC))
        print()

        text=exam_text.loc[j,"preprocessed_email"]
        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))
else:
    for i in range(exam_text.shape[0]):
        print('')
        print("*"*50)
        print('*********  preprocessed_email ********')
        print("*"*50)

        print("{:<30}{:<50}".format(bcolors.OKBLUE+"snapshot_id : "+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"snapshot_id"]+bcolors.ENDC))
        print()
        print("{:<30}{:<50}".format(bcolors.OKBLUE+"Ground True Label"+bcolors.ENDC, bcolors.WARNING+exam_text.loc[i,"is_complaint"]+bcolors.ENDC))
        print()

        text=exam_text.loc[i,"preprocessed_email"]
        pattern = "|".join(keyword)
        highlighted_text = re.sub(pattern, f"<span style='color:{color};{style}'>\g<0></span>", text)
        display(HTML(wrapper.fill(highlighted_text)))


# In[ ]:




