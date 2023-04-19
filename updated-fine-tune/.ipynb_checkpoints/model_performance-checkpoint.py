#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# In[2]:


# model_name="roberta-large"
# output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
# output_dir=os.path.join(os.getcwd(),output_dir,"val_recall")

# with open(os.path.join(output_dir,"y_true_pred.txt"),"r") as file:
#     true_y=[]
#     pred_y=[]
#     prob_y=[]
#     for line in file:
#         x,y,z=line.strip().split(',')
#         true_y.append(int(x))
#         pred_y.append(int(y))
#         prob_y.append(float(z))


# In[3]:


# print()
# print(f"\n===========Test Set Performance===============\n")
# print()

# print(classification_report(true_y, pred_y))
# print()
# print()
# print(confusion_matrix(true_y, pred_y))  


# In[38]:


def metric_table(table_name="metrics_test.txt"):
    Model_Type=[]
    Total_Complaint=[]
    False_Positive=[]
    False_Negative=[]
    Precision=[]
    Recall=[]
    F1_Score=[]
    ROC_AUC=[]
    PR_AUC=[]

    with open(table_name,'r') as f:
        for line in f:
            Model_Type.append(str(line.split(",")[0]))
            Total_Complaint.append(int(line.split(",")[1]))
            False_Positive.append(int(line.split(",")[2]))
            False_Negative.append(int(line.split(",")[3]))
            Precision.append(float(line.split(",")[4]))
            Recall.append(float(line.split(",")[5]))
            F1_Score.append(float(line.split(",")[6]))
            ROC_AUC.append(float(line.split(",")[7]))
            PR_AUC.append(float(line.split(",")[8]))

    metrics=pd.DataFrame({"model_type":Model_Type,"total complaint #":Total_Complaint,"false_positive":False_Positive,"false_negative":False_Negative,                         "precision":Precision,"recall":Recall,"f1_score":F1_Score,"roc_auc":ROC_AUC,"pr_auc":PR_AUC})
    # metrics.drop_duplicates(subset=["model_type","epoch"],inplace=True)
    # metrics.sort_values(by=['model_type','epoch'],inplace=True)       
    
    return metrics

def style_format(metrics, type="test set"):
    # metrics=metrics[metrics["model_type"].apply(lambda x : x.split("-")[0]==model.split("-")[0])].reset_index(drop=True)
    return metrics.style.format({"total complaint #":"{:,}","false_positive":"{:,}","false_negative":"{:,}", "precision":"{:.2%}", "recall":"{:.2%}",                                 "f1_score":"{:.2%}", "roc_auc":"{:.2%}","pr_auc":"{:.2%}"})     .set_caption(f"Performance Summary For {type} ")     .set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'red'),
            ('font-size', '20px')
        ]
    }])


# In[40]:


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_90")
metric_bert=metric_table(table_name=os.path.join(output_dir,"metrics_test.txt"))


# In[41]:


model_name="longformer-base"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
metric_longformer_base=metric_table(table_name=os.path.join(output_dir,"metrics_test.txt"))


# In[42]:


model_name="longformer-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
# output_dir=os.path.join(os.getcwd(),output_dir)
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
metric_longformer_large=metric_table(table_name=os.path.join(output_dir,"metrics_test.txt"))


# In[43]:


output_dir=os.path.join('/opt/omniai/work/instance1/jupyter/email-complaints/lightgbm+transformers', "tfidf+structure")
metric_lightgbm=metric_table(table_name=os.path.join(output_dir,"lightgbm_metrics_test.txt"))
metric_xgboost=metric_table(table_name=os.path.join(output_dir,"xgboost_metrics_test.txt"))
metric_catboost=metric_table(table_name=os.path.join(output_dir,"catboost_metrics_test.txt"))
metric_randomforest=metric_table(table_name=os.path.join(output_dir,"randomforest_metrics_test.txt"))


# In[44]:


# metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_catboost,metric_lightgbm,metric_bert], axis=0)
# metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_catboost,metric_lightgbm,metric_bert,metric_longformer], axis=0)
metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_lightgbm,metric_bert,metric_longformer_base,metric_longformer_large], axis=0)
# metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_lightgbm,metric_bert], axis=0)
metric_test['model_type'] = metric_test['model_type'].replace(['roberta-large'], 'BERT-large')
metric_test['model_type'] = metric_test['model_type'].replace(['longformer-base-4096'], 'longformer-base')
metric_test['model_type'] = metric_test['model_type'].replace(['longformer-large-4096'], 'longformer-large')
metric_test['model_type'] = metric_test['model_type'].replace(['randomforest'], 'random-forest')
metric_test.drop_duplicates(subset=['model_type'],inplace=True)
style_format(metric_test,  type="test set")


# In[45]:


metric_test = pd.concat([metric_bert,metric_longformer_base,metric_longformer_large], axis=0)
metric_test['model_type'] = metric_test['model_type'].replace(['roberta-large'], 'BERT-large')
metric_test['model_type'] = metric_test['model_type'].replace(['longformer-base-4096'], 'longformer-base')
metric_test['model_type'] = metric_test['model_type'].replace(['longformer-large-4096'], 'longformer-large')
metric_test.drop_duplicates(subset=['model_type'],inplace=True)
style_format(metric_test,  type="test set")


# In[48]:


def response_rate_eval(logit,label,topk):
    DF=pd.DataFrame(columns=["pred_score","actual_label"])
    DF["pred_score"]=logit
    DF["actual_label"]=label
    DF.sort_values(by="pred_score", ascending=False, inplace=True)
    response_rate={}
    for p in topk:
        N=math.ceil(int(DF.shape[0]*p))
        DF2=DF.nlargest(N,"pred_score",keep="first")
        response_rate[str(int(p*100))+"%"]=DF2.actual_label.sum()/DF2.shape[0]
    return response_rate

from matplotlib.ticker import FuncFormatter
def bar_plot(data, colors=None, total_width=0.8, single_width=1, legend=True,title=None,subtitle=None,axis_truncation=0.5):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    
    fig, ax = plt.subplots(figsize =(15, 8))
    
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values.values()):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
    
    ax.set_ylabel('Accuracy',fontsize=15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: "{:.0%}".format(y)))
    ind=np.arange(len(data[list(data.keys())[0]]))
    ax.set_xticks(ind)
    ax.set_xticklabels( ('top 1% score', 'top 2% score', 'top 5% score','top 10% score') )
    ax.set_title(f"Top Predicted Score  ",fontsize=15)
    
    #     plt.xlim([0, 1])
    # plt.ylim([axis_truncation, 1])
    plt.show()


# In[49]:


def table_read(table_name):
    with open(table_name,"r") as file:
        true_y=[]
        prob_y=[]
        for line in file:
            x,y,z=line.strip().split(',')
            true_y.append(int(x))
            prob_y.append(float(z)) 
    return true_y, prob_y


# In[50]:


output_dir=os.path.join('/opt/omniai/work/instance1/jupyter/email-complaints/lightgbm+transformers', "tfidf+structure")
lightgbm_true, lightgbm_prob=table_read(table_name=os.path.join(output_dir,"lightgbm_y_true_pred.txt"))
xgboost_true, xgboost_prob=table_read(table_name=os.path.join(output_dir,"xgboost_y_true_pred.txt"))
randomforest_true, randomforest_prob=table_read(table_name=os.path.join(output_dir,"randomforest_y_true_pred.txt"))


# In[51]:


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_90")
table_name=os.path.join(output_dir,"y_true_pred.txt")
bert_true, bert_prob=table_read(table_name)


# In[52]:


model_name="longformer-base"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
table_name=os.path.join(output_dir,"y_true_pred.txt")
longformer_base_true, longformer_base_prob=table_read(table_name)


# In[53]:


model_name="longformer-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
table_name=os.path.join(output_dir,"y_true_pred.txt")
longformer_large_true, longformer_large_prob=table_read(table_name)


# In[54]:


topk=[0.01,0.02,0.05,0.1]

response_lightgbm = response_rate_eval(lightgbm_prob,lightgbm_true, topk)
response_xgboost = response_rate_eval(xgboost_prob,xgboost_true, topk)
response_randomforest = response_rate_eval(randomforest_prob, randomforest_true, topk)

response_bert = response_rate_eval(bert_prob, bert_true, topk)
response_longformer_base = response_rate_eval(longformer_base_prob, longformer_base_true, topk)
response_longformer_large = response_rate_eval(longformer_large_prob, longformer_large_true, topk)


if __name__ == "__main__":
    data = {
        "tfidf+lightgbm": response_lightgbm,
        "tfidf+xgboost": response_xgboost,
        "tfidf+random-forest": response_randomforest,
        "BERT-large": response_bert,
        "longformer-base": response_longformer_base,
        "longformer-large": response_longformer_large
        
    }

    
    CL=['r', 'g', 'b', 'c', 'y', 'darkorange', 'lime', 'grey','gold','bisque', 'lightseagreen', 'purple']
    bar_plot(data, colors=CL,total_width=.7, single_width=1,title="(response rate)",subtitle="Test Set ",axis_truncation=0.50)


# In[ ]:





# In[32]:


import os
import time
import datetime
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import logging

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc 
from sklearn.metrics import f1_score, precision_score, recall_score


# In[33]:


def read_output(file_name):
    with open(file_name,"r") as file:
        true_y=[]
        pred_y=[]
        prob_y=[]
        for line in file:
            x,y,z=line.strip().split(',')
            true_y.append(int(x))
            pred_y.append(int(y))
            prob_y.append(float(z))
    return true_y, pred_y, prob_y


# In[ ]:


output_dir=os.path.join('/opt/omniai/work/instance1/jupyter/email-complaints/lightgbm+transformers', "tfidf+structure")

file_name=os.path.join(output_dir,"xgboost_y_true_pred.txt")
xgboost_true_y, xgboost_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"lightgbm_y_true_pred.txt")
lightgbm_true_y, lightgbm_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"catboost_y_true_pred.txt")
catboost_true_y, catboost_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"randomforest_y_true_pred.txt")
randomforest_true_y, randomforest_pred_y, _ =read_output(file_name)


# In[35]:


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_90")
file_name=os.path.join(output_dir,"y_true_pred.txt")
roberta_true_y, roberta_pred_y, _ =read_output(file_name)


# In[36]:


model_name="longformer-base"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
file_name=os.path.join(output_dir,"y_true_pred.txt")
longformer_base_true_y, longformer_base_pred_y, _ =read_output(file_name)


# In[37]:


model_name="longformer-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),'updated-fine-tune',output_dir,"val_recall_95")
file_name=os.path.join(output_dir,"y_true_pred.txt")
longformer_large_true_y, longformer_large_pred_y, _ =read_output(file_name)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(xgboost_true_y, xgboost_pred_y)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("xgboost Model")
plt.show()


# In[ ]:


confusion_matrix = metrics.confusion_matrix(lightgbm_true_y, lightgbm_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("lightgbm Model")
plt.show()


# In[ ]:


confusion_matrix = metrics.confusion_matrix(randomforest_true_y, randomforest_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("Random Forest Model")
plt.show()


# In[ ]:


confusion_matrix = metrics.confusion_matrix(catboost_true_y, catboost_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("catboost Model")
plt.show()


# In[38]:


confusion_matrix = metrics.confusion_matrix(roberta_true_y, roberta_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("BERT-large Model")
plt.show()


# In[39]:


confusion_matrix = metrics.confusion_matrix(longformer_base_true_y, longformer_base_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("Longformer-base Model")
plt.show()


# In[40]:


confusion_matrix = metrics.confusion_matrix(longformer_large_true_y, longformer_large_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("Longformer-large Model")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def throughput_table(table_name="throughput.txt"):
    Model_Type=[]
    latency=[]
    throughput=[]
    device=[]

    with open(table_name,'r') as f:
        for line in f:
            Model_Type.append(str(line.split(",")[0]))
            latency.append(float(line.split(",")[1]))
            throughput.append(float(line.split(",")[2]))
            device.append(str(line.split(",")[3]))

    metrics=pd.DataFrame({"model_type":Model_Type,"time(second) per email":latency,"# of email per second":throughput,"device":device})
    metrics.drop_duplicates(subset=["model_type","device"],inplace=True)
    # metrics.sort_values(by=['model_type','epoch'],inplace=True)       
    
    return metrics

def style_format(metrics):
    return metrics.style.format({"time(second) per email":"{:.4f}","# of email per second":"{:.2f}"})     .set_caption("Latency of Different Model")     .set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'red'),
            ('font-size', '20px')
        ]
    }])


# In[25]:


data_dir="/opt/omniai/work/instance1/jupyter/email-complaints/fine-tuning/updated-fine-tune/latency"
df1=throughput_table(table_name=os.path.join(data_dir,"throughput.txt"))
df1['model_type'] = df1['model_type'].replace(['roberta-large'], 'BERT-large')
df1['model_type'] = df1['model_type'].replace(['longformer-base-4096'], 'longformer-base')
df1['model_type'] = df1['model_type'].replace(['longformer-large-4096'], 'longformer-large')
df1['device'] = df1['device'].replace("cuda\n",'one gpu')
df1['device'] = df1['device'].replace("multiple-gpus\n",'multi-gpus')
df1=df1.loc[:,['model_type','device','time(second) per email','# of email per second']]
style_format(df1)


# In[29]:


from transformers import AutoTokenizer, AutoModel, AutoConfig
root_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/fine-tune-LM")

bert_path=os.path.join(root_dir, "roberta_large_repo")
bert_tokenizer=AutoTokenizer.from_pretrained(bert_path)
bert_model=AutoModel.from_pretrained(bert_path)
bert_params=sum([p.nelement() for p in bert_model.parameters()])

LF_base_path=os.path.join(root_dir, "longformer_base_repo")
LF_base_tokenizer=AutoTokenizer.from_pretrained(LF_base_path)
LF_base_model=AutoModel.from_pretrained(LF_base_path)
LF_base_params=sum([p.nelement() for p in LF_base_model.parameters()])

LF_large_path=os.path.join(root_dir, "longformer_large_repo")
LF_large_tokenizer=AutoTokenizer.from_pretrained(LF_large_path)
LF_large_model=AutoModel.from_pretrained(LF_large_path)
LF_large_params=sum([p.nelement() for p in LF_large_model.parameters()])

df2=pd.DataFrame()
df2["model_type"]=["BERT-large","longformer-base","longformer-large"]
df2["maximal input word"]=[bert_tokenizer.model_max_length,LF_base_tokenizer.model_max_length,LF_large_tokenizer.model_max_length]
df2["parameter_num"]=[bert_params,LF_base_params,LF_large_params]
df2["memory_footprint(Megabyte)"]=[bert_params*4/(1024**2),LF_base_params*4/(1024**2),LF_large_params*4/(1024**2)]
df2.style.format({"maximal input word":"{:,}","parameter_num":"{:,}","memory_footprint(Megabyte)":"{:.0f}"})


# In[37]:


data_dir="/opt/omniai/work/instance1/jupyter/email-complaints/fine-tuning/v1-fine-tune/latency"
df2=throughput_table(table_name=os.path.join(data_dir,"throughput.txt"))
df2['model_type'] = df2['model_type'].replace(['longformer-large-4096'], 'longformer-large-1600')
df2['device'] = df2['device'].replace("cuda\n",'one gpu')
df2['device'] = df2['device'].replace("multiple-gpus\n",'>1 gpu')
df2=df2.loc[:,['model_type','device','time(second) per email','# of email per second']]
style_format(df2)


# In[36]:


df2


# In[ ]:




