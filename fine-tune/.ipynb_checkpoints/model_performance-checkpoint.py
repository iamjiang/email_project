#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# In[9]:


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),output_dir)

with open(os.path.join(output_dir,"y_true_pred.txt"),"r") as file:
    true_y=[]
    pred_y=[]
    prob_y=[]
    for line in file:
        x,y,z=line.strip().split(',')
        true_y.append(int(x))
        pred_y.append(int(y))
        prob_y.append(float(z))


# In[12]:


print()
print(f"\n===========Test Set Performance===============\n")
print()

print(classification_report(true_y, pred_y))
print()
print()
print(confusion_matrix(true_y, pred_y))  


# In[45]:


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


# In[46]:


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),output_dir)
metric_bert=metric_table(table_name=os.path.join(output_dir,"metrics_test.txt"))


# In[56]:


model_name="longformer-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),output_dir)
metric_longformer=metric_table(table_name=os.path.join(output_dir,"metrics_test.txt"))


# In[47]:


output_dir=os.path.join('/opt/omniai/work/instance1/jupyter/email-complaints/lightgbm+transformers', "tfidf+structure")
metric_lightgbm=metric_table(table_name=os.path.join(output_dir,"lightgbm_metrics_test.txt"))
metric_xgboost=metric_table(table_name=os.path.join(output_dir,"xgboost_metrics_test.txt"))
metric_catboost=metric_table(table_name=os.path.join(output_dir,"catboost_metrics_test.txt"))
metric_randomforest=metric_table(table_name=os.path.join(output_dir,"randomforest_metrics_test.txt"))


# In[58]:


# metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_catboost,metric_lightgbm,metric_bert], axis=0)
metric_test = pd.concat([metric_randomforest, metric_xgboost, metric_catboost,metric_lightgbm,metric_bert,metric_longformer], axis=0)
metric_test['model_type'] = metric_test['model_type'].replace(['roberta-large'], 'BERT')
metric_test['model_type'] = metric_test['model_type'].replace(['longformer-large-4096'], 'Longformer')
style_format(metric_test,  type="test set")


# In[ ]:

######## draw confusion matrix #########

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

output_dir=os.path.join('/opt/omniai/work/instance1/jupyter/email-complaints/lightgbm+transformers', "tfidf+structure")

file_name=os.path.join(output_dir,"xgboost_y_true_pred.txt")
xgboost_true_y, xgboost_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"lightgbm_y_true_pred.txt")
lightgbm_true_y, lightgbm_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"catboost_y_true_pred.txt")
catboost_true_y, catboost_pred_y, _ =read_output(file_name)

file_name=os.path.join(output_dir,"randomforest_y_true_pred.txt")
randomforest_true_y, randomforest_pred_y, _ =read_output(file_name)


model_name="roberta-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),output_dir)
file_name=os.path.join(output_dir,"y_true_pred.txt")
roberta_true_y, roberta_pred_y, _ =read_output(file_name)


model_name="longformer-large"
output_dir=model_name.split("-")[0] + "_" + model_name.split("-")[1] + "_repo"
output_dir=os.path.join(os.getcwd(),output_dir)
file_name=os.path.join(output_dir,"y_true_pred.txt")
longformer_true_y, longformer_pred_y, _ =read_output(file_name)

confusion_matrix = metrics.confusion_matrix(xgboost_true_y, xgboost_pred_y)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("xgboost Model")
plt.show()

confusion_matrix = metrics.confusion_matrix(lightgbm_true_y, lightgbm_pred_y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot(values_format=',')
plt.title("lightgbm Model")
plt.show()
