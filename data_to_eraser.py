# content of Explainability_Calculation_NB.ipynb as a python file
import json
from tqdm.notebook import tqdm
import more_itertools as mit
import os

# get_annotated_data method is used to load the dataset
from Preprocess import *
from Preprocess.dataCollect import *

dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

# We need to load the dataset with the labels as 'hatespeech', 'offensive', and 'normal' (3-class). 

params = {}
params['num_classes']=2
params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']

data_all_labelled=get_annotated_data(params)

# The important key here is the 'bert_token'. Set it to True for Bert based models and False for Others.

params_data={
    'include_special':False,  #True is want to include <url> in place of urls if False will be removed
    'bert_tokens':True, #True /False
    'type_attention':'softmax', #softmax
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':10,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}


if(params_data['bert_tokens']):
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
else:
    print('Loading Normal tokenizer...')
    tokenizer=None

# Load the whole dataset and get the tokenwise rationales
def get_training_data(data):
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    
    final_binny_output = []
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data)):
        annotation=row['final_label']
        
        text=row['text']
        post_id=row['post_id']
        annotation_list=[row['label1'],row['label2'],row['label3']]
        tokens_all = list(row['text'])
#         attention_masks =  [list(row['explain1']),list(row['explain2']),list(row['explain1'])]
        
        if(annotation!= 'undecided'):
            tokens_all,attention_masks=returnMask(row, params_data, tokenizer)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output

training_data=get_training_data(data_all_labelled)

training_data[19]

# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]
            
# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id, 
              "end_sentence": -1, 
              "end_token": end, 
              "start_sentence": -1, 
              "start_token": start, 
              "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output

# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):  
    final_output = []
    
    if save_split:
        train_fp = open(save_path+'train.jsonl', 'w')
        val_fp = open(save_path+'val.jsonl', 'w')
        test_fp = open(save_path+'test.jsonl', 'w')
            
    for tcount, eachrow in enumerate(dataset):
        
        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]
        
        if majority_label=='normal':
            continue

        if majority_label=='non-toxic':
            continue
        
        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))
        
        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]
        
            
        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)
        
        if save_split:
            if not os.path.exists(save_path+'docs'):
                os.makedirs(save_path+'docs')
            
            with open(save_path+'docs/'+post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))
            
            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp)+'\n')
            else:
                print(post_id)
    
    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()
        
    return final_output

# The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
with open('./Data/post_id_divisions.json') as fp:
    id_division = json.load(fp)

method = 'union'
save_split = True
save_path = './Data/Evaluation/Model_Eval/'  #The dataset in Eraser Format will be stored here.
convert_to_eraser_format(training_data, method, save_split, save_path, id_division)

