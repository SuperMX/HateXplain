import json

''' Example:
------------------------
Plausibility
IOU F1 : 0.343
Token F1 : 0.212

Faithfulness
Comprehensiveness : 0.294
Sufficiency : 0.088

------------------------
              precision    recall  f1-score   support

           0      0.957     0.982     0.969      1078
           1      0.513     0.294     0.374        68

    accuracy                          0.942      1146
   macro avg      0.735     0.638     0.672      1146
weighted avg      0.930     0.942     0.934      1146

------------------------
'''

# print the required results
with open('../model_explain_output.json') as fp:
    output_data = json.load(fp)

print("\n------------------------")
	
print('Plausibility')
if 'iou_scores' in output_data:
	print('IOU F1 :', round(output_data['iou_scores'][0]['macro']['f1'], 3))
	print('Token F1 :', round(output_data['token_prf']['instance_macro']['f1'], 3))
	
if 'token_soft_metrics' in output_data:
	print('AUPRC :', round(output_data['token_soft_metrics']['auprc'], 3))

print('\nFaithfulness')
if 'classification_scores' in output_data:
	print('Comprehensiveness :', round(output_data['classification_scores']['comprehensiveness'], 3))
	print('Sufficiency :', round(output_data['classification_scores']['sufficiency'], 3))
else:
	print('--')
print("")

print("------------------------")
print("              precision    recall  f1-score   support")
print("")
print("           0      "+"-"+"     "+"-"+"     "+"-"+"      "+"-"+"")
print("           1      "+"-"+"     "+"-"+"     "+"-"+"        "+"-"+"")
print("")
print("    accuracy                          "+"-"+"      "+"-"+"")
print("   macro avg      "+"-"+"     "+"-"+"     "+"-"+"      "+"-"+"")
print("weighted avg      "+"-"+"     "+"-"+"     "+"-"+"      "+"-"+"")
print("")
print("------------------------")
