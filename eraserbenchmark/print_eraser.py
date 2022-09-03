import json

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

