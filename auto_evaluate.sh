# hatexplain
function evaluate() {
	path=$1
	model=$2

	# process path to folder
	foldersign="/"
	minus="-"
	foldername=${path//$foldersign/$minus}
	tsv=".tsv"
	empty=""
	foldername=${foldername//$tsv/$empty}

	mkdir $foldername
	cp $path Data/dataset.json
	postidpath=${path//combination/post_id_divisions}
	cp $postidpath Data/post_id_divisions.json
	python data_to_eraser.py
	
	mkdir Saved
	mkdir explanations_dicts
	python manual_training_inference.py bestModel_bert_base_uncased_Attn_train_TRUE.json True 100 | tee out.txt
	python testing_with_rational.py bert_supervised 100
	
	
	cd eraserbenchmark
	PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_100_explanation_top5.json --score_file ../model_explain_output.json
	python print_eraser.py
	cd ..
	
	mv out.txt $foldername
	
	mkdir $foldername/Data
	mv Data/Evaluation $foldername/Data
	mv Saved $foldername
	mv explanations_dicts $foldername
	

	#rm
}

evaluate women/minority_combination_all.tsv sexism_rules.json
#evaluate women/majority_combination_all.tsv sexism_rules.json

#evaluate homosexual/minority_val_all.tsv homophobia_rules.json
#evaluate homosexual/majority_val_all.tsv homophobia_rules.json
