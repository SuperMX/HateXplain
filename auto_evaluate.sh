# hatexplain
function evaluate() {
	path=$1
	model=$2
	voting=$3

	# process path to folder
	foldersign="/"
	minus="-"
	foldername=${path//$foldersign/$minus}
	tsv=".tsv"
	empty=""
	foldername=${foldername//$tsv/$empty}
	foldername="$foldername-$model"

	mkdir $foldername
	cp $path Data/dataset.json
	postidpath=${path//combination/post_id_divisions}
	cp $postidpath Data/post_id_divisions.json
	python data_to_eraser.py
	
	mkdir -p Saved
	mkdir -p explanations_dicts
	
	if [[ "$model" == "bert" ]]; then
		#python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json True
		echo "ERROR: Not supported yet."
	elif [[ "$model" == "bert_supervised" ]]; then
		python manual_training_inference.py bestModel_bert_base_uncased_Attn_train_TRUE.json True 100 | tee out.txt
		python testing_with_rational.py bert_supervised 100 | tee out2.txt
		predictions_file="../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_100_explanation_top5.json"
	elif [[ "$model" == "cnngru" ]]; then
		python manual_training_inference.py best_model_json/bestModel_cnn_gru.json True 100 | tee out.txt #dummy attention lambda
		python testing_with_lime.py cnngru 100 100 | tee out2.txt #dummy attention lambda
		predictions_file="../explanations_dicts/bestModel_cnngru_100_explanation_top5.json"
	elif [[ "$model" == "birnn_att" ]]; then
		#python manual_training_inference.py best_model_json/bestModel_birnnatt.json
		echo "ERROR: Not supported yet."
	elif [[ "$model" == "birnn_scrat" ]]; then
		python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 100 | tee out.txt
		python testing_with_rational.py birnn_scrat 100 | tee out2.txt
		predictions_file="../explanations_dicts/bestModel_birnnscrat_100_explanation_top5.json"
	else
		echo "ERROR: Unknown model string."
	fi

	mv out.txt $foldername
	mv out2.txt $foldername
	
	cd eraserbenchmark
	PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval --results $predictions_file --score_file ../model_explain_output.json
	
	python print_eraser.py
	cd ..
	
	mv out.txt $foldername
	
	mkdir -p $foldername/Data
	mv Data/Evaluation $foldername/Data
	mv Saved $foldername
	mv explanations_dicts $foldername
	mv model_explain_output.json $foldername

}

if false; then
	rm -R Saved
	rm -R explanations_dicts
	rm -f model_explain_output.json
fi

# # models
#'bert': 				BERT
#'bert_supervised':		BERT-HateXplain		with [Attn]
#'birnn':				BiRNN
#'cnngru':				CNN-GRU				with [LIME]
#'birnn_att':			BiRNN-Attn
#'birnn_scrat':			BiRNNN-HateXplain	with [Attn]

evaluate women/minority_combination_all.json bert_supervised minority
evaluate women/majority_combination_all.json bert_supervised majority
evaluate homosexual/minority_combination_all.json bert_supervised minority
evaluate homosexual/majority_combination_all.json bert_supervised majority

evaluate women/minority_combination_all.json birnn_scrat minority
evaluate women/majority_combination_all.json birnn_scrat majority
evaluate homosexual/minority_combination_all.json birnn_scrat minority
evaluate homosexual/majority_combination_all.json birnn_scrat majority

evaluate women/minority_combination_all.json cnngru minority
evaluate women/majority_combination_all.json cnngru majority
evaluate homosexual/minority_combination_all.json cnngru minority
evaluate homosexual/majority_combination_all.json cnngru majority
