# hatexplain
function evaluate() {
	path=$1
	model=$2
	voting=$3

	# process path to folder
	foldersign="/"
	minus="-"
	foldername=${path//$foldersign/$minus}
	tsv=".json"
	empty=""
	foldername=${foldername//$tsv/$empty}
	foldername="$foldername-$model"

    if [[ $path == *"women"* ]]; then
		target="Women"
    elif [[ $path == *"homosexual"* ]]; then
		target="Homosexual"
	else
		echo "ERROR: Could not guess target."
	fi

	mkdir $foldername
	cp $path Data/dataset.json
	postidpath=${path//combination/post_id_divisions}
	cp $postidpath Data/post_id_divisions.json
	
	#remove cached data
	rm -Rf Data/Total*
	
	mkdir -p Saved
	mkdir -p explanations_dicts
	mkdir -p Data/Evaluation
	mkdir -p Data/Evaluation/Model_Eval
	
	python data_to_eraser.py $voting $target
	
	if [[ "$model" == "bert" ]]; then
		#python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json True
		echo "ERROR: Not supported yet."
	elif [[ "$model" == "bert_supervised" ]]; then
		python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 100 $voting $target | tee out.txt
		python testing_with_rational.py bert_supervised 100 $voting $target | tee out2.txt
		predictions_file="../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_100_explanation_top5.json"
	elif [[ "$model" == "cnngru" ]]; then
		python manual_training_inference.py best_model_json/bestModel_cnn_gru.json True 100 $voting $target | tee out.txt #dummy attention lambda
		python testing_with_lime.py cnngru 100 100 $voting $target | tee out2.txt #dummy attention lambda
		predictions_file="../explanations_dicts/bestModel_cnngru_100_explanation_top5.json"
	elif [[ "$model" == "birnn_att" ]]; then
		#python manual_training_inference.py best_model_json/bestModel_birnnatt.json
		echo "ERROR: Not supported yet."
	elif [[ "$model" == "birnn_scrat" ]]; then
		python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 100 $voting $target | tee out.txt
		python testing_with_rational.py birnn_scrat 100 $voting $target | tee out2.txt
		predictions_file="../explanations_dicts/bestModel_birnnscrat_100_explanation_top5.json"
	else
		echo "ERROR: Unknown model string."
	fi

	mv out.txt $foldername
	mv out2.txt $foldername
	
	cd eraserbenchmark
	PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval --results $predictions_file --score_file ../model_explain_output.json | tee eraser.txt
	
	python print_eraser.py | tee final.txt
	cd ..
	
	mv eraserbenchmark/eraser.txt $foldername
	mv eraserbenchmark/final.txt $foldername
	
	mkdir -p $foldername/Data
	mv Data/Evaluation $foldername/Data
	mv Saved $foldername
	mv explanations_dicts $foldername
	mv model_explain_output.json $foldername

}

if true; then
	rm -Rf Saved/*
	rm -Rf explanations_dicts/*
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

#evaluate women/minority_combination_all.json cnngru minority
#evaluate women/majority_combination_all.json cnngru majority
#evaluate homosexual/minority_combination_all.json cnngru minority
#evaluate homosexual/majority_combination_all.json cnngru majority
