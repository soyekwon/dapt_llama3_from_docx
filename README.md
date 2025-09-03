### preprocess

python dapt_llama3_from_docx.py preprocess   --input_dir ./dummy   --output_jsonl ./vehicle_specs.jsonl   --min_chars 100

### summarize

python dapt_llama3_from_docx.py summarize   --input_dir ./dummy   --output_jsonl ./vehicle_specs_summ.jsonl   --target_ratio 0.25 --max_sentences 40 --min_chars 100

### train

python dapt_llama3_from_docx.py train   —model_id meta-llama/Llama-3.1-8B-Instruct   —train_jsonl ./vehicle_specs_summ.jsonl   —output_dir ./llama3_vehicle_dapt_lora   —use_lora —use_qlora —packing  —bf16 —gradient_checkpointing   —eval_ratio 0.1
