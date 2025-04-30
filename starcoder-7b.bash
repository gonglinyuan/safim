mkdir -p outputs_block results_block outputs_control results_control outputs_api results_api cache
true > to_generate.bash
true > to_eval.bash

function run_generation_evaluate_and_show_results {
  local completion_type=$1
  local mode=$2
  local post_processor=$3
  local file_suffix=$4
  local output_file="outputs_${completion_type}/${model_cfg}-${file_suffix}.jsonl"
  local result_file="results_${completion_type}/${model_cfg}-${file_suffix}.jsonl"

  if [ ! -f "${output_file}" ]; then
    echo "python generate.py \\
  bigcode/${model_cfg} \\
  ${completion_type} \\
  cache/${model_cfg}.json \\
  ${output_file} \\
  ${mode} \\
  ${post_processor}" >> to_generate.bash
  fi

  if [ ! -f "${result_file}" ]; then
    echo "python evaluate.py \\
  ${completion_type} \\
  ${output_file} \\
  ${result_file}" >> to_eval.bash
  else
    python show_detailed_results.py \
      ${completion_type} \
      ${result_file} \
      --lang python
  fi
}

model_cfg=starcoderbase-7b

run_generation_evaluate_and_show_results "block" "infilling" "--post_processors truncate_line_until_block" "fim-tb"
run_generation_evaluate_and_show_results "block" "reverse_infilling" "--post_processors truncate_line_until_block" "rfim-tb"

run_generation_evaluate_and_show_results "control_fixed" "infilling" "--post_processors truncate_control_remove_colon" "fim-tcrc"
run_generation_evaluate_and_show_results "control_fixed" "reverse_infilling" "--post_processors truncate_control_remove_colon" "rfim-tcrc"

run_generation_evaluate_and_show_results "api" "infilling" "--post_processors truncate_api_call" "fim-ta"
run_generation_evaluate_and_show_results "api" "reverse_infilling" "--post_processors truncate_api_call" "rfim-ta"
