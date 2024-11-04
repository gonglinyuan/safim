mkdir -p outputs_block results_block outputs_control results_control cache
true > to_generate.bash
true > to_eval.bash

function run_generation_evaluate_and_show_results {
  local completion_type=$1
  local mode=$2
  local post_processor=$3
  local file_suffix=$4
  local output_file="outputs_${completion_type}/Llama-3.2-${MODEL_CFG}-${file_suffix}.jsonl"
  local result_file="results_${completion_type}/Llama-3.2-${MODEL_CFG}-${file_suffix}.jsonl"

  if [ ! -f "${output_file}" ]; then
    echo "python generate.py \\
  meta-llama/Llama-3.2-${MODEL_CFG} \\
  ${completion_type} \\
  cache/Llama-3.2-${MODEL_CFG}.json \\
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
#    echo "Results of ${result_file}:"
    python show_detailed_results.py \
      ${completion_type} \
      ${result_file}
  fi
}

for MODEL_CFG in 1B; do
  run_generation_evaluate_and_show_results "block" "left_to_right" "--post_processors truncate_line_until_block" "l2r-tb"
  run_generation_evaluate_and_show_results "block" "reverse_infilling" "--post_processors truncate_line_until_block" "rfim-tb"
  run_generation_evaluate_and_show_results "block" "fewshot" "--post_processors truncate_fewshot truncate_line_until_block" "few-tb"
  run_generation_evaluate_and_show_results "block" "prefix_feeding" "--block_comments --post_processors truncate_line_until_block" "pf-bc-tb"

  run_generation_evaluate_and_show_results "control" "left_to_right" "--post_processors truncate_control" "l2r-tc"
  run_generation_evaluate_and_show_results "control" "reverse_infilling" "--post_processors truncate_control" "rfim-tc"
  run_generation_evaluate_and_show_results "control" "fewshot" "--post_processors truncate_fewshot truncate_control" "few-tc"
  run_generation_evaluate_and_show_results "control" "prefix_feeding" "--block_comments --post_processors truncate_control" "pf-bc-tc"

  run_generation_evaluate_and_show_results "api" "left_to_right" "--post_processors truncate_api_call" "l2r-ta"
  run_generation_evaluate_and_show_results "api" "reverse_infilling" "--post_processors truncate_api_call" "rfim-ta"
  run_generation_evaluate_and_show_results "api" "fewshot" "--post_processors truncate_fewshot truncate_api_call" "few-ta"
  run_generation_evaluate_and_show_results "api" "prefix_feeding" "--block_comments --post_processors truncate_api_call" "pf-bc-ta"
done
