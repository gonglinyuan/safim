from data_utils import load_dataset, stream_jsonl


def pretty_display(code):
    print(code.replace(" ", "▁").replace("\n", "↵\n").replace("\r", "⤓"))


def compare_results(data_type_or_path, output_path_1, output_path_2, result_path_1, result_path_2):
    # why is 1 worse than 2?
    for m_data, m_o1, m_o2, m_r1, m_r2 in zip(
        load_dataset(data_type_or_path),
        stream_jsonl(output_path_1),
        stream_jsonl(output_path_2),
        stream_jsonl(result_path_1),
        stream_jsonl(result_path_2)
    ):
        if not m_r1["passed"] and m_r2["passed"]:
            print("❌")
            pretty_display(m_data['eval_prompt'].replace("{{completion}}", m_o1["completion"]))
            print("✅")
            pretty_display(m_data['eval_prompt'].replace("{{completion}}", m_o2["completion"]))
