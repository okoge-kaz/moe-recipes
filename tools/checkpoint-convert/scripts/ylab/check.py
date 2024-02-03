from transformers import MixtralForCausalLM
import torch
import argparse


def compare_state_dicts(original_state_dict, converted_state_dict):
    differences = {}

    for key in original_state_dict.keys():
        if key not in converted_state_dict:
            differences[key] = ("Only in original state_dict", )
            continue

        if not torch.equal(original_state_dict[key], converted_state_dict[key]):
            differences[key] = (original_state_dict[key].numpy(), converted_state_dict[key].numpy())  # テンソルの値をnumpy配列として保存

    for key in converted_state_dict.keys():
        if key not in original_state_dict:
            differences[key] = ("Only in converted state_dict", )

    return differences


parser = argparse.ArgumentParser()
parser.add_argument("--base-hf-model-path", type=str, required=True)
parser.add_argument("--converted-hf-model-path", type=str, required=True)
args = parser.parse_args()

# モデルをロード
original_model = MixtralForCausalLM.from_pretrained(
    args.base_hf_model_path,
    device_map="cpu"
)

converted_model = MixtralForCausalLM.from_pretrained(
    args.converted_hf_model_path,
    device_map="cpu"
)

# state_dictを取得
original_state_dict = original_model.state_dict()  # type: ignore
converted_state_dict = converted_model.state_dict()  # type: ignore

# state_dictの差分を比較
diffs = compare_state_dicts(
    original_state_dict=original_state_dict,
    converted_state_dict=converted_state_dict
)

for key, values in diffs.items():
    print(f"Key: {key}")
    if len(values) == 2:
        print(f"  original Value (Shape {values[0].shape}):\n{values[0]}")
        print(f"  converted 2 Value (Shape {values[1].shape}):\n{values[1]}")
    else:
        print(f"  {values[0]}")

print("Done!")
