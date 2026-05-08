import argparse
import json
import os

from datasets import Dataset, DatasetDict


STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")


def maybe_login_hf() -> None:
    with open("tokens.json", "r") as f:
        token = json.load(f)["huggingface"]

    from huggingface_hub import login

    login(token=token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, default="")
    parser.add_argument("--max_score", type=float, default=0.7)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--uncertainty_delta", type=float, default=0.20)
    parser.add_argument("--num_uncertainty_intervals", type=int, default=10)
    parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
    parser.add_argument("--local_train_file", type=str, default="")
    args = parser.parse_args()
    args.min_score = max(args.min_score, 0.5 - args.uncertainty_delta)
    args.max_score = min(args.max_score, 0.5 + args.uncertainty_delta)

    datas = []
    for i in range(8):
        try:
            with open(f"{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json", "r") as f:
                data = json.load(f)
                datas.extend(data)
        except Exception:
            print(f"File {args.experiment_name}_{i}_results.json not found")
            continue

    for i in range(8):
        try:
            os.remove(f"{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json")
        except Exception:
            print(f"File {args.experiment_name}_{i}_results.json not found")
            continue

    scores = [data["score"] for data in datas]
    # print the distribution of scores
    import matplotlib.pyplot as plt

    plt.hist(scores, bins=11)
    plt.savefig("scores_distribution.png")

    filtered_datas = []
    for data in datas:
        score = float(data.get("score", 0.0))
        answer = data.get("answer", "")
        if score < args.min_score or score > args.max_score or answer == "" or answer == "None":
            continue
        uncertainty_state = min(int(args.num_uncertainty_intervals * max(0.0, min(1.0, score))), args.num_uncertainty_intervals - 1)
        filtered_datas.append(
            {
                "problem": data["question"],
                "answer": answer,
                "score": score,
                "consensus_ratio": score,
                "uncertainty_state": uncertainty_state,
                "uncertainty_boundary_error": abs(score - 0.5),
            }
        )
    print(len(filtered_datas))

    train_dataset = Dataset.from_list(filtered_datas)
    dataset_dict = {"train": train_dataset}
    dataset = DatasetDict(dataset_dict)

    local_train_file = args.local_train_file or f"{STORAGE_PATH}/datasets/{args.experiment_name}/train.parquet"
    os.makedirs(os.path.dirname(local_train_file), exist_ok=True)
    train_dataset.to_parquet(local_train_file)
    print(f"Saved local training dataset to: {local_train_file}")

    if args.repo_name != "":
        maybe_login_hf()
        config_name = f"{args.experiment_name}"
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}", private=True, config_name=config_name)
        print(f"Pushed dataset to hub: {HUGGINGFACENAME}/{args.repo_name}")


if __name__ == "__main__":
    main()





