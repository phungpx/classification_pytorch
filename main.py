import argparse
from handlers.eval_config import eval_config, load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_yaml/config.yaml")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--gpu-indices", type=str, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    stages = eval_config(config=load_yaml(args.config))

    gpu_indices = []
    if args.gpu_indices is not None:
        gpu_indices = [
            eval(gpu_index.strip()) for gpu_index in args.gpu_indices.split(",")
        ]

    # training
    if "trainer" in stages:
        trainer = stages["trainer"]
        trainer(
            num_epochs=args.num_epochs,
            resume_path=args.resume_path,
            checkpoint_path=args.checkpoint_path,
            gpu_indices=gpu_indices,
        )
    # evaluation
    elif "evaluator" in stages:
        evaluator = stages["evaluator"]
        evaluator(
            checkpoint_path=args.checkpoint_path,
            gpu_indices=gpu_indices,
        )
    else:
        print("No Run !.")
