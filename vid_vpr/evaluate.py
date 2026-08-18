import argparse
import json
import logging

from vid_vpr.config import load_config, project_path, require_sections
from vid_vpr.evaluation.retrieval import evaluate_model
from vid_vpr.models.factory import load_student, load_teacher
from vid_vpr.training.runtime import close_runtime, setup_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VID-VPR checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("student", "teacher"), default="student")
    parser.add_argument("--checkpoint")
    parser.add_argument("overrides", nargs="*", help="Configuration overrides")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    require_sections(config, "experiment", "model", "data", "evaluation")
    config["experiment"]["name"] = f"evaluate_{args.model}"
    runtime = setup_runtime(config)
    try:
        checkpoint = args.checkpoint or config["model"][f"{args.model}_path"]
        if args.model == "teacher":
            model = load_teacher(config["model"], project_path(checkpoint))
            use_vlm = True
        else:
            model = load_student(config["model"], project_path(checkpoint))
            use_vlm = False
        model.to(runtime.device)
        results, _ = evaluate_model(
            model, config, runtime.device, use_vlm=use_vlm, runtime=runtime
        )
        if runtime.is_main:
            output = runtime.run_dir / "metrics.json"
            output.write_text(json.dumps(results, indent=2) + "\n")
            logging.info("Saved metrics to %s", output)
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()
