import argparse
import copy
import random
from pathlib import Path

from sklearn.model_selection import ParameterGrid

import yaml


def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def save_yaml(yml, path):
    with open(path, "w") as outfile:
        return yaml.dump(yml, outfile)


def load_text(path):
    with open(path, "r") as infile:
        return infile.read()


def save_text(text, path):
    with open(path, "w") as outfile:
        return outfile.write(text)


def rand_id():
    return str(random.randint(1_000_000, 2_000_000))


def hyper_to_configs(hyper_conf, random_runs=None):
    """
    Create the set of configurations from the grid
    """
    configs = []
    hyper_settings = hyper_conf.pop("hyper", None)
    name_template = hyper_conf["templates"].pop("run_name", None)

    if hyper_settings is not None:
        # Convert list of hyperparams to sweep
        grid = ParameterGrid(hyper_settings)
        if random_runs:
            random_runs = None if random_runs == -1 else random_runs
            grid = sorted(grid, key=lambda k: random.random())[:random_runs]
        
        for params in grid:
            # Deep copy to avoid overwriting
            conf = copy.deepcopy(hyper_conf)
            # Fill in value of each configuation
            conf["params"].update(params)
            conf["conf_name"] = name_template.format(**params) if name_template else rand_id()
            configs.append(conf)
        
        # Make sure no repeated names
        assert(len(set(conf["conf_name"] for conf in configs)) == len(configs))
        return configs
    else:
        return [hyper_conf]


def hyper(
    hyper_settings_yml_path: str,
    base_config_yml_path: str,
    base_output_path: str,
    random_runs: int = -1,
    skip_if_file_exists: str = None,
    dry_run: bool = False,
    seed: int = None,
    ):
    """
    Generate batch file and directories for the runs
    """
    random.seed(seed)

    # 1) Generate all the configuration files and directories
    hyper_conf = load_yaml(hyper_settings_yml_path)
    
    # hyper_conf_path is a yml file defining the hyper parameter sweep
    configs = hyper_to_configs(hyper_conf, random_runs)

    # base_config_yml_path is the template for the configuration
    base_config = load_yaml(base_config_yml_path)
    job_name = hyper_conf["job_name"]
    run_template = hyper_conf["templates"]["command"]
    commands = []

    print(f"Found {len(configs)} possible configurations.")
    if dry_run:
        return
        
    for c in configs:
        # This defines the path like models/{base_output_path}/{conf_name}
        conf_name = c["conf_name"]

        # Create the output directory and the config file
        output_dir = Path(base_output_path, conf_name).absolute()
        conf_path = Path(output_dir, "config.yml")
        if skip_if_file_exists and Path(output_dir, skip_if_file_exists).exists():
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        filled_conf = {**base_config, **c["params"], 'output_dir': str(output_dir)}
        save_yaml(filled_conf, conf_path)

        run_command = run_template.format(config_path=conf_path)
        commands.append(run_command)

    # add slurm-specific items
    slurm_template = hyper_conf["templates"].pop("slurm_header", None)
    if slurm_template:
        slurm_header = slurm_template.format(n_jobs=len(commands)-1, job_name=job_name)
        commands = [slurm_header] + [
            f"test ${{SLURM_ARRAY_TASK_ID}} -eq {run_id} && {run_command}"
            for run_id, run_command in enumerate(commands)
        ]

    save_text("\n".join(commands), f"{job_name}-runs.sh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyper_settings_yml_path")
    parser.add_argument("base_config_yml_path")
    parser.add_argument("base_output_path")
    parser.add_argument(
        "--random_runs",
        type=int,
        default=None,
        help="Randomize the grid and use the first `random_runs` runs. Use all with -1"
    )
    parser.add_argument(
        "--skip_if_file_exists",
        type=str,
        default=None,
        help="If this file exists in an output model directory, do not create a run"
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=11235)
    
    args = parser.parse_args()
    hyper(
        hyper_settings_yml_path=args.hyper_settings_yml_path,
        base_config_yml_path=args.base_config_yml_path,
        base_output_path=args.base_output_path,
        random_runs=args.random_runs,
        skip_if_file_exists=args.skip_if_file_exists,
        dry_run=args.dry_run,
        seed=args.seed,
    )