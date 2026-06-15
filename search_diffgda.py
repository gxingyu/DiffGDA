import argparse
import csv
import itertools
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


LR_SPACE = [0.0001, 0.001, 0.01]
ALPHA_SPACE = [0.1, 0.2, 0.3]
ETA_SPACE = [0.1, 0.2, 0.3, 0.4, 0.5]
T_SPACE = [20, 40, 60, 80, 100]

ALL_SCENARIOS = [
    "ACMv9:Citationv1",
    "ACMv9:DBLPv7",
    "Citationv1:ACMv9",
    "Citationv1:DBLPv7",
    "DBLPv7:ACMv9",
    "DBLPv7:Citationv1",
    "USA:BRAZIL",
    "USA:EUROPE",
    "BRAZIL:USA",
    "BRAZIL:EUROPE",
    "EUROPE:USA",
    "EUROPE:BRAZIL",
    "Blog1:Blog2",
    "Blog2:Blog1",
]

AIRPORT = {"USA", "BRAZIL", "EUROPE"}
CITATION = {"ACMv9", "Citationv1", "DBLPv7"}
BLOG = {"Blog1", "Blog2"}
MICRO_RE = re.compile(r"micro-f1:\s*([0-9]*\.?[0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", default="USA")
    parser.add_argument("--target", default="BRAZIL")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="logs_search")
    parser.add_argument("--scenarios", nargs="*", default=["USA:BRAZIL"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--strategy", choices=["grid", "random"], default="grid")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=521070)
    parser.add_argument("--keep-top-k", type=int, default=10)
    parser.add_argument("--save-all-results", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume-root", default="logs_search")
    parser.add_argument("--resume-from", nargs="*", default=[])
    parser.add_argument("--resume-key", default="default")
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alignment", choices=["mmd", "adv"], default="mmd")
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--s-pnums", type=int, default=0)
    parser.add_argument("--t-pnums", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--allow-alpha-zero", action="store_true")
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--diffusion-epochs", type=int, default=None)
    parser.add_argument("--round", type=int, default=0)
    return parser.parse_args()


def scenario_config(source):
    return source


def validate_params(alpha, eta, diffusion_steps, allow_alpha_zero=False):
    if allow_alpha_zero:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1] when --allow-alpha-zero is set.")
    elif not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]. alpha=0 creates an empty subgraph.")
    if not (0.0 <= eta <= 0.5):
        raise ValueError("eta must be in [0, 0.5].")
    if not (1 <= diffusion_steps <= 150):
        raise ValueError("diffusion_steps must be in [1, 150]. T=0 is not valid for this implementation.")


def param_grid(args):
    combos = list(itertools.product(LR_SPACE, ALPHA_SPACE, ETA_SPACE, T_SPACE))
    rng = random.Random(args.seed)
    if args.strategy == "random":
        rng.shuffle(combos)
    if args.max_trials and args.max_trials > 0:
        combos = combos[: args.max_trials]
    for lr, alpha, eta, diffusion_steps in combos:
        validate_params(alpha, eta, diffusion_steps, args.allow_alpha_zero)
        yield {
            "lr": lr,
            "alpha": alpha,
            "eta": eta,
            "diffusion_steps": diffusion_steps,
        }


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_topk(out_dir, scenario, rows, keep_top_k):
    rows = sorted(rows, key=lambda item: item.get("best_micro_f1", -1.0), reverse=True)[:keep_top_k]
    stem = scenario.replace(":", "_to_")
    csv_path = out_dir / f"top{keep_top_k}_{stem}.csv"
    json_path = out_dir / f"top{keep_top_k}_{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def trial_key(scenario, round_idx, params, alignment, resume_key):
    return (
        scenario,
        int(round_idx),
        resume_key,
        alignment,
        f"{float(params['lr']):.10g}",
        f"{float(params['alpha']):.10g}",
        f"{float(params['eta']):.10g}",
        int(params["diffusion_steps"]),
    )


def parse_command_line(line):
    try:
        parts = shlex.split(line[len("COMMAND: ") :])
    except ValueError:
        return {}

    parsed = {}
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.startswith("--"):
            name = part[2:].replace("-", "_")
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                parsed[name] = parts[i + 1]
                i += 2
            else:
                parsed[name] = True
                i += 1
        else:
            i += 1
    return parsed


def row_from_log(log_path):
    command_args = {}
    trial_json = None
    final_json = None
    return_code = None
    best_micro = None

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("COMMAND: "):
                    command_args = parse_command_line(line)
                elif line.startswith("TRIAL "):
                    try:
                        trial_json = json.loads(line[len("TRIAL ") :])
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("RESULT_JSON "):
                    try:
                        final_json = json.loads(line[len("RESULT_JSON ") :])
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("RETURN_CODE:"):
                    try:
                        return_code = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

                match = MICRO_RE.search(line)
                if match:
                    value = float(match.group(1))
                    best_micro = value if best_micro is None else max(best_micro, value)
    except OSError:
        return None

    data = {}
    if command_args:
        data.update(command_args)
    if trial_json:
        data.update(trial_json)

    if "alignment" not in data:
        data["alignment"] = "adv"
    if "resume_key" not in data:
        data["resume_key"] = "legacy"
    required = ["source", "target", "round", "resume_key", "alignment", "lr", "alpha", "eta", "diffusion_steps"]
    if not all(name in data for name in required):
        return None
    if return_code != 0 or final_json is None:
        return None

    scenario = f"{data['source']}:{data['target']}"
    final_micro = final_json.get("micro_f1")
    final_macro = final_json.get("macro_f1")
    if final_micro is not None:
        best_micro = max(best_micro if best_micro is not None else -1.0, float(final_micro))
    if final_json.get("best_micro_f1") is not None:
        best_micro = max(best_micro if best_micro is not None else -1.0, float(final_json["best_micro_f1"]))

    return {
        "scenario": scenario,
        "source": data["source"],
        "target": data["target"],
        "round": int(data["round"]),
        "resume_key": data["resume_key"],
        "trial": -1,
        "return_code": return_code,
        "elapsed_sec": None,
        "best_micro_f1": best_micro if best_micro is not None else -1.0,
        "final_micro_f1": final_micro,
        "final_macro_f1": final_macro,
        "lr": float(data["lr"]),
        "alpha": float(data["alpha"]),
        "eta": float(data["eta"]),
        "diffusion_steps": int(data["diffusion_steps"]),
        "alignment": data["alignment"],
        "diffusion_epochs": int(data["diffusion_epochs"]) if data.get("diffusion_epochs") not in (None, True) else None,
        "dropout": float(data.get("dropout", 0.2)),
        "epoch": int(data.get("epoch", 150)),
        "nhid": int(data.get("nhid", 64)),
        "num_layers": int(data.get("num_layers", 3)),
        "s_pnums": int(data.get("s_pnums", 0)),
        "t_pnums": int(data.get("t_pnums", 10)),
        "log_path": str(log_path),
    }


def scan_completed_trials(paths):
    completed = {}
    for raw_path in paths:
        root = Path(raw_path)
        if not root.exists():
            continue
        for log_path in root.glob("**/trial_logs/**/*.log"):
            row = row_from_log(log_path)
            if not row:
                continue
            key = trial_key(
                row["scenario"],
                row["round"],
                {
                    "lr": row["lr"],
                    "alpha": row["alpha"],
                    "eta": row["eta"],
                    "diffusion_steps": row["diffusion_steps"],
                },
                row["alignment"],
                row["resume_key"],
            )
            old = completed.get(key)
            if old is None or row["best_micro_f1"] >= old["best_micro_f1"]:
                completed[key] = row
    return completed


def run_one_subprocess(args, source, target, params, round_idx, trial_idx, out_dir):
    scenario = f"{source}:{target}"
    log_dir = out_dir / "trial_logs" / scenario.replace(":", "_to_")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"trial{trial_idx:04d}_r{round_idx}_lr{params['lr']}_a{params['alpha']}"
        f"_eta{params['eta']}_T{params['diffusion_steps']}_{args.alignment}_{args.resume_key}"
    ).replace(".", "p")
    log_path = log_dir / f"{tag}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trial",
        "--source",
        source,
        "--target",
        target,
        "--config",
        scenario_config(source),
        "--device",
        args.device,
        "--nhid",
        str(args.nhid),
        "--dropout",
        str(args.dropout),
        "--alignment",
        args.alignment,
        "--resume-key",
        args.resume_key,
        "--epoch",
        str(args.epoch),
        "--num-layers",
        str(args.num_layers),
        "--weight-decay",
        str(args.weight_decay),
        "--s-pnums",
        str(args.s_pnums),
        "--t-pnums",
        str(args.t_pnums),
        "--lr",
        str(params["lr"]),
        "--alpha",
        str(params["alpha"]),
        "--eta",
        str(params["eta"]),
        "--diffusion-steps",
        str(params["diffusion_steps"]),
        "--seed",
        str(args.seed + round_idx),
        "--round",
        str(round_idx),
    ]
    if args.diffusion_epochs is not None:
        cmd.extend(["--diffusion-epochs", str(args.diffusion_epochs)])

    start = time.time()
    best_micro = None
    final_json = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        log_f.write("COMMAND: " + " ".join(cmd) + "\n")
        log_f.write("START: " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        for line in proc.stdout:
            log_f.write(line)
            match = MICRO_RE.search(line)
            if match:
                value = float(match.group(1))
                best_micro = value if best_micro is None else max(best_micro, value)
            if line.startswith("RESULT_JSON "):
                final_json = json.loads(line[len("RESULT_JSON ") :])
        return_code = proc.wait()
        log_f.write("\nEND: " + datetime.now().isoformat(timespec="seconds") + "\n")
        log_f.write(f"RETURN_CODE: {return_code}\n")

    elapsed = time.time() - start
    row = {
        "scenario": scenario,
        "source": source,
        "target": target,
        "round": round_idx,
        "resume_key": args.resume_key,
        "trial": trial_idx,
        "return_code": return_code,
        "elapsed_sec": round(elapsed, 3),
        "best_micro_f1": best_micro if best_micro is not None else -1.0,
        "final_micro_f1": None,
        "final_macro_f1": None,
        "lr": params["lr"],
        "alpha": params["alpha"],
        "eta": params["eta"],
        "diffusion_steps": params["diffusion_steps"],
        "alignment": args.alignment,
        "diffusion_epochs": args.diffusion_epochs,
        "dropout": args.dropout,
        "epoch": args.epoch,
        "nhid": args.nhid,
        "num_layers": args.num_layers,
        "s_pnums": args.s_pnums,
        "t_pnums": args.t_pnums,
        "log_path": str(log_path),
    }
    if final_json:
        row["final_micro_f1"] = final_json.get("micro_f1")
        row["final_macro_f1"] = final_json.get("macro_f1")
        row["best_micro_f1"] = max(
            row["best_micro_f1"],
            final_json.get("best_micro_f1", final_json.get("micro_f1", -1.0)),
        )
    return row


def controller(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ALL_SCENARIOS if args.all_scenarios else args.scenarios
    trials_csv = out_dir / "trials.csv"
    trials_jsonl = out_dir / "trials.jsonl"
    controller_log = out_dir / "controller.log"
    top_by_scenario = {}
    resume_paths = []
    if not args.no_resume:
        resume_paths = [out_dir, Path(args.resume_root), *[Path(path) for path in args.resume_from]]
    completed_trials = scan_completed_trials(resume_paths) if resume_paths else {}

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_space": {
            "lr": LR_SPACE,
            "alpha": ALPHA_SPACE,
            "eta": ETA_SPACE,
            "diffusion_steps": T_SPACE,
            "fixed": {
                "nhid": args.nhid,
                "dropout": args.dropout,
                "alignment": args.alignment,
                "diffusion_epochs": args.diffusion_epochs,
                "epoch": args.epoch,
                "num_layers": args.num_layers,
                "s_pnums": args.s_pnums,
                "t_pnums": args.t_pnums,
            },
        },
        "scenarios": scenarios,
        "strategy": args.strategy,
        "max_trials": args.max_trials,
        "rounds": args.rounds,
        "save_all_results": args.save_all_results,
        "schedule": "first round full grid, later rounds selected top-k only",
        "rerun_top_k": args.keep_top_k,
        "resume_enabled": not args.no_resume,
        "resume_key": args.resume_key,
        "resume_paths": [str(path) for path in resume_paths],
        "completed_trials_loaded": len(completed_trials),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    with controller_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", **run_meta}, ensure_ascii=False) + "\n")
    if args.rounds <= 0:
        return

    trial_idx = 0

    def execute_trial(source, target, scenario, params, round_idx):
        nonlocal trial_idx
        key = trial_key(scenario, round_idx, params, args.alignment, args.resume_key)
        if key in completed_trials:
            row = completed_trials[key]
            top_by_scenario[scenario].append(row)
            top_by_scenario[scenario] = write_topk(
                out_dir, scenario, top_by_scenario[scenario], args.keep_top_k
            )
            print(
                f"[skip] {scenario} round={round_idx} params={params} "
                f"best={row['best_micro_f1']} log={row['log_path']}",
                flush=True,
            )
            with controller_log.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "trial_skip_completed",
                            "scenario": scenario,
                            "round": round_idx,
                            "params": params,
                            "best_micro_f1": row["best_micro_f1"],
                            "log_path": row["log_path"],
                            "time": datetime.now().isoformat(timespec="seconds"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return row

        trial_idx += 1
        print(f"[trial {trial_idx}] {scenario} round={round_idx} params={params}", flush=True)
        with controller_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "trial_start",
                        "trial": trial_idx,
                        "scenario": scenario,
                        "round": round_idx,
                        "params": params,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        row = run_one_subprocess(args, source, target, params, round_idx, trial_idx, out_dir)
        if args.save_all_results:
            append_csv(trials_csv, row)
            with trials_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        top_by_scenario[scenario].append(row)
        top_by_scenario[scenario] = write_topk(
            out_dir, scenario, top_by_scenario[scenario], args.keep_top_k
        )
        with controller_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "trial_done",
                        "trial": trial_idx,
                        "scenario": scenario,
                        "round": round_idx,
                        "return_code": row["return_code"],
                        "best_micro_f1": row["best_micro_f1"],
                        "elapsed_sec": row["elapsed_sec"],
                        "log_path": row["log_path"],
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(
            f"[done] rc={row['return_code']} best={row['best_micro_f1']} "
            f"elapsed={row['elapsed_sec']}s log={row['log_path']}",
            flush=True,
        )
        if row["return_code"] == 0 and row["best_micro_f1"] >= 0:
            completed_trials[key] = row
        return row

    for scenario in scenarios:
        source, target = scenario.split(":")
        top_by_scenario[scenario] = []

        first_round_rows = []
        for params in param_grid(args):
            first_round_rows.append(execute_trial(source, target, scenario, params, 0))

        valid_first_round = [
            row for row in first_round_rows
            if row["return_code"] == 0 and row["best_micro_f1"] >= 0
        ]
        selected = sorted(
            valid_first_round, key=lambda item: item["best_micro_f1"], reverse=True
        )[: args.keep_top_k]
        selected_params = [
            {
                "rank": rank,
                "scenario": row["scenario"],
                "resume_key": row.get("resume_key", args.resume_key),
                "best_micro_f1": row["best_micro_f1"],
                "final_micro_f1": row["final_micro_f1"],
                "final_macro_f1": row["final_macro_f1"],
                "lr": row["lr"],
                "alpha": row["alpha"],
                "eta": row["eta"],
                "diffusion_steps": row["diffusion_steps"],
                "alignment": row["alignment"],
                "diffusion_epochs": row.get("diffusion_epochs"),
                "round0_log_path": row["log_path"],
            }
            for rank, row in enumerate(selected, start=1)
        ]
        stem = scenario.replace(":", "_to_")
        selected_json = out_dir / f"selected_top{args.keep_top_k}_{stem}.json"
        selected_csv = out_dir / f"selected_top{args.keep_top_k}_{stem}.csv"
        selected_json.write_text(json.dumps(selected_params, indent=2, ensure_ascii=False), encoding="utf-8")
        with selected_csv.open("w", newline="", encoding="utf-8") as f:
            if selected_params:
                writer = csv.DictWriter(f, fieldnames=list(selected_params[0].keys()))
                writer.writeheader()
                writer.writerows(selected_params)

        with controller_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "selected_top_params",
                        "scenario": scenario,
                        "selected": selected_params,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        selected_param_dicts = [
            {
                "lr": row["lr"],
                "alpha": row["alpha"],
                "eta": row["eta"],
                "diffusion_steps": row["diffusion_steps"],
            }
            for row in selected
        ]
        for round_idx in range(1, args.rounds):
            for params in selected_param_dicts:
                execute_trial(source, target, scenario, params, round_idx)


def load_dataset(name, device):
    import os.path as osp

    import torch
    from pygda.datasets import AirportDataset, BlogDataset, CitationDataset
    from torch_geometric.utils import degree

    root = Path(__file__).resolve().parent
    if name in CITATION:
        dataset = CitationDataset(osp.join(root, "data", "Citation", name), name)
    elif name in AIRPORT:
        dataset = AirportDataset(osp.join(root, "data", "Airport", name), name)
    elif name in BLOG:
        dataset = BlogDataset(osp.join(root, "data", "Blog", name), name)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0].to(device)
    if not hasattr(data, "x") or data.x is None:
        default_num_features = 241
        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        data.x = torch.nn.functional.one_hot(
            node_degrees, num_classes=default_num_features
        ).float().to(device)
    return data


def trial(args):
    validate_params(args.alpha, args.eta, args.diffusion_steps, args.allow_alpha_zero)

    import numpy as np
    import torch
    import yaml
    from easydict import EasyDict as edict
    from pygda.metrics import eval_macro_f1, eval_micro_f1

    from model.dugda import DUGDA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_name = args.config or scenario_config(args.source)
    with open(Path(__file__).resolve().parent / "config" / f"{config_name}.yaml", "r", encoding="utf-8") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    config.sde.x.num_scales = args.diffusion_steps
    config.sde.adj.num_scales = args.diffusion_steps
    config.train.lr = args.lr
    if args.diffusion_epochs is not None:
        config.train.num_epochs = args.diffusion_epochs

    source_data = load_dataset(args.source, args.device)
    target_data = load_dataset(args.target, args.device)
    num_features = source_data.x.size(1)
    num_classes = len(np.unique(source_data.y.detach().cpu().numpy()))

    print(
        "TRIAL "
        + json.dumps(
            {
                "source": args.source,
                "target": args.target,
                "config": config_name,
                "round": args.round,
                "resume_key": args.resume_key,
                "seed": args.seed,
                "lr": args.lr,
                "alpha": args.alpha,
                "eta": args.eta,
                "diffusion_steps": args.diffusion_steps,
                "alignment": args.alignment,
                "diffusion_epochs": args.diffusion_epochs,
                "dropout": args.dropout,
                "epoch": args.epoch,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model = DUGDA(
        in_dim=num_features,
        hid_dim=args.nhid,
        num_classes=num_classes,
        device=args.device,
        config=config,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epoch=args.epoch,
        weight=args.eta,
        alignment=args.alignment,
        s_pnums=args.s_pnums,
        t_pnums=args.t_pnums,
        alpha=args.alpha,
    )
    model.seed = args.seed
    model.fit(source_data, target_data)

    logits, labels = model.predict(target_data)
    preds = logits.argmax(dim=1)
    micro_f1 = eval_micro_f1(labels, preds)
    macro_f1 = eval_macro_f1(labels, preds)
    print("micro-f1: " + str(micro_f1), flush=True)
    print("macro-f1: " + str(macro_f1), flush=True)
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "micro_f1": float(micro_f1),
                "macro_f1": float(macro_f1),
                "best_micro_f1": float(model.best_metrics["micro_f1"]),
                "best_macro_f1": float(model.best_metrics["macro_f1"]),
                "best_diff_epoch": int(model.best_metrics["diff_epoch"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    args = parse_args()
    if args.trial:
        trial(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
