# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

import neuralset as ns
from neuralset.events.study import EventsTransform
from neuraltrain.losses import BaseLoss
from neuraltrain.models import BaseModelConfig as ModelConfig
from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.utils import WandbLoggerConfig

from . import transforms as _transforms  # noqa: F401  (registers EventsTransforms)
from .callbacks import LogSentencePredictions
from .metrics import CER
from .pl_module import BrainModule
from .utils import (
    ChannelPositions2D,
    EegChannelPositions2D,
    SentenceGroupedDistributedSampler,
    materialize_lazy_params,
)


def _windows_safe_timeline_cache_key(timeline: dict) -> str:
    """Stable Exca item key that contains no Windows-reserved filename chars."""
    payload = json.dumps(timeline, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    readable = "-".join(
        str(timeline.get(key, "unknown")) for key in ("subject", "session", "task")
    )
    return f"{readable}-{digest}"


class Data(pydantic.BaseModel):
    """Builds train/val/test dataloaders of keystroke-aligned neural windows.

    Runs the study, applies the preprocessing/split transforms, prepares the
    feature extractors, and creates one dataloader per split whose sampler keeps
    each sentence's keystrokes on a single rank for the sentence transformer.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    study: ns.events.Study
    transforms: list[EventsTransform] = pydantic.Field(default_factory=list)
    neuro: ns.extractors.BaseExtractor
    feature: ns.extractors.BaseExtractor

    num_classes: int = 29
    start: float = -0.2
    duration: float = 0.5
    batch_size: int = 128
    val_batch_size: int = 2048
    test_batch_size: int = 2048
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    def build_events(self):
        # neuralset 0.2.2 uses raw JSON as an Exca filename for timeline caches.
        # JSON's quotes and colons are invalid in Windows filenames; replacing the
        # key function affects only cache addressing, never event values.
        if os.name == "nt" and self.study.infra_timelines.folder is not None:
            self.study.infra_timelines.item_uid = _windows_safe_timeline_cache_key
        events = self.study.run()
        for transform in self.transforms:
            events = transform.run(events)
        return ns.events.standardize_events(events)

    def build(self) -> dict[str, DataLoader]:
        events = self.build_events()
        self.neuro.prepare(events)
        self.feature.prepare(events)

        # Match the modality selected by the neuro extractor ("Meg" or "Eeg").
        # The previous MEG literal made the otherwise modality-agnostic pipeline
        # fail while preparing subject ids for EEG events.
        subject_id = ns.extractors.LabelEncoder(
            event_types=self.neuro.event_types, event_field="subject"
        )
        subject_id.prepare(events)
        position_cls = (
            EegChannelPositions2D
            if self.study.__class__.__name__ == "Pinet2024Eeg"
            else ChannelPositions2D
        )
        channel_positions = position_cls(neuro=self.neuro)
        channel_positions.prepare(events)

        extractors = {
            "neuro": self.neuro,
            "feature": self.feature,
            "subject_id": subject_id,
            "channel_positions": channel_positions,
        }
        batch_sizes = {
            "train": self.batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size,
        }
        loaders: dict[str, DataLoader] = {}
        for split, batch_size in batch_sizes.items():
            mask = (events.split == split) & (events.type == "Keystroke")
            segments = ns.segments.list_segments(
                events, mask, start=self.start, duration=self.duration
            )
            if not segments:
                continue
            dataset = ns.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=True,
            )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                sampler=SentenceGroupedDistributedSampler(dataset.segments),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Train and evaluate the Brain2Qwerty V1 keystroke decoder."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    data: Data
    brain_model_config: ModelConfig
    transformer_config: ModelConfig
    loss: BaseLoss
    optimizer: LightningOptimizer

    seed: int = 33
    n_epochs: int = 100
    patience: int = 80
    grad_max_norm: float | None = None
    save_checkpoints: bool = True
    devices: int = 8
    output_dir: str = "."
    log_every_n_steps: int = 5
    accumulate_grad_batches: int = 1
    limit_train_batches: int | float = 1.0
    limit_val_batches: int | float = 1.0
    limit_test_batches: int | float = 1.0

    eval_only: bool = False
    ckpt_path: str | None = None
    wandb_config: WandbLoggerConfig | None = None

    _module: BrainModule | None = None
    _trainer: pl.Trainer | None = None

    def _accelerator(self) -> tuple[str, int]:
        if torch.cuda.is_available():
            return "gpu", max(1, min(self.devices, torch.cuda.device_count()))
        return "cpu", 1

    def _build_modules(
        self, loader: DataLoader
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        n_in = next(iter(loader)).data["neuro"].shape[1]
        hidden = self.brain_model_config.hidden
        brain = self.brain_model_config.build(n_in_channels=n_in, n_outputs=hidden)
        transformer = self.transformer_config.build(dim=hidden)
        return brain, transformer

    def _trainer_setup(self) -> pl.Trainer:
        accelerator, devices = self._accelerator()
        if self.eval_only:
            # Evaluate in a single process: DDP would shard the split across ranks
            # and the prediction callback would only capture one rank's sentences.
            # Running on one device guarantees the saved predictions are complete.
            devices = 1
        callbacks = [
            LogSentencePredictions(save_dir=self.output_dir),
            EarlyStopping(monitor="val_CER", mode="min", patience=self.patience),
        ]
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best",
                    monitor="val_CER",
                    mode="min",
                    save_last=True,
                    save_top_k=1,
                )
            )
        loggers: list = [CSVLogger(self.output_dir, name="logs")]
        if self.wandb_config is not None:
            loggers.append(self._build_wandb_logger())
        strategy = DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto"
        return pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=self.n_epochs,
            accumulate_grad_batches=self.accumulate_grad_batches,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            gradient_clip_val=self.grad_max_norm,
            log_every_n_steps=self.log_every_n_steps,
            use_distributed_sampler=False,
            enable_checkpointing=self.save_checkpoints,
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=self.output_dir,
        )

    def _build_wandb_logger(self):
        try:
            xp_config = self.model_dump(mode="json")
        except Exception:
            xp_config = None
        return self.wandb_config.build(save_dir=self.output_dir, xp_config=xp_config)

    def run(self) -> None:
        pl.seed_everything(self.seed, workers=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        loaders = self.data.build()
        brain, transformer = self._build_modules(loaders["train"])
        metrics = {"CER": CER()}

        if self.eval_only:
            assert self.ckpt_path is not None, "eval requires a checkpoint."
            self._module = BrainModule.load_from_checkpoint(
                self.ckpt_path,
                model=brain,
                transformer=transformer,
                loss=self.loss.build(),
                metrics=metrics,
                optimizer=self.optimizer,
            )
            self._trainer = self._trainer_setup()
            self._trainer.test(self._module, dataloaders=loaders["test"])
            return

        self._module = BrainModule(
            model=brain,
            transformer=transformer,
            loss=self.loss.build(),
            metrics=metrics,
            optimizer=self.optimizer,
        )
        materialize_lazy_params(self._module, loaders["train"])
        self._trainer = self._trainer_setup()
        self._trainer.fit(self._module, loaders["train"], loaders["val"])
        if "test" in loaders:
            self._trainer.test(self._module, dataloaders=loaders["test"])


def _apply_runtime_overrides(cfg: dict, args) -> None:
    """Apply compute-only CLI overrides without changing scientific settings."""
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        for key in ("batch_size", "val_batch_size", "test_batch_size"):
            cfg["data"][key] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        cfg["data"]["num_workers"] = args.num_workers
        if args.num_workers == 0:
            cfg["data"]["persistent_workers"] = False
    if args.no_pin_memory:
        cfg["data"]["pin_memory"] = False
    if args.accumulate_grad_batches is not None:
        if args.accumulate_grad_batches < 1:
            raise ValueError("--accumulate-grad-batches must be positive")
        cfg["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be positive")
        cfg["n_epochs"] = args.max_epochs
    for arg_name, config_name in (
        ("limit_train_batches", "limit_train_batches"),
        ("limit_val_batches", "limit_val_batches"),
        ("limit_test_batches", "limit_test_batches"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            if value < 1:
                raise ValueError(f"--{arg_name.replace('_', '-')} must be positive")
            cfg[config_name] = value


def main(argv: list[str] | None = None) -> None:
    """Run the experiment in a given mode:

    ``python -m brain2qwerty_v1.main {debug,train,eval,cache}`` — every command is
    the same ``Experiment``, only the config (and eval/cache mode) differs.
    """
    import argparse

    import studies  # noqa: F401  (registers the SpanishBCBL study)

    from .cli import add_wandb_args, wandb_config
    from .config.xp_config import (
        debug_config,
        eeg_debug_config,
        eeg_experiment_config,
        experiment_config,
    )

    parser = argparse.ArgumentParser(prog="brain2qwerty_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("debug", help="1-timeline smoke test (default debug config)")
    p_train = sub.add_parser("train", help="full training (1 node, 8 GPUs)")
    p_train.add_argument("--seed", type=int, default=None, help="override the seed")
    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test split")
    p_eval.add_argument("--ckpt", required=True, help="checkpoint to evaluate")
    p_cache = sub.add_parser("cache", help="pre-warm the feature cache")
    p_cache.add_argument("--debug", action="store_true", help="only the debug subset")
    for p in sub.choices.values():
        p.add_argument(
            "--modality",
            choices=("meg", "eeg"),
            default="meg",
            help="recording modality (EEG enables the pre-hardware experiment)",
        )
        p.add_argument(
            "--condition",
            choices=("full", "eeg32"),
            default="full",
            help="EEG montage; ignored for MEG",
        )
        p.add_argument("--output-dir", default=None, help="override results directory")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--no-pin-memory", action="store_true")
        p.add_argument("--accumulate-grad-batches", type=int, default=None)
        p.add_argument("--max-epochs", type=int, default=None)
        p.add_argument("--limit-train-batches", type=int, default=None)
        p.add_argument("--limit-val-batches", type=int, default=None)
        p.add_argument("--limit-test-batches", type=int, default=None)
    for p in (sub.choices["debug"], p_train, p_eval):
        add_wandb_args(p)
    args = parser.parse_args(argv)

    if args.command == "cache":
        if args.modality == "eeg":
            cfg = (
                eeg_debug_config(args.condition)
                if args.debug
                else eeg_experiment_config(args.condition)
            )
        else:
            cfg = debug_config() if args.debug else experiment_config()
        _apply_runtime_overrides(cfg, args)
        print("[brain2qwerty_v1] pre-warming the feature cache...")
        Experiment(**cfg).data.build()
        print("[brain2qwerty_v1] cache warmed.")
        return

    if args.modality == "eeg":
        cfg = (
            eeg_debug_config(args.condition)
            if args.command == "debug"
            else eeg_experiment_config(args.condition)
        )
    else:
        cfg = debug_config() if args.command == "debug" else experiment_config()
    _apply_runtime_overrides(cfg, args)
    if args.command == "eval":
        cfg["eval_only"] = True
        cfg["ckpt_path"] = args.ckpt
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    wandb = wandb_config(args, args.command, cfg.get("seed", 0))
    if wandb is not None:
        cfg["wandb_config"] = wandb
    print(f"[brain2qwerty_v1] running in '{args.command}' mode (seed={cfg.get('seed')})")
    Experiment(**cfg).run()


if __name__ == "__main__":
    main()
