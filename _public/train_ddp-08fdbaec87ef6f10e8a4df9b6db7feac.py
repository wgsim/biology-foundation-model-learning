"""Minimal DistributedDataParallel (DDP) fine-tuning script for the course (module 06).

Runs with `torchrun` on 1 process (reference), N CPU processes (gloo; smoke test on a laptop),
or N GPUs (nccl; cluster). It fine-tunes ESM-2 with LoRA to regress a scalar from the hidden state
at one residue per sequence, using synthetic data so that it needs no dataset download.

The point of the script is the *distributed mechanics*, which notebook 03 checks:
  * DistributedSampler shards the data (rank r gets samples r, r+W, r+2W, ...),
  * DDP averages gradients across ranks (mean-reduced loss => same update as one process
    with the union of the rank batches),
  * all ranks hold identical parameters after every step.

Example (2 CPU processes):
  torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 train_ddp.py --steps 4 --per-rank-batch 8 --out /tmp/ddp.json
  (--local-addr=127.0.0.1 is needed on some macOS setups where the hostname does not resolve to a reachable address)
Example (reference, 1 process, same global batch):
  torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=1 train_ddp.py --steps 4 --per-rank-batch 16 --out /tmp/single.json
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import EsmModel


class SynthSites(Dataset):
    """Random protein-like sequences with a deterministic scalar target (no download needed)."""

    AA = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, n, length, tok, seed=0):
        rng = np.random.default_rng(seed)
        seqs = ["".join(rng.choice(list(self.AA), size=length)) for _ in range(n)]
        self.ids = tok(seqs, return_tensors="pt")["input_ids"]
        self.pos = torch.tensor(rng.integers(1, length + 1, size=n))
        self.y = torch.tensor(rng.normal(size=n).astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.pos[i], self.y[i]


class SiteRegressor(nn.Module):
    def __init__(self, backbone, d):
        super().__init__()
        self.backbone, self.head = backbone, nn.Linear(d, 1)

    def forward(self, ids, pos):
        h = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        return self.head(h[torch.arange(len(ids), device=ids.device), pos]).squeeze(-1)


def param_hash(model):
    """Order-stable hash of all trainable parameters (to compare ranks / runs)."""
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().cpu().double().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--revision", default=None, help="pin the Hub revision (recommended on a cluster)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--per-rank-batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-params", default=None, help="rank 0 saves the flattened trainable parameters (.npy) for comparisons")
    ap.add_argument("--no-sampler", action="store_true", help="NEGATIVE CONTROL: do not shard data (every rank sees the same samples)")
    a = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(device)

    torch.manual_seed(a.seed)                                   # same init on every rank (LoRA A/B and head)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    backbone = EsmModel.from_pretrained(a.model, revision=a.revision, add_pooling_layer=False)
    backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["query", "value"], bias="none"))
    net = SiteRegressor(backbone, backbone.base_model.model.config.hidden_size).to(device)
    model = DDP(net, device_ids=[local_rank] if use_cuda else None) if world > 1 else net
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=a.lr)     # SGD: easy to compare across process counts

    ds = SynthSites(a.n, a.length, tok, seed=a.seed)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True) if (world > 1 and not a.no_sampler) else None
    dl = DataLoader(ds, batch_size=a.per_rank_batch, sampler=sampler, shuffle=False, drop_last=True)

    losses, step, epoch = [], 0, 0
    model.train()
    while step < a.steps:                                       # repeat epochs until the requested number of steps is done
        if sampler is not None:
            sampler.set_epoch(epoch)                            # reshuffles when shuffle=True (a no-op here, kept for correctness)
        n_batches = 0
        for ids, pos, y in dl:
            if step >= a.steps:
                break
            loss = nn.functional.mse_loss(model(ids.to(device), pos.to(device)), y.to(device))      # mean over the rank batch
            opt.zero_grad()
            loss.backward()                                                                        # DDP all-reduces (averages) gradients here
            opt.step()
            losses.append(loss.item())
            step += 1
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("dataset smaller than per-rank-batch x world_size: no batch can be formed")
        epoch += 1

    h = param_hash(net)
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, h)
    else:
        gathered = [h]
    if rank == 0:
        res = dict(world=world, per_rank_batch=a.per_rank_batch, steps=step, losses=losses, param_hash_per_rank=gathered, all_ranks_equal=len(set(gathered)) == 1)
        print(json.dumps(res))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f)
        if a.save_params:
            np.save(a.save_params, np.concatenate([p.detach().cpu().double().numpy().ravel() for n, p in sorted(net.named_parameters()) if p.requires_grad]))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
