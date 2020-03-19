#!/usr/bin/env python
"""
Convert trajectories from `imitation` format to openai/baselines GAIL format.
"""

import argparse
import os
from pathlib import Path
import pickle
from typing import List

import numpy as np

from imitation.util import rollout


def convert_trajs_to_sb(trajs: List[rollout.Trajectory]) -> dict:
  """Converts Trajectories into the dict format used by Stable Baselines GAIL.
  """
  trans = rollout.flatten_trajectories(trajs)
  return dict(
    acs=trans.acts,
    rews=trans.rews,
    obs=trans.obs,
    ep_rets=np.array([np.sum(t.rews) for t in trajs]),
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("src_path", type=str)
  parser.add_argument("dst_path", type=str)
  args = parser.parse_args()

  src_path = Path(args.src_path)
  dst_path = Path(args.dst_path)

  assert src_path.is_file()
  with open(src_path, "rb") as f:
    src_trajs = pickle.load(f)  # type: List[rollout.Trajectory]

  dst_trajs = convert_trajs_to_sb(src_trajs)
  os.makedirs(dst_path.parent, exist_ok=True)
  with open(dst_path, "wb") as f:
    np.savez_compressed(f, **dst_trajs)

  print(f"Dumped rollouts to {dst_path}")


if __name__ == "__main__":
  main()