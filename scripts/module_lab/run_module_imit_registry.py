from __future__ import annotations
import argparse, json
from typing import Any
import torch
from scripts.module_lab.module_imit_registry import IMIT_REGISTRY, get_imitator

def summarize(value:Any)->Any:
    if torch.is_tensor(value):
        return {"type":"tensor","shape":list(value.shape),"finite":bool(torch.isfinite(value).all().item())}
    if isinstance(value,dict):
        return {k:summarize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [summarize(v) for v in value[:3]]
    return value

def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--module", default="all")
    args=parser.parse_args()
    names=sorted(IMIT_REGISTRY) if args.module=="all" else [args.module]
    result={}
    for name in names:
        result[name]=summarize(get_imitator(name)())
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
