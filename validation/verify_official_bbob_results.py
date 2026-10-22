#!/usr/bin/env python3
"""Reject incomplete or malformed COCO/BBOB GitHub Actions archives.
This checks execution integrity only; it does not certify algorithmic superiority.
"""
import csv,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
out=Path("validation_output")
csv_path=out/"raw_coco_bbob.csv"
manifest_path=out/"manifest.json"
if not csv_path.is_file() or not manifest_path.is_file():
    raise SystemExit("VALIDATION_FAILED: missing raw_coco_bbob.csv or manifest.json")
with manifest_path.open(encoding="utf-8") as f: m=json.load(f)
with csv_path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
expected=24*2*5*3*3
if m.get("status")!="completed":raise SystemExit("VALIDATION_FAILED: manifest status is not completed")
if len(rows)!=expected:raise SystemExit(f"VALIDATION_FAILED: expected {expected} rows, got {len(rows)}")
if m.get("rows")!=expected:raise SystemExit(f"VALIDATION_FAILED: manifest rows {m.get('rows')} != {expected}")
computed=hashlib.sha256(csv_path.read_bytes()).hexdigest()
if m.get("raw_sha256")!=computed:raise SystemExit("VALIDATION_FAILED: raw SHA256 does not match")
keys=set(); funcs=set();dims=set();instances=set();budgets=set(); variants=set()
for i,r in enumerate(rows):
    fid=int(r["function"]);dim=int(r["dimension"]);ins=int(r["instance"]);factor=int(r["budget_multiplier"]);budget=int(r["budget"]);var=r["variant"]
    if fid not in range(1,25) or dim not in (10,20) or ins not in range(1,6) or factor not in (50,150,300) or budget!=factor*dim:raise SystemExit(f"VALIDATION_FAILED: design invalid in row {i}")
    if int(r["evaluations"])!=budget:raise SystemExit(f"VALIDATION_FAILED: FE mismatch row {i}")
    v=float(r["error"]);val=float(r["best_f"]);opt=float(r["optimum_f"])
    if not all(map(math.isfinite,(v,val,opt))) or v<0 or not math.isclose(v,max(0.,val-opt),rel_tol=1e-12,abs_tol=1e-12):raise SystemExit(f"VALIDATION_FAILED: invalid error row {i}")
    key=(fid,dim,ins,factor,var)
    if key in keys:raise SystemExit(f"VALIDATION_FAILED: repeated key {key}")
    keys.add(key);funcs.add(fid);dims.add(dim);instances.add(ins);budgets.add(factor);variants.add(var)
if funcs!=set(range(1,25)) or dims!={10,20} or instances!={1,2,3,4,5} or budgets!={50,150,300} or variants!={"full","no_handoff","de_stage_only"}:raise SystemExit("VALIDATION_FAILED: coverage mismatch")
msg=f"OFFICIAL_BBOB_DATA_INTEGRITY_PASSED: {expected} rows; complete 24 functions, D10/20, 5 instances, 3 budgets, 3 variants; checksum {computed}"
print(msg,flush=True)
summary=Path("validation_output/RESULTS_VERIFIED.txt")
summary.write_text(msg+"\n"+json.dumps({"manifest":m,"unique_experiment_keys":len(keys)},indent=2)+"\n")
