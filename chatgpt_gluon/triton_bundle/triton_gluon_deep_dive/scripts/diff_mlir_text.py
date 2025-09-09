
#!/usr/bin/env python3
import sys, re, collections, pathlib
DIALECT_RX = re.compile(r'([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)')
PASS_RX = re.compile(r'^\*\*\* IR Dump After (.+)$')
def scan(path):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    ops = collections.Counter(); passes = collections.Counter()
    for line in text:
        m = DIALECT_RX.search(line); 
        if m: ops[m.group(1)] += 1
        b = PASS_RX.match(line.strip())
        if b: passes[b.group(1)] += 1
    return {"ops": ops, "passes": passes}
def show(a, b):
    def sec(name, ca, cb):
        keys = sorted(set(ca)|set(cb))
        print(f"\n# {name}")
        for k in keys:
            va, vb = ca.get(k,0), cb.get(k,0)
            tag = "==" if va and vb else ("- " if va else "+ ")
            print(f"{tag} {k:32s} {va:6d} | {vb:6d}")
    sec("Dialect/Op prefixes", a["ops"], b["ops"])
    sec("Pass banners", a["passes"], b["passes"])
if __name__ == "__main__":
    if len(sys.argv)!=3: 
        print("usage: diff_mlir_text.py A.mlir B.mlir"); sys.exit(1)
    A = scan(sys.argv[1]); B = scan(sys.argv[2]); show(A,B)
