
import importlib, inspect, sys
def where(obj):
    try:
        mod = inspect.getmodule(obj)
        return getattr(mod, "__name__", str(mod))
    except Exception:
        return "<builtin/unknown>"
def show(name):
    try:
        import triton.language as tl
    except Exception as e:
        print("ImportError: install triton first:", e); sys.exit(1)
    obj = getattr(tl, name)
    print(f"{name:20s} -> {where(obj)} | type={type(obj)}")
if __name__ == "__main__":
    for sym in ["load","store","program_id","sum","maximum","where","sort","associative_scan"]:
        show(sym)
