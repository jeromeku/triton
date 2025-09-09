
#!/usr/bin/env python3
import importlib, inspect, json
def where(obj):
    try:
        f = inspect.getsourcefile(obj) or inspect.getfile(obj)
        lines, start = inspect.getsourcelines(obj); return {"file": f, "line": start, "nlines": len(lines)}
    except Exception as e:
        return {"file": str(e), "line": -1, "nlines": -1}
out = {}
try:
    from triton.runtime import jit as triton_jit_mod
    deco = getattr(triton_jit_mod, "jit"); out["triton.runtime.jit.jit"] = where(deco)
    JF = getattr(triton_jit_mod, "JITFunction", None)
    if JF: out["triton.runtime.jit.JITFunction"] = where(JF)
except Exception as e: out["jit_error"] = str(e)
try:
    import triton.language as tl
    for name in ["load","store","dot","arange","full","program_id"]:
        if hasattr(tl, name): out[f"triton.language.{name}"] = where(getattr(tl, name))
except Exception as e: out["tl_error"] = str(e)
try:
    gl = importlib.import_module("triton.experimental.gluon.language")
    out["triton.experimental.gluon.language"] = where(gl)
    for name in ["set_auto_layout","convert_layout","static_assert"]:
        if hasattr(gl, name): out[f"gluon.language.{name}"] = where(getattr(gl, name))
except Exception as e: out["gluon_error"] = str(e)
print(json.dumps(out, indent=2))
