
import ast, inspect, triton, triton.language as tl
@triton.jit
def saxpy(a, x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    pid = tl.program_id(0)
    i = pid + tl.arange(0, n)
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + i)
    tl.store(out_ptr + i, a * x + y)
if __name__ == "__main__":
    src = inspect.getsource(saxpy.fn if hasattr(saxpy,"fn") else saxpy)
    print("=== Python source ==="); print(src)
    tree = ast.parse(src)
    print("\n=== Python AST ==="); print(ast.dump(tree, include_attributes=False, indent=2))
