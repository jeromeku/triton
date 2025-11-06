#!/usr/bin/env python3
import re
import sys

def fix_links_in_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern 1: Absolute paths with line numbers
    # /home/jeromeku/triton/path/to/file.py:123 or :123-456
    content = re.sub(
        r'/home/jeromeku/triton/((?:python|lib|include|third_party)/[^\s:`]+\.(?:py|cc|cpp|h|td))(?::(\d+)(?:-(\d+))?)?',
        lambda m: f'[{m.group(1)}{f":{m.group(2)}" if m.group(2) else ""}](../../../{m.group(1)}{f"#L{m.group(2)}" if m.group(2) else ""})',
        content
    )
    
    # Pattern 2: File references in markdown format already but with absolute paths
    # **File**: `/home/jeromeku/triton/...`
    content = re.sub(
        r'\*\*File\*\*:\s*`/home/jeromeku/triton/((?:python|lib|include|third_party)/[^`]+)`',
        lambda m: f'**File**: [`{m.group(1)}`](../../../{m.group(1)})',
        content
    )
    
    # Pattern 3: Location comments with absolute paths
    # **Location**: `/home/jeromeku/triton/path/file.py:123`
    content = re.sub(
        r'\*\*Location\*\*:\s*`/home/jeromeku/triton/((?:python|lib|include|third_party)/[^\s:`]+\.(?:py|cc|cpp|h|td))(?::(\d+)(?:-(\d+))?)?`',
        lambda m: f'**Location**: [`{m.group(1)}{f":{m.group(2)}" if m.group(2) else ""}`](../../../{m.group(1)}{f"#L{m.group(2)}" if m.group(2) else ""})',
        content
    )
    
    # Pattern 4: Inline paths in backticks without Location/File prefix
    # `path/to/file.py:123` that haven't been converted yet
    content = re.sub(
        r'(?<!Location\]: )(?<!File\]: )`/home/jeromeku/triton/((?:python|lib|include|third_party)/[^\s:`]+\.(?:py|cc|cpp|h|td))(?::(\d+))?`(?!\])',
        lambda m: f'[`{m.group(1)}{f":{m.group(2)}" if m.group(2) else ""}`](../../../{m.group(1)}{f"#L{m.group(2)}" if m.group(2) else ""})',
        content
    )
    
    # Pattern 5: Fix references to the trace documents themselves
    content = content.replace(
        '`/home/jeromeku/triton/frame_by_frame_trace_test_warp_specialize_basic_ir.md`',
        '[`frame_by_frame_trace_test_warp_specialize_basic_ir.md`](./frame_by_frame_trace_test_warp_specialize_basic_ir.md)'
    )
    content = content.replace(
        '`/home/jeromeku/triton/frame_by_frame_trace_test_warp_specialize_tma_matmul.md`',
        '[`frame_by_frame_trace_test_warp_specialize_tma_matmul.md`](./frame_by_frame_trace_test_warp_specialize_tma_matmul.md)'
    )
    content = content.replace(
        '(/home/jeromeku/triton/frame_by_frame_trace_test_warp_specialize_basic_ir.md)',
        '(./frame_by_frame_trace_test_warp_specialize_basic_ir.md)'
    )
    content = content.replace(
        '(/home/jeromeku/triton/frame_by_frame_trace_test_warp_specialize_tma_matmul.md)',
        '(./frame_by_frame_trace_test_warp_specialize_tma_matmul.md)'
    )
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Fixed links in {filename}")

if __name__ == "__main__":
    fix_links_in_file("warp_specialization.md")
    fix_links_in_file("frame_by_frame_trace_test_warp_specialize_basic_ir.md")
    fix_links_in_file("frame_by_frame_trace_test_warp_specialize_tma_matmul.md")
    print("All files updated!")
