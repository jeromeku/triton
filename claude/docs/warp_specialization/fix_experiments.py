#!/usr/bin/env python3
import re

def fix_experiments(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Fix experiments directory references
    content = re.sub(
        r'\*\*File\*\*:\s*`/home/jeromeku/triton/(experiments/[^\s:`]+\.py)(?::(\d+)(?:-(\d+))?)?`',
        lambda m: f'**File**: [`{m.group(1)}{f":{m.group(2)}" if m.group(2) else ""}{f"-{m.group(3)}" if m.group(3) else ""}`](../../../{m.group(1)}{f"#L{m.group(2)}" if m.group(2) else ""})',
        content
    )
    
    # Fix plain references
    content = re.sub(
        r'`/home/jeromeku/triton/(experiments/[^\s:`]+\.py)(?::(\d+)(?:-(\d+))?)?`',
        lambda m: f'[`{m.group(1)}{f":{m.group(2)}" if m.group(2) else ""}{f"-{m.group(3)}" if m.group(3) else ""}`](../../../{m.group(1)}{f"#L{m.group(2)}" if m.group(2) else ""})',
        content
    )
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Fixed experiments links in {filename}")

fix_experiments("frame_by_frame_trace_test_warp_specialize_tma_matmul.md")
