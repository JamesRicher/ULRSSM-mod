import os

INPUT_DIR = "/Users/james/Documents/Imperial/ShapeMatching/Code/ULRSSM/results/faust/visualization"
OUTPUT_DIR = "/Users/james/Documents/Imperial/ShapeMatching/Code/ULRSSM/results/faust/fixed_vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".obj"):
        continue

    with open(os.path.join(INPUT_DIR, filename), 'r') as f:
        lines = f.readlines()

    other = [l for l in lines if not l.startswith('vt') and not l.startswith('f ')]
    vts   = [l for l in lines if l.startswith('vt')]
    faces = [l for l in lines if l.startswith('f ')]

    # Write: everything else, then vt, then f
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        f.writelines(other + vts + faces)

    print(f"Fixed {filename}")