"""Save src/detectors/patches/<name>.diff: each adapted script against the authors' original at the pinned commit."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ROOT / "third_party" / "low-resource-hallucination-detection" / "scripts"
PATCHES = ROOT / "src" / "detectors" / "patches"
# adapted copy -> authors' file, relative to ORIGINALS
PAIRS = {
    "ModelArtifacts_g6": "MAM_HD_methods/ModelArtifacts.py",
    "ModelArtifacts_Classifier_g6": "MAM_HD_methods/ModelArtifacts_Classifier.py",
    "ModelArtifacts_multiple_gen_tokens_g6": "MAM_HD_methods/ModelArtifacts_multiple_gen_tokens.py",
    "ModelArtifacts_Classifier_multiple_gen_tokens_g6": "MAM_HD_methods/ModelArtifacts_Classifier_multiple_gen_tokens.py",
    "selfcheckGPT_g6": "Sampling_HD_methods/selfcheckGPT.py",
    "semanticEntropy_g6": "Sampling_HD_methods/semanticEntropy.py",
}


def make_diff(name: str) -> str:
    """Relative paths and no `index` lines keep the diff the same on every machine and checkout."""
    original = (ORIGINALS / PAIRS[name]).relative_to(ROOT).as_posix()
    ours = f"src/detectors/{name}.py"
    # git exits 1 when the files differ, which is the normal case here
    out = subprocess.run(["git", "diff", "--no-index", "--no-color", "--ignore-space-at-eol", original, ours],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    return "".join(line + "\n" for line in out.splitlines() if not line.startswith("index "))


def main() -> None:
    PATCHES.mkdir(parents=True, exist_ok=True)
    for name in PAIRS:
        (PATCHES / f"{name}.diff").write_text(make_diff(name), encoding="utf-8", newline="\n")
        print(f"saved {name}.diff")


if __name__ == "__main__":
    main()
