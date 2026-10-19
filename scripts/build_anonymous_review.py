"""Export an anonymous manuscript-only ZIP, preserving the identified paper."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paper = ROOT / "paper"
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    start = tex.index("\\author{")
    end = tex.index("\\begin{document}", start)
    tex = tex[:start] + "\\author{Anonymous Authors}\n\\hypersetup{pdfauthor={},pdfsubject={Anonymous review manuscript}}\n\n" + tex[end:]
    start = tex.index("The Python implementations, trial-level measurements")
    end = tex.index("Reported benchmark times", start)
    tex = tex[:start] + (
        "The computational companion contains Python implementations, trial-level\n"
        "measurements, ordered alphabet coordinates, coordinate-level energy audits,\n"
        "figure generators, and four Google Colab notebooks. Its identifying\n"
        "repository URL and software citation are withheld for double-anonymous\n"
        "review. The experimental protocols and numerical results reported here\n"
        "are unchanged.\n"
    ) + tex[end:]
    bib = (paper / "references.bib").read_text(encoding="utf-8")
    bib, removed = re.subn(r"@misc\{castro2026complexsoftware,.*?\n\}\n", "", bib, flags=re.S)
    assert removed == 1
    assert "Fideliszan" not in tex + bib
    assert "\\thanks" not in tex
    assert "castro2026complexsoftware" not in tex + bib
    contents = {"main.tex": tex.encode(), "references.bib": bib.encode()}
    for folder in ("Figures", "Tables"):
        for path in sorted((paper / folder).rglob("*")):
            if path.is_file():
                contents[path.relative_to(paper).as_posix()] = path.read_bytes()
    contents["REVIEW_README.txt"] = (
        "Compile main.tex with pdfLaTeX and BibTeX. This is the anonymous manuscript-only copy.\n"
        "Author names/affiliations and identifying companion links are omitted. Published\n"
        "literature citations are retained in third-person form. Scientific text, data,\n"
        "figures and tables are preserved apart from the anonymized availability paragraph.\n"
        "Do not attach the identified reproducibility archive to a double-anonymous review.\n"
        "Author metadata, ORCIDs and any applicable disclosures belong in the submission workflow.\n"
    ).encode()
    filename = "Complex_Voronoi_TNNLS_Anonymous_Review_1.0.2.zip"
    target = output / filename
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(z.read(name) == data for name, data in contents.items())
    record = {"filename": filename, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
    (output / "anonymous_package_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
