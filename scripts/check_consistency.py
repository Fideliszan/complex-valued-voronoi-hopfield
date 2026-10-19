"""Check archive integrity, generated tables, cited keys, and manuscript assets."""
from pathlib import Path
import hashlib,json,re,tempfile
import nbformat
import pandas as pd
from build_tables import build

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'validation/manifest.json').read_text())
    for name,digest in manifest.items():
        p=ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise AssertionError(f'Checksum mismatch: {name}')
    (ROOT/'runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'runs') as d:
        out=Path(d);build(ROOT/'data/paper',out)
        for p in out.iterdir():
            if p.suffix=='.tex':
                assert p.read_text()==(ROOT/'tables'/p.name).read_text(),p.name
                assert p.read_text()==(ROOT/'paper/Tables'/p.name).read_text(),p.name
    tex=(ROOT/'paper/main.tex').read_text(encoding='utf8');bib=(ROOT/'paper/references.bib').read_text(encoding='utf8')
    assert not re.search(r'(?m)^[ \t]*\w+\s*=\s*[^\n]*[^,\s]\s*\n[ \t]*\w+\s*=',bib), 'Missing BibTeX field separator'
    known=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bib))
    cited={k.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',tex) for k in group.split(',')}
    assert not cited-known, cited-known
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex):
        p=ROOT/'paper'/name;assert p.exists(),name
        assert p.read_bytes()==(ROOT/'figures'/p.name).read_bytes(),name
    for name in re.findall(r'\\input\{([^}]+)\}',tex):assert (ROOT/'paper'/name).exists(),name
    assert 'https://github.com/Fideliszan/complex-valued-voronoi-hopfield' in tex
    notebook_checks=json.loads((ROOT/'validation/notebook_execution.json').read_text())
    for p in (ROOT/'colab').glob('*.ipynb'):
        notebook=nbformat.read(p,as_version=4)
        nbformat.validate(notebook)
        cells=[c for c in notebook.cells if c.cell_type=='code']
        record=notebook_checks[p.name]
        assert record['status']=='passed' and record['mode']=='full',p.name
        assert [hashlib.sha256(c.source.encode()).hexdigest() for c in cells]==record['code_cell_sha256'],p.name
        assert all(c.execution_count is not None for c in cells),p.name
        assert all(not any(o.output_type=='error' for o in c.outputs) for c in cells),p.name
    for name,rows in [('experiment1_final_runs.csv',1600),('experiment2_noise_results.csv',2400),('experiment2_load_results.csv',64)]:
        d=pd.read_csv(ROOT/'data/paper'/name);assert len(d)==rows,name
        assert d.converged.all(),name
    queries=json.loads((ROOT/'validation/query_manifest.json').read_text())
    for name,digest in queries.items():
        assert hashlib.sha256((ROOT/'data/queries'/name).read_bytes()).hexdigest()==digest,name
    boundary=pd.read_csv(ROOT/'validation/boundary_validation.csv');assert boundary.disagreements.sum()==0
    costs=pd.read_csv(ROOT/'data/paper/experiment3_raw_timings.csv')
    assert set(costs.cardinality)=={16,64,256,1024}
    assert len(costs)==600 and costs.agreement_with_distance.min()==1
    for stage in ('saturation','recall'):
        audit=json.loads((ROOT/'validation'/f'{stage}_coordinate_audit_summary.json').read_text())
        assert audit['coordinate_energy_increases']==audit['terminal_rule_violations']==0
    from verify_hosted_results import main as verify_colab
    verify_colab()
    from verify_manuscript_values import verify as verify_values
    verify_values()
    from verify_geometry import verify as verify_geometry
    verify_geometry()
    print(f'PASS: {len(manifest)} checksums; generated tables; {len(cited)} citation keys; five figure assets; four notebooks; full protocols; query arrays; coordinate audits.')

if __name__=='__main__':main()
