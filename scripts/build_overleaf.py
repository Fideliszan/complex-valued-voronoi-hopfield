"""Build an Overleaf ZIP and a complete computational source ZIP from tracked files."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    # Export Git blob bytes, not platform-specific working-tree line endings.
    archived=subprocess.check_output(['git','archive','--format=zip','HEAD'],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archived)) as z:
        sources={p:z.read(p) for p in z.namelist() if not p.endswith('/')}
    if not sources:raise RuntimeError('Commit the complete source tree first.')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    version=(ROOT/'RELEASE_VERSION').read_text().strip()
    manifests={}
    for name,overleaf in [('Complex_Voronoi_Overleaf_'+version+'.zip',True),('complex-valued-voronoi-hopfield_'+version+'_source.zip',False)]:
        content={}
        for name_in_repo,payload in sources.items():
            p=Path(name_in_repo)
            if overleaf:
                content['reproducibility/'+p.as_posix()]=payload
                if p.parts[0]=='paper':content[Path(*p.parts[1:]).as_posix()]=payload
            else:
                content['complex-valued-voronoi-hopfield/'+p.as_posix()]=payload
        manifest={'git_commit':commit,'version':version,'files':{k:hashlib.sha256(v).hexdigest() for k,v in content.items()}}
        content['PACKAGE_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
        if overleaf:
            content['OVERLEAF_README.txt']=b'Compile main.tex with pdfLaTeX and BibTeX. The complete computational companion is in reproducibility/. Its README.md explains independent full runs, notebooks, data, and numerical validation. The package manifest records every source file and its Git commit.\n'
        with zipfile.ZipFile(output/name,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for path,data in sorted(content.items()):z.writestr(path,data)
        with zipfile.ZipFile(output/name) as z:
            assert z.testzip() is None
            for path,digest in manifest['files'].items():assert hashlib.sha256(z.read(path)).hexdigest()==digest,path
        manifests[name]=dict(sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),bytes=(output/name).stat().st_size,git_commit=commit,files=len(content))
    (output/'delivery_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
    print(json.dumps(manifests,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();build(args.output)
