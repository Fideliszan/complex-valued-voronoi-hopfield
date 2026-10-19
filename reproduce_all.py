"""Run all three experiments sequentially, then compare their trial results."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory.')
    for stage in ['saturation','recall','activation_cost']:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),stage,'--mode',a.mode,'--output',str(out/stage)],check=True,cwd=ROOT)
    comparison=[sys.executable,str(ROOT/'scripts/compare_results.py'),str(out),'--reference',str(ROOT/'data/paper'),'--output',str(out/'comparison.json')]
    if a.mode=='smoke':comparison.append('--allow-partial')
    subprocess.run(comparison,check=True,cwd=ROOT)
