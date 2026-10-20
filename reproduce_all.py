"""Run all three experiments sequentially, then compare their trial results."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cvhnn import settings
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-save-data',dest='save_data',action='store_false',default=settings.SAVE_DATA,help='Keep only comparison.json and run records, not the trial data.')
    p.add_argument('--no-save-figures',dest='save_figures',action='store_false',default=settings.SAVE_FIGURES,help='Discard generated figures.')
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory.')
    # Stages keep everything until the comparison has read it; the switches apply afterwards.
    env={**os.environ,'CVHNN_SAVE_DATA':'1','CVHNN_SAVE_FIGURES':'1'}
    for stage in ['saturation','recall','activation_cost']:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),stage,'--mode',a.mode,'--output',str(out/stage)],check=True,cwd=ROOT,env=env)
    comparison=[sys.executable,str(ROOT/'scripts/compare_results.py'),str(out),'--reference',str(ROOT/'data/paper'),'--output',str(out/'comparison.json')]
    if a.mode=='smoke':comparison.append('--allow-partial')
    subprocess.run(comparison,check=True,cwd=ROOT)
    for stage in ['saturation','recall','activation_cost']:
        settings.retain(out/stage,a.save_data,a.save_figures)
