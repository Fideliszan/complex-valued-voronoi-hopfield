"""Execute the paper protocol with independently implemented numerical kernels."""
from pathlib import Path
import argparse, importlib, json, os, platform, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main(stage=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if stage is None: parser.add_argument('stage',choices=['saturation','recall','activation_cost'])
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engine',choices=['independent','reference'],default='independent')
    args=parser.parse_args()
    stage=stage or args.stage
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(f'Choose a new output directory: {out}')
    m=importlib.import_module('cvhnn.'+stage)
    m.OUTPUT_DIR=out
    audits=None
    if args.mode=='smoke':
        if stage=='saturation': m.N_RUNS=2; m.N=20
        elif stage=='recall': m.N_TRIALS_NOISE=1; m.N_TRIALS_LOAD=1
        else: m.N_QUERY_UNIFORM=200; m.N_QUERY_FIELDS=200; m.N_REPEATS=2; m.MIN_FIELD_TRIALS=1
    if args.engine=='independent' and stage!='activation_cost':
        from cvhnn.independent_adapter import install
        audits=install(m,stage)
    if args.engine=='independent' and stage=='activation_cost':
        from cvhnn.independent_benchmark import install
        install(m)
    # A single BLAS thread avoids oversubscribing the many small matrix operations.
    from threadpoolctl import threadpool_limits
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        if stage=='saturation':
            m.run_experiment()
        else:
            m.SAVE_FIGURES=(stage=='recall')
            m.main()
    config={k:v for k,v in vars(m).items() if k.isupper() and isinstance(v,(str,int,float,bool,tuple))}
    import numpy, pandas, scipy
    record={'stage':stage,'mode':args.mode,'engine':args.engine,'configuration':config,'python':sys.version,'platform':platform.platform(),'packages':{'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__},'blas_threads':1,'seconds':time.perf_counter()-start}
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    if audits is not None:
        from cvhnn.independent_adapter import save_audits
        save_audits(audits,out)

if __name__=='__main__':main()
