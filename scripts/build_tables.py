"""Generate the numerical manuscript tables and complete cardinality summary from trial-level CSVs."""
from pathlib import Path
import argparse, json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def build(data,out):
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(data/'experiment1_final_runs.csv').query('init_protocol == "uniform"')
    rows=[];tex=[]
    for alphabet in ['16-PSK','16-QAM','16-Polar','19-Hexagonal']:
        c=d[(d.alphabet==alphabet)&(d.rule=='classic')];v=d[(d.alphabet==alphabet)&(d.rule=='voronoi')]
        r={'alphabet':alphabet,'rho0':c.initial_rho.mean(),'b0':c.initial_boundary_fraction.mean(),'rho_c':c.rho.mean(),'rho_v':v.rho.mean(),'rho_v_std':v.rho.std(),'b_c':c.boundary_fraction.mean(),'b_v':v.boundary_fraction.mean(),'b_v_std':v.boundary_fraction.std()};rows.append(r)
        tex.append(f"{alphabet} & {r['rho0']:.3f} & {r['b0']:.3f} & {r['rho_c']:.3f} & ${r['rho_v']:.3f} \\pm {r['rho_v_std']:.3f}$ & {r['b_c']:.3f} & ${r['b_v']:.3f} \\pm {r['b_v_std']:.3f}$ \\\\")
    pd.DataFrame(rows).to_csv(out/'saturation.csv',index=False);(out/'saturation.tex').write_text('\n'.join(tex)+'\n')
    d=pd.read_csv(data/'experiment2_noise_results.csv').query('noise_level == 0.1')
    rows=[];tex=[];ranges={}
    for kind,label in [('symbol','Sym.'),('hsv','HSV')]:
        if tex:tex.append(r'\midrule')
        for alphabet in ['64-PSK','64-QAM','64-Polar','61-Hexagonal']:
            q=d[(d.corruption_kind==kind)&(d.alphabet==alphabet)];c=q[q.rule=='classic'];v=q[q.rule=='voronoi']
            r={'kind':kind,'alphabet':alphabet,'initial_SER':c.initial_SER.mean(),'SER_c':c.SER.mean(),'SER_v':v.SER.mean(),'radial_c':c.radial_error.mean(),'radial_v':v.radial_error.mean(),'exact_v':v.exact_recovery.mean()};rows.append(r)
            tex.append(f"{label} & {alphabet.replace('Hexagonal','Hex.')} & {r['SER_c']:.3f} & {r['SER_v']:.3f} & {r['radial_c']:.3f} & {r['radial_v']:.3f} & {r['exact_v']:.3f} \\\\")
    moderate=pd.DataFrame(rows);moderate.to_csv(out/'recall.csv',index=False);(out/'recall.tex').write_text('\n'.join(tex)+'\n')
    timing=pd.read_csv(data/'experiment3_raw_timings.csv')
    timing.groupby(['ensemble','family','cardinality','method'],as_index=False).agg(
        mean_us=('time_per_query_us','mean'),median_us=('time_per_query_us','median'),
        min_us=('time_per_query_us','min'),max_us=('time_per_query_us','max')).to_csv(out/'activation_cost.csv',index=False)
    vals={'symbol_initial_SER_range':[float(x) for x in moderate.query('kind=="symbol"').initial_SER.agg(['min','max'])],
          'hsv_initial_SER_range':[float(x) for x in moderate.query('kind=="hsv"').initial_SER.agg(['min','max'])],
          'max_projection_residual':float(pd.read_csv(data/'experiment2_weight_validation.csv').projection_residual.max())}
    (out/'text_values.json').write_text(json.dumps(vals,indent=2)+'\n')
    headers={
        'saturation':('lcccccc',r'Alphabet & $\rho(0)$ & $b(0)$ & $\rho_{\infty}^{c}$ & $\rho_{\infty}^{V}$ & $b_{\infty}^{c}$ & $b_{\infty}^{V}$'),
        'recall':('llccccc',r'Pert. & Alphabet & $\operatorname{SER}_{c}$ & $\operatorname{SER}_{V}$ & $e_{r,c}$ & $e_{r,V}$ & Exact$_V$')}
    for name,(columns,header) in headers.items():
        f=out/(name+'.tex');body=f.read_text()
        f.write_text('\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+body+'\\bottomrule\n\\end{tabular}\n')
    return vals

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/paper');p.add_argument('--output',type=Path,default=ROOT/'tables');a=p.parse_args();print(build(a.data,a.output))
