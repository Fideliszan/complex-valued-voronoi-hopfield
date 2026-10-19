"""Keep experimental inputs/CSV schemas shared while replacing numerical kernels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import independent as core


def install(module, stage):
    audits=[]
    if stage=='saturation':
        module.hermitian_weight_matrix=core.random_weights
        def runner(alphabet,w,initial_indices,schedule,rule_name,activation,max_sweeps=80):
            result=core.dynamics(alphabet,w,initial_indices,schedule,rule_name,max_sweeps,refresh=False)
            audits.append(dict(rule=rule_name,**result.audit))
            history=pd.DataFrame(result.history)
            column='E_c' if rule_name=='classic' else 'E_r'
            return module.Trajectory(history,result.indices,result.sweeps*len(initial_indices),result.converged,
                                     int((history[column].diff()>1e-10).sum()),column)
        module.run_dynamics=runner
    elif stage=='recall':
        module.build_projection_weights=core.projection_weights
        module.quantize_complex_to_indices=core.quantize
        module.corrupt_symbol_indices=core.corrupt_symbols
        module.corrupt_hsv_and_quantize=core.corrupt_hsv
        module.compute_metrics=core.retrieval_metrics
        module.hsv_fields_to_complex=core.encode
        def runner(alphabet,w,initial_indices,schedule,rule,max_sweeps=80):
            result=core.dynamics(alphabet,w,initial_indices,schedule,rule,max_sweeps,relative=True)
            audits.append(dict(rule=rule,**result.audit))
            history=pd.DataFrame(result.history)
            column='E_c' if rule=='classic' else 'E_r'
            return module.DynamicsResult(result.indices,result.sweeps,result.sweeps*len(initial_indices),
                result.changed,result.converged,int((history[column].diff()>1e-9).sum()),
                history.E_c.iloc[0],history.E_c.iloc[-1],history.E_r.iloc[0],history.E_r.iloc[-1])
        module.run_dynamics=runner
    else:
        raise ValueError(stage)
    return audits


def save_audits(audits,output):
    frame=pd.DataFrame(audits)
    frame.to_csv(Path(output)/'coordinate_audits.csv',index=False)
    report=dict(implementation='independent equation-based kernels',trajectories=len(frame),
        changed_updates=int(frame.changed_updates.sum()),
        coordinate_energy_increases=int(frame.coordinate_energy_increases.sum()),
        nonnegative_energy_changes=int(frame.nonnegative_energy_changes.sum()),
        terminal_rule_violations=int(frame.terminal_rule_violations.sum()),
        max_field_residual=float(frame.max_field_residual.max()),
        max_energy_accumulation_residual=float(frame.max_energy_accumulation_residual.max()))
    (Path(output)/'coordinate_audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
