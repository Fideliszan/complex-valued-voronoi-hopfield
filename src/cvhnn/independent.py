"""Equation-based implementation, independent of the reference kernels.

Alphabet coordinates and experimental seeds are input data. No activation,
network-update, weight, corruption, or metric routine is imported from a reference
module. The adapters reuse only its trial enumeration and CSV presentation.
"""
from dataclasses import dataclass
import numpy as np
from matplotlib.colors import hsv_to_rgb


def decision(alphabet, field, current=None, rule="voronoi", atol=1e-12,
             relative=False):
    """Conservative minimization; the array position is the fixed total order."""
    if rule == "voronoi":
        value = (alphabet.real-field.real)**2 + (alphabet.imag-field.imag)**2
    elif rule == "classic":
        value = -(alphabet.real*field.real + alphabet.imag*field.imag)
    else:
        raise ValueError(rule)
    best = value.min()
    tolerance = atol * (max(1., abs(best)) if relative else 1.)
    eligible = value <= best + tolerance
    if current is not None and eligible[current]:
        return int(current)
    return int(np.flatnonzero(eligible)[0])


def quantize(z, alphabet, chunk_size=4096):
    """Static quantizer: smallest index among numerical nearest neighbours."""
    z = np.asarray(z).reshape(-1)
    answer = np.empty(z.size, dtype=np.int64)
    for first in range(0, z.size, chunk_size):
        block = z[first:first+chunk_size]
        d2 = (block.real[:, None]-alphabet.real)**2 + (block.imag[:, None]-alphabet.imag)**2
        answer[first:first+len(block)] = np.argmax(d2 <= d2.min(axis=1)[:,None]+1e-12, axis=1)
    return answer


def random_weights(n, rng, diagonal_value=0., coupling_scale=1.):
    a = (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))) / np.sqrt(2*n)
    np.fill_diagonal(a, 0)
    w = coupling_scale*(a + a.conj().T)/2
    np.fill_diagonal(w, diagonal_value)
    return w


def projection_weights(memories, rcond=1e-10):
    x = np.asarray(memories, dtype=complex).T
    gram = x.conj().T @ x
    w = x @ np.linalg.pinv(gram, rcond=rcond) @ x.conj().T
    w = (w+w.conj().T)/2
    info = dict(rank_gram=int(np.linalg.matrix_rank(gram, tol=rcond)),
                hermitian_error=float(np.max(np.abs(w-w.conj().T))),
                min_diag_real=float(w.diagonal().real.min()),
                max_diag_imag_abs=float(np.abs(w.diagonal().imag).max()),
                projection_residual=float(np.linalg.norm(w@x-x)/max(np.linalg.norm(x),1e-12)),
                diag_nonnegative=bool(w.diagonal().real.min() >= -1e-10))
    return w, info


def energy(state, w):
    ec = -.5 * np.vdot(state, w@state).real
    return float(ec), float(ec+.5*np.vdot(state,state).real)


def encode(hue, saturation, rmax=1.):
    return rmax*np.asarray(saturation)*np.exp(1j*(2*np.pi*np.asarray(hue)-np.pi))


def rgb(state, rmax=1.):
    hue = ((np.angle(state)+np.pi)/(2*np.pi)) % 1.
    sat = np.minimum(np.abs(state)/rmax,1.)
    return hsv_to_rgb(np.stack((hue,sat,np.full_like(hue,.8)),axis=-1))


def corrupt_symbols(target, size, probability, rng):
    indices = np.array(target, copy=True)
    mask = rng.random(indices.shape) < probability
    indices[mask] = rng.integers(size, size=mask.sum())
    return indices


def corrupt_hsv(hue, saturation, alphabet, noise_level, rng):
    h = (hue + rng.normal(0.,noise_level,size=hue.shape)) % 1
    s = np.clip(saturation + rng.normal(0.,noise_level,size=saturation.shape),0,1)
    return quantize(encode(h,s,np.abs(alphabet).max()),alphabet)


def retrieval_metrics(final_indices, target_indices, alphabet, image_shape=None):
    a,b = alphabet[final_indices], alphabet[target_indices]
    rmax = np.abs(alphabet).max()
    ser = np.mean(final_indices != target_indices)
    mse = np.mean((rgb(a,rmax)-rgb(b,rmax))**2)
    visible = (np.abs(a)/rmax > 1e-10) & (np.abs(b)/rmax > 1e-10)
    angular = np.abs((np.angle(a)-np.angle(b))/(2*np.pi))
    angular = np.minimum(angular,1-angular)
    return dict(SER=float(ser), exact_recovery=float(ser==0),
                radial_error=float(np.mean(np.abs(np.abs(a)-np.abs(b)))/rmax),
                complex_similarity=float(abs(np.vdot(b,a))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),
                angular_error_cycles=float(angular[visible].mean()) if visible.any() else 0.,
                rgb_psnr=float(100. if mse<=1e-15 else -10*np.log10(mse)))


@dataclass
class Run:
    indices: np.ndarray
    history: list
    sweeps: int
    changed: int
    converged: bool
    audit: dict


def dynamics(alphabet, w, initial, schedule, rule, max_sweeps=80, relative=False,
             refresh=True, check_every_step=True):
    """Asynchronous descent with independent row-dot audits at each change.

    Delta E is checked from the actual pre-update state and a fresh dot product,
    not from the cached field used by the decision. At sweep endpoints the sum
    of these changes is compared to a fresh full quadratic-form evaluation.
    """
    a = np.asarray(alphabet, dtype=complex)
    indices = np.array(initial,dtype=np.int64,copy=True)
    state = a[indices].copy()
    field = w@state
    if not np.allclose(w,w.conj().T,atol=1e-12,rtol=0) or w.diagonal().real.min() < 0:
        raise ValueError("The experiment requires Hermitian weights with nonnegative diagonal.")
    rmax = np.abs(a).max()
    history=[]
    def record(sweep):
        ec,er = energy(state,w)
        history.append(dict(sweep=sweep,attempted_updates=sweep*len(indices),E_c=ec,E_r=er,
                            rho=float(np.vdot(state,state).real/(len(state)*rmax*rmax)),
                            boundary_fraction=float(np.mean(np.abs(np.abs(state)-rmax)<=1e-10))))
        return ec if rule=="classic" else er
    baseline = record(0)
    predicted = baseline
    audit=dict(changed_updates=0,coordinate_energy_increases=0,
               nonnegative_energy_changes=0,max_coordinate_energy_increase=0.,
               max_field_residual=0.,max_energy_accumulation_residual=0.,
               terminal_rule_violations=0)
    converged=False
    for sweep, order in enumerate(schedule[:max_sweeps],1):
        changes=0
        for k in order:
            old=int(indices[k])
            new=decision(a,field[k],old,rule,relative=relative)
            if old==new:
                continue
            delta=a[new]-a[old]
            actual_field=np.dot(w[k],state) if check_every_step else field[k]
            de=-np.real(np.conj(delta)*actual_field)-.5*w[k,k].real*abs(delta)**2
            if rule=="voronoi":
                de += .5*(abs(a[new])**2-abs(a[old])**2)
            audit['coordinate_energy_increases'] += int(de > 1e-10)
            audit['nonnegative_energy_changes'] += int(de >= 0)
            audit['max_coordinate_energy_increase']=max(audit['max_coordinate_energy_increase'],float(de))
            predicted += de
            state[k]=a[new];indices[k]=new
            field += delta*w[:,k]
            changes+=1
        audit['changed_updates']+=changes
        exact_field=w@state
        audit['max_field_residual']=max(audit['max_field_residual'],float(np.abs(field-exact_field).max()))
        if refresh:
            field=exact_field
        current_energy=record(sweep)
        audit['max_energy_accumulation_residual']=max(audit['max_energy_accumulation_residual'],float(abs(current_energy-predicted)))
        if changes==0:
            converged=True
            break
    if converged:
        audit['terminal_rule_violations']=sum(decision(a,exact_field[k],int(indices[k]),rule,relative=relative)!=indices[k] for k in range(len(state)))
    if audit['coordinate_energy_increases'] or audit['terminal_rule_violations']:
        raise AssertionError(audit)
    return Run(indices,history,sweep,audit['changed_updates'],converged,audit)
