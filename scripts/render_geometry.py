"""Exact clipped Voronoi cells with definition-linked geometric annotations.

The illustrative alphabet is deliberately sparse. Experimental alphabets are
imported without modification from the saturation protocol.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D

INK = '#263440'
BLUE, TEAL, PURPLE, ORANGE = '#2864A0', '#16877A', '#8F6593', '#CB701B'


def illustrative_alphabet():
    outer = np.exp(1j * np.deg2rad([0, 65, 130, 195, 250, 305]))
    inner = np.array([-.28+.15j, .30+.18j, .05+.60j,
                      -.50-.35j, .10-.40j, .55-.40j])
    return np.r_[inner, outer]


def cell_polygon(z, index, limit=1.22):
    from render_figures import clip
    p = np.array([z[index].real, z[index].imag])
    poly = np.array([[-limit,-limit],[limit,-limit],
                     [limit,limit],[-limit,limit]], dtype=float)
    for j, value in enumerate(z):
        if index == j or not len(poly):
            continue
        q = np.array([value.real, value.imag])
        poly = clip(poly, 2*(q-p), q@q-p@p)
    return poly


def bisector_segment(z, i, j, limit=1.22, active=False):
    """Intersect a bisector with the window, optionally with all cell halfspaces."""
    p = np.array([z[i].real, z[i].imag])
    q = np.array([z[j].real, z[j].imag])
    mid = (p+q)/2
    normal = q-p
    direction = np.array([-normal[1],normal[0]])/np.linalg.norm(normal)
    constraints = [(np.array(a),limit) for a in [(1,0),(-1,0),(0,1),(0,-1)]]
    if active:
        for k, value in enumerate(z):
            if k != i:
                r = np.array([value.real,value.imag])
                constraints.append((2*(r-p),r@r-p@p))
    lo, hi = -np.inf, np.inf
    for a,b in constraints:
        slope, rhs = a@direction, b-a@mid
        if abs(slope) < 1e-12:
            if rhs < -1e-10:
                return None
        elif slope > 0:
            hi = min(hi,rhs/slope)
        else:
            lo = max(lo,rhs/slope)
    if lo > hi-1e-10:
        return None
    return np.array([mid+lo*direction,mid+hi*direction])


def legend_handles():
    return [Line2D([], [], ls='none', marker='o', ms=4, color=INK,
                   label=r'Symbol $s\in\mathcal{S}$'),
            Line2D([], [], ls='none', marker='o', ms=6, mfc='none', mec=PURPLE,
                   label=r'Maximum modulus $\partial_{\max}\mathcal{S}$'),
            Line2D([], [], color='#657481', lw=.7, label='Voronoi boundary'),
            Line2D([], [], color=PURPLE, ls=(0, (1.2, 2.5)), lw=1,
                   label=r'$|z|=R_{\max}$')]


def geometry(ax, z, title, pair, compact=False, labels=None):
    """Exact cells with no overlaid boxes, construction lines, or selected fill."""
    z = np.asarray(z)
    polygons = [cell_polygon(z, k) for k in range(len(z))]
    # Greedy coloring of shared-face adjacency; colors carry no numerical value.
    fills = ['#E7EFF7', '#F5EADF', '#EAE5F0', '#E5F0ED', '#F1EFE3']
    colors = []
    for k, poly in enumerate(polygons):
        neighbors = {colors[j] for j in range(k)
                     if bisector_segment(z, k, j, active=True) is not None}
        color = next((c for c in range(len(fills)) if c not in neighbors), k % len(fills))
        colors.append(color)
        ax.add_patch(Polygon(poly, facecolor=fills[color], edgecolor='#657481',
                             linewidth=.55 if compact else .7, zorder=1))
    rmax = np.abs(z).max()
    ax.add_patch(Circle((0, 0), rmax, fill=False, edgecolor=PURPLE,
                       linestyle=(0, (1.2, 2.5)), linewidth=.9, zorder=3))
    outer = np.isclose(np.abs(z), rmax, rtol=0, atol=1e-12)
    ax.scatter(z.real, z.imag, s=10 if compact else 23, color=INK,
               linewidths=0, zorder=5)
    ax.scatter(z.real[outer], z.imag[outer], s=39 if compact else 80,
               facecolors='none', edgecolors=PURPLE, linewidths=.9, zorder=4)
    # All mathematical labels that describe sets live in the shared legend.
    # Only the larger illustrative panel labels a generator and its cell.
    if not compact:
        ax.annotate(r'$s$', (z[0].real, z[0].imag), xytext=(7, -12),
                    textcoords='offset points', color=INK, fontsize=10, zorder=6)
        ax.text(-.39, .34, r'$V(s)$', fontsize=10, ha='center', color=INK, zorder=6)
    ax.axhline(0, color='#AEB8C0', lw=.4, alpha=.65, zorder=2)
    ax.axvline(0, color='#AEB8C0', lw=.4, alpha=.65, zorder=2)
    ax.set(xlim=(-1.22, 1.22), ylim=(-1.22, 1.22), aspect='equal')
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.grid(False)
    ax.tick_params(length=3, width=.6, labelsize=8 if compact else 9, pad=3)
    ax.set_xlabel(r'$\mathrm{Re}(z)$', fontsize=9 if compact else 10, labelpad=3)
    ax.set_ylabel(r'$\mathrm{Im}(z)$', fontsize=9 if compact else 10, labelpad=3)
    ax.set_title(title, fontsize=10 if compact else 11, pad=9, weight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#9BA6AF'); spine.set_linewidth(.6)
    return polygons, bisector_segment(z, *pair, active=True)


def render(out):
    from render_figures import style, save, sat
    out=Path(out);(out/'source_data').mkdir(parents=True,exist_ok=True)
    style()
    z=illustrative_alphabet()
    pd.DataFrame({'real':z.real,'imag':z.imag}).to_csv(out/'source_data/irregular_symbols.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.65,4.05))
    fig.subplots_adjust(left=.15,right=.97,bottom=.22,top=.98)
    geometry(ax,z,'',(0,1))
    fig.legend(handles=legend_handles(),loc='lower center',ncol=2,fontsize=7.4,
               frameon=False,handlelength=1.8,columnspacing=1.1,bbox_to_anchor=(.54,.01))
    save(fig,'generic_voronoi',out)
    alphabets=sat.build_alphabets()
    fig,axes=plt.subplots(1,4,figsize=(7.4,2.65))
    fig.subplots_adjust(left=.047,right=.997,bottom=.32,top=.87,wspace=.40)
    records=[]
    for panel,(ax,(name,values)) in enumerate(zip(axes,alphabets.items())):
        # Choose an upper-right cell, then its closest adjacent generator.
        target={'16-PSK':.71+.71j,'16-QAM':.71+.71j,
                '16-Polar':.71+.71j,'19-Hexagonal':.5+.87j}[name]
        i=int(np.argmin(abs(values-target)))
        neighbors=[k for k in np.argsort(abs(values-values[i])) if k!=i
                   and bisector_segment(values,i,int(k),active=True) is not None]
        neighbor_target={'16-PSK':.924+.383j,'16-QAM':.235+.707j,
                         '16-Polar':.53+.53j,'19-Hexagonal':0+.866j}[name]
        j=int(min(neighbors,key=lambda k:abs(values[k]-neighbor_target)))
        label_positions={
            '16-PSK':dict(s=(.40,.80),neighbor=(1.01,.24),cell=(.89,1.08)),
            '16-QAM':dict(s=(.78,.67),neighbor=(.08,.80),cell=(.95,1.08)),
            '16-Polar':dict(s=(.83,.68),neighbor=(.28,.47),cell=(.95,1.08)),
            '19-Hexagonal':dict(s=(.62,.80),neighbor=(-.18,.75),cell=(.74,1.10)),
        }[name]
        geometry(ax,values,f'({chr(97+panel)}) {name}',(i,j),compact=True,labels=label_positions)
        for k,value in enumerate(values):
            records.append(dict(alphabet=name,index=k,real=value.real,imag=value.imag,
                                selected=k==i,neighbor=k==j))
    fig.legend(handles=legend_handles(),loc='lower center',ncol=4,fontsize=7.5,
               frameon=False,handlelength=1.7,columnspacing=1.6,bbox_to_anchor=(.52,.015))
    save(fig,'constellations',out)
    pd.DataFrame(records).to_csv(out/'source_data/geometry_alphabets.csv',index=False)
    print('Geometry: 12 illustrative symbols; experimental alphabets unchanged; 1 x 4 panels.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
    render(p.parse_args().output)
