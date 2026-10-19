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
    return [Line2D([],[],ls='none',marker='o',ms=3.5,color=INK,label=r'$s\in\mathcal{S}$'),
            Line2D([],[],ls='none',marker='o',ms=6,mfc='none',mec=PURPLE,label=r'$\partial_{\max}\mathcal{S}$'),
            Patch(fc='#CDE8E3',ec=TEAL,label=r'$V(s)$'),
            Line2D([],[],color=INK,lw=.8,label='Cell boundary'),
            Line2D([],[],color=ORANGE,ls='--',lw=1,label=r'$\ell_{s,s^{\prime}}$'),
            Line2D([],[],color=PURPLE,ls=':',lw=1,label=r'$|z|=R_{\max}$')]


def geometry(ax, z, title, pair, compact=False, labels=None):
    z = np.asarray(z); i,j = pair
    polygons = [cell_polygon(z,k) for k in range(len(z))]
    for k,poly in enumerate(polygons):
        ax.add_patch(Polygon(poly,fc='#F4F7FA' if k!=i else '#CDE8E3',
                             ec='#677784',lw=.55,zorder=1))
    ax.add_patch(Polygon(polygons[i],fc='none',ec=TEAL,lw=1.2,zorder=3))
    # The full mediatrix and its active Voronoi face are different objects.
    line = bisector_segment(z,i,j)
    face = bisector_segment(z,i,j,active=True)
    if face is None:
        raise ValueError('The annotated symbols must share a nontrivial face.')
    ax.plot(*line.T,color=ORANGE,ls=(0,(4,3)),lw=.95,zorder=4)
    ax.plot(*face.T,color=BLUE,lw=1.5,zorder=5)
    rmax = np.abs(z).max()
    ax.add_patch(Circle((0,0),rmax,fill=False,ec=PURPLE,ls=':',lw=.9,zorder=2))
    outer = np.isclose(np.abs(z),rmax,rtol=0,atol=1e-12)
    ax.scatter(z.real,z.imag,s=10 if compact else 20,color=INK,ec='white',lw=.35,zorder=7)
    ax.scatter(z.real[outer],z.imag[outer],s=34 if compact else 70,
               facecolors='none',edgecolors=PURPLE,lw=.9,zorder=6)
    offset = 5 if compact else 8
    for index,label,key in [(i,r'$s$','s'),(j,r'$s^{\prime}$','neighbor')]:
        ax.annotate(label,(z[index].real,z[index].imag),xytext=labels[key] if labels else (offset,offset),
                    textcoords='data' if labels else 'offset points',fontsize=7.5 if compact else 9,
                    arrowprops=dict(arrowstyle='-',lw=.5,color=INK) if compact else None,
                    bbox=dict(fc='white',ec='none',pad=.3,alpha=.92),zorder=9)
    poly = polygons[i]
    # A convex combination of vertices lies in the cell; choose the location
    # farthest from generators to leave a readable label in narrow PSK sectors.
    candidates = .8*poly + .2*poly.mean(axis=0)
    candidate = labels['cell'] if labels else (-.36,-.08)
    ax.text(*candidate,r'$V(s)$',ha='center',va='center',color=TEAL,
            fontsize=7.5 if compact else 10,zorder=10,
            bbox=dict(fc='white',ec='none',pad=.4,alpha=.9))
    ax.axhline(0,color='#9AA6AF',lw=.45,zorder=0)
    ax.axvline(0,color='#9AA6AF',lw=.45,zorder=0)
    ax.set(xlim=(-1.22,1.22),ylim=(-1.22,1.22),aspect='equal',
           xlabel=r'$\mathrm{Re}(z)$',ylabel=r'$\mathrm{Im}(z)$',title=title)
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.grid(False)
    ax.tick_params(length=3,width=.65,labelsize=7.5 if compact else 9,pad=2)
    ax.set_xlabel(r'$\mathrm{Re}(z)$',fontsize=8 if compact else 10,labelpad=3)
    ax.set_ylabel(r'$\mathrm{Im}(z)$',fontsize=8 if compact else 10,labelpad=3)
    ax.set_title(title,fontsize=9 if compact else 11,pad=7,weight='bold')
    for name in ('left','bottom'):
        ax.spines[name].set_color(INK);ax.spines[name].set_linewidth(.8)
    if not compact:
        p,q = z[i],z[j]
        ax.plot([p.real,q.real],[p.imag,q.imag],color=ORANGE,lw=.8,ls=':',zorder=6)
        mid=(p+q)/2; u=(q-p)/abs(q-p); v=1j*u
        corner=np.array([mid+.06*u,mid+.06*u+.06*v,mid+.06*v])
        ax.plot(corner.real,corner.imag,color=ORANGE,lw=.85,zorder=8)
        ax.annotate(r'$\ell_{s,s^{\prime}}$',line[1],xytext=(9,-17),textcoords='offset points',
                    color=ORANGE,fontsize=10,zorder=10,
                    bbox=dict(fc='white',ec='none',pad=.6))
        ax.plot(0,0,marker='+',ms=5,color=PURPLE,zorder=8)
        angle=np.deg2rad(250)
        end=rmax*np.exp(1j*angle)
        ax.annotate('',(end.real,end.imag),(0,0),
                    arrowprops=dict(arrowstyle='<->',lw=.75,color=PURPLE),zorder=3)
        ax.text(-.34,-.58,r'$R_{\max}$',fontsize=9,color=PURPLE,
                rotation=70,ha='center',va='center',zorder=8,
                bbox=dict(fc='white',ec='none',pad=.5))
    return polygons, face


def render(out):
    from render_figures import style, save, sat
    out=Path(out);(out/'source_data').mkdir(parents=True,exist_ok=True)
    style()
    z=illustrative_alphabet()
    pd.DataFrame({'real':z.real,'imag':z.imag}).to_csv(out/'source_data/irregular_symbols.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.5,4.0))
    fig.subplots_adjust(left=.16,right=.97,bottom=.23,top=.98)
    geometry(ax,z,'',(0,1))
    fig.legend(handles=legend_handles(),loc='lower center',ncol=3,fontsize=6.9,
               frameon=False,handlelength=1.8,columnspacing=1.1,bbox_to_anchor=(.53,.005))
    save(fig,'generic_voronoi',out)
    alphabets=sat.build_alphabets()
    fig,axes=plt.subplots(1,4,figsize=(7.4,2.5))
    fig.subplots_adjust(left=.047,right=.997,bottom=.30,top=.87,wspace=.39)
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
    fig.legend(handles=legend_handles(),loc='lower center',ncol=6,fontsize=7,
               frameon=False,handlelength=1.7,columnspacing=1.6,bbox_to_anchor=(.52,.015))
    save(fig,'constellations',out)
    pd.DataFrame(records).to_csv(out/'source_data/geometry_alphabets.csv',index=False)
    print('Geometry: 12 illustrative symbols; experimental alphabets unchanged; 1 x 4 panels.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
    render(p.parse_args().output)
