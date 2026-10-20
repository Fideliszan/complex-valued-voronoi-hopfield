"""Plot all four cardinalities without averaging distinct alphabet families."""
from pathlib import Path
import argparse
import pandas as pd
from plotnine import (ggplot,aes,geom_line,geom_point,geom_linerange,facet_grid,
    scale_x_log10,scale_y_log10,scale_color_manual,scale_shape_manual,
    scale_linetype_manual,theme_minimal,theme,element_text,element_blank,element_line,labs)

ROOT=Path(__file__).resolve().parents[1]
PALETTE={'Distance':'#2864A0','Inertial score':'#CB701B','KD-tree':'#16877A','Structured':'#8F6593'}


def render(data,out):
    from figure_typography import configure
    configure()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=Path(data)
    raw=pd.read_csv(data/'experiment3_raw_timings.csv')
    # Named aggregation avoids relying on pandas' multi-index display behavior.
    d=raw.groupby(['ensemble','family','cardinality','method'],as_index=False).agg(
        median=('time_per_query_us','median'),lower=('time_per_query_us','min'),upper=('time_per_query_us','max'))
    d['Method']=pd.Categorical(d.method.map({'distance':'Distance','inertial_score':'Inertial score','kdtree':'KD-tree','structured':'Structured'}),categories=list(PALETTE),ordered=True)
    d['family']=pd.Categorical(d.family,categories=['PSK','QAM','Polar','Hexagonal'],ordered=True)
    d['Inputs']=pd.Categorical(d.ensemble.map({'uniform':'Uniform fields','local_fields':'Network fields'}),categories=['Uniform fields','Network fields'],ordered=True)
    (out/'source_data').mkdir(exist_ok=True)
    d.to_csv(out/'source_data/activation_scaling.csv',index=False)
    plot=(ggplot(d,aes('cardinality','median',color='Method',shape='Method',linetype='Method',group='Method'))+
        geom_linerange(aes(ymin='lower',ymax='upper'),size=.55,alpha=.85,linetype='solid')+
        geom_line(size=.75)+geom_point(size=2.4,stroke=.6,fill='white')+
        facet_grid('Inputs ~ family')+
        scale_x_log10(breaks=[16,64,256,1024],labels=['16','64','256','1024'],expand=(.13,0))+
        scale_y_log10(breaks=[.3,1,3,10,30],labels=lambda xs:[f'{x:g}' for x in xs])+scale_color_manual(values=PALETTE)+
        scale_shape_manual(values={'Distance':'o','Inertial score':'s','KD-tree':'^','Structured':'D'})+
        scale_linetype_manual(values={'Distance':'solid','Inertial score':'dashed','KD-tree':'dashdot','Structured':'dotted'})+
        theme_minimal(base_size=10)+theme(figure_size=(7.4,4.9),legend_position='bottom',
            text=element_text(family='Termes Figure',color='#263440'),axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=8),axis_line=element_line(color='#52606C',size=.5),
            axis_ticks=element_line(color='#52606C',size=.4),axis_ticks_minor=element_blank(),
            panel_spacing_x=.04,panel_spacing_y=.055,panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color='#E6EAF0',size=.4),
            strip_text=element_text(weight='normal',size=9),legend_title=element_blank())+
        labs(x='Alphabet cardinality |S|',y='Microseconds per query (batch average)',color='',shape='',linetype=''))
    for ext in ('pdf','png'):
        plot.save(out/f'activation_scaling.{ext}',dpi=600,verbose=False)
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/paper');p.add_argument('--output',type=Path,default=ROOT/'figures')
    args=p.parse_args();render(args.data,args.output)
