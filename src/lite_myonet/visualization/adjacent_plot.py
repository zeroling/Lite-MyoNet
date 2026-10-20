"""Adjacent bone-edge plotting; no embedded data or results."""
from pathlib import Path
import json,csv,math
import numpy as np
MODELS=("teacher","logit_kd","full_st_gtp")
TITLES=("Teacher","Logit KD only","Full ST-GTP")
FRAME=60

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def table(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def analyze(out,result):
    matrices=np.load(out/'data/affinity_matrices.npz')
    edge_array=np.array(result['bones']); child,parent=edge_array.T
    weights={}
    for model in MODELS:
        for typ in ('static','dynamic'):
            p=matrices[f'{model}_{typ}'].astype(np.float64)
            weights[f'{model}_{typ}']=(p[:,child,parent]+p[:,parent,child])*.5
    np.savez_compressed(out/'data/adjacent_edge_weights.npz',edges=edge_array,**weights)
    window_rows=[]; ranking=[]
    for w in result['windows']:
        i=w['window_index']
        row=dict(patient=w['patient'],window_index=i,main_window=w['main_window'])
        for typ in ('static','dynamic'):
            teacher=weights[f'teacher_{typ}'][i]
            for model in MODELS[1:]:
                delta=weights[f'{model}_{typ}'][i]-teacher
                row[f'{typ}_{model}_rmse']=float(np.sqrt(np.mean(delta**2)))
                row[f'{typ}_{model}_mae']=float(np.mean(np.abs(delta)))
        window_rows.append(row)
    for patient in sorted({w['patient'] for w in result['windows']}):
        rows=[r for r in window_rows if r['patient']==patient]
        first=next(r for r in rows if r['main_window'])
        w=result['windows'][first['window_index']]
        row=dict(patient=patient,diagnosis=result['diagnoses'][str(patient)],
                 window_index=first['window_index'],window_count=len(rows),
                 source_file=w['source_file'],display_frame_zero_based=FRAME)
        for metric in [k for k in first if k.endswith(('_rmse','_mae'))]:
            row[metric]=first[metric]
            row[metric+'_all_window_mean']=float(np.mean([r[metric] for r in rows]))
        ranking.append(row)
    ranking.sort(key=lambda r:(r['static_full_st_gtp_rmse'],r['patient']))
    for rank,row in enumerate(ranking,1):
        row['rank']=rank; row['selected']=rank<=3
    selected=[r['patient'] for r in ranking[:3]]
    edge_rows=[]
    for w in result['windows']:
        i=w['window_index']
        for typ in ('static','dynamic'):
            for model in MODELS:
                for e,(c,p) in enumerate(result['bones']):
                    edge_rows.append(dict(patient=w['patient'],window_index=i,main_window=w['main_window'],
                                          topology=typ,model=model,child_node=c,parent_node=p,
                                          undirected_affinity=float(weights[f'{model}_{typ}'][i,e])))
    result.update(ranking=ranking,selected_patients=selected,window_edge_differences=window_rows,
                  selection=dict(scope='first valid sorted 120-frame window, fixed frame 60',
                                 criterion='ascending static_full_st_gtp_rmse over 22 actual skeletal connections; tie: patient ID',
                                 rmse='sqrt(mean_e((w_full[e]-w_teacher[e])**2)), 22 undirected weights',
                                 edge_weight='(P[i,j]+P[j,i])/2',
                                 edge_renormalization=False,
                                 teacher_display='0.5*P_spatial+0.5*P_temporal, then select and symmetrize actual edges',
                                 selected_after_observing_results=True))
    table(out/'patient_ranking.csv',ranking)
    table(out/'window_edge_differences.csv',window_rows)
    table(out/'adjacent_edge_weights.csv',edge_rows)
    write(out/'results.json',result)
    print('Selected patients:',selected,flush=True)
    return weights

def projected_coordinates(xyz,azimuth,elevation=12.):
    az,el=math.radians(azimuth),math.radians(elevation)
    basis=np.array([[-math.sin(az),-math.sin(el)*math.cos(az)],
                    [math.cos(az),-math.sin(el)*math.sin(az)],[0.,math.cos(el)]])
    return xyz@basis

def render(out,result,weights,selected_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors,cm,font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    font=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':11,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    xyz_all=np.load(out/'data/display_coordinates.npz')['coordinates']
    first={w['patient']:w['window_index'] for w in result['windows'] if w['main_window']}
    ranked={r['patient']:r for r in result['ranking']}
    cmap=colors.LinearSegmentedColormap.from_list('adjacent_affinity',['#24456B','#25878E','#D6AC4E'])
    styles={}
    edge_array=np.array(result['bones'])
    for patient in sorted({w['patient'] for w in result['windows']}):
        wi=first[patient]; xyz=xyz_all[wi]
        hip=xyz[6,:2]-xyz[1,:2]
        azimuth=math.degrees(math.atan2(-float(hip[0]),float(hip[1])))+15.
        values=np.stack([weights[f'{m}_static'][wi] for m in MODELS])
        styles[str(patient)]=dict(camera_azimuth_degrees=azimuth,elevation_degrees=12.,
                                  vmin=float(values.min()),vmax=float(values.max()))

    def figure(patients):
        n=len(patients)
        fig=plt.figure(figsize=(7.1,2.8*n+.35))
        top=.94; bottom=.115 if n==1 else .065; pitch=(top-bottom)/n
        for row,patient in enumerate(patients):
            wi=first[patient]; style=styles[str(patient)]
            xy=projected_coordinates(xyz_all[wi],style['camera_azimuth_degrees'])
            centre=(xy.min(0)+xy.max(0))*.5; height=float(np.ptp(xy[:,1]))
            xhalf=max(.32*height,.65*float(np.ptp(xy[:,0])))
            norm=colors.Normalize(vmin=style['vmin'],vmax=style['vmax'])
            y0=top-(row+1)*pitch; h=pitch*.78
            for col,(model,title) in enumerate(zip(MODELS,TITLES)):
                ax=fig.add_axes([.035+.303*col,y0,.287,h]); ax.set_aspect('equal'); ax.axis('off')
                ax.set_xlim(centre[0]-xhalf,centre[0]+xhalf)
                ax.set_ylim(xy[:,1].min()-.055*height,xy[:,1].max()+.045*height)
                strength=weights[f'{model}_static'][wi]
                segments=xy[edge_array]
                collection=LineCollection(segments,colors=cmap(norm(strength)),
                                          linewidths=2.05+.95*norm(strength),capstyle='round',joinstyle='round',zorder=2)
                ax.add_collection(collection)
                ax.scatter(xy[:,0],xy[:,1],s=10,facecolors='white',edgecolors='#415261',linewidths=.55,zorder=3)
                for node,label,offset,ha in ((3,'L',(-7,-3),'right'),(8,'R',(7,-3),'left')):
                    ax.annotate(label,xy[node],xytext=offset,textcoords='offset points',ha=ha,va='center',
                                fontsize=8.5,color='#3b4652')
                if row==0: ax.set_title(title,y=1.18,pad=0,fontweight='bold')
                text='Teacher reference' if model=='teacher' else f"Edge RMSE = {ranked[patient]['static_'+model+'_rmse']:.5f}"
                ax.text(.5,-.055,text,transform=ax.transAxes,ha='center',va='top',fontsize=9,color='#303c48')
            heading=f"Patient {patient}  |  {result['diagnoses'][str(patient)]}" if n>1 else f'Patient {patient}  |  Window 1, frame 61'
            fig.text(.035,y0+h+.022,heading,fontsize=10,fontweight='bold',color='#293846')
            cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.956,y0+h*.10,.012,h*.81]))
            cb.solids.set_rasterized(False)
            cb.locator=MaxNLocator(nbins=4); cb.formatter=FormatStrFormatter('%.3f'); cb.update_ticks()
            cb.ax.tick_params(labelsize=8,length=2,width=.5); cb.outline.set_linewidth(.5)
            cb.set_label('Adjacent-joint affinity',fontsize=9,labelpad=5)
        fig.text(.5,.015,'Static feature affinity on 22 skeletal connections  |  L/R: anatomical sides',ha='center',fontsize=8.8,color='#384654')
        return fig

    fig=figure(result['selected_patients'])
    for suffix in ('pdf','png','svg'):
        options=dict(bbox_inches='tight',pad_inches=.045)
        if suffix=='png':options['dpi']=600
        fig.savefig(out/f'selected_three_adjacent.{suffix}',**options)
    plt.close(fig)
    if not selected_only:
        with PdfPages(out/'all_patients_adjacent.pdf') as pdf:
            for patient in sorted({w['patient'] for w in result['windows']}):
                fig=figure([patient]); pdf.savefig(fig,bbox_inches='tight',pad_inches=.045)
                fig.savefig(out/f'patients/patient_{patient:02d}.png',dpi=600,bbox_inches='tight',pad_inches=.045)
                plt.close(fig)
                if patient%10==0:print(f'Drawn patient {patient}',flush=True)
    parameters=dict(font='Times New Roman',font_file=font,png_dpi=600,
                    connections=result['bones'],display_frame_zero_based=FRAME,projection='orthographic',
                    camera_rule='hip lateral axis + 15 degrees oblique; elevation 12 degrees',
                    color_anchors=['#24456B','#25878E','#D6AC4E'],
                    color_semantics='absolute static feature affinity; low blue, high gold',
                    line_width_mapping_points=[2.05,3.],node_style='white interior, slate outline',
                    node_labels='anatomical L/R only',
                    shared_scale='min/max of 22 edge weights across all three models within each case; no per-model scaling',
                    per_patient=styles)
    write(out/'plot_parameters.json',parameters)
