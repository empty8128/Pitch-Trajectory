import numpy as np
import plotly.graph_objects as go
from pybaseball import statcast_pitcher
from pybaseball import statcast_batter
from pybaseball import playerid_lookup
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from pybaseball import pitching_stats
import pandas as pd
import math
from datetime import datetime

Game_Type = 'R'
Per=0.001
g_acceleretion=-32.17405

def frange(start, end, step):
    list = [start]
    n = start
    while n + step < end:
        n = n + step
        list.append(n)
    return list

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
server = app.server

y=[]

for t in range(2015,datetime.now().year+1,1):
    y.append(t)

app.layout = html.Div(
    [
        html.Div([
            html.Div([
                html.H1(children='Player0', id='h0'),
                html.P(children='', id='account0'),
                ],
                style={'width': '100%','height': '100%','display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'flex-end'}
            ),
            html.Div(dcc.Dropdown(
                id='Year0',
                options=[{'label':x, 'value':x} for x in y],
                placeholder='Year',
                style={'width': '100%'}
            )),
            html.Div(dcc.Dropdown(
                id='Name0',
                placeholder="Please select a year",
                style={'width': '100%'}
            )),
            html.Div(dcc.Dropdown(
                id='PitchCount0',
                placeholder="Please select a year",
                style={'width': '100%'}
            )),
            html.Div([
                html.H1(children='Player1', id='h1'),
                html.P(children='', id='account1'),
                ],
                style={'width': '100%','height': '100%','display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'flex-end'}
            ),
            html.Div(dcc.Dropdown(
                id='Year1',
                options=[{'label':x, 'value':x} for x in y],
                placeholder='Year',
                style={'width': '100%'}
            )),
            html.Div(dcc.Dropdown(
                id='Name1',
                placeholder="Please select a year",
                style={'width': '100%'}
            )),
            html.Div(dcc.Dropdown(
                id='PitchCount1',
                placeholder="Please select a year",
                style={'width': '100%'}
            ))
        ]),
        html.Div([
            dcc.Graph(
                id='Chart0',
                style={'width': '75%','height': '900px'}),
            html.Div([
                dcc.Graph(
                    id='Chart1',
                    style={'width': '100%','height': '50%'}),
                dcc.Graph(
                    id='Chart2',
                    style={'width': '100%','height': '50%'}),
                ],
                style={'width': '25%','height': '100%'}
            ),
            ],
            style={'width': '100%','height': '100%','display': 'flex', 'flex-wrap': 'wrap'}
        )
    ]
)

###################################################################年

@app.callback(
    Output('Name0', 'options'),
    Output('Name0', 'value'),
    Input('Year0', 'value'),
)
def first(p):
    if p is None:
        k=''
        l = None
    else:
        k=[{'label': 'Name', 'value': 'Name', 'disabled':True}]
        k.extend([{'label':x, 'value':x} for x in sorted(pitching_stats(p, qual=1)['Name'].unique())])
        l='Name'
    return k, l

@app.callback(
    Output('Name0', 'placeholder'),
    Input('Year0','value'),
)
def first(p):
    if p is None:
        k='Please select a year'
    else:
        k='Gathering Data...'
    return k

@app.callback(
    Output('Name1', 'options'),
    Output('Name1', 'value'),
    Input('Year1', 'value')
)
def first(p):
    if p is None:
        k=''
        l = None
    else:
        k=[{'label': 'Name', 'value': 'Name', 'disabled':True}]
        k.extend([{'label':x, 'value':x} for x in sorted(pitching_stats(p, qual=1)['Name'].unique())])
        l='Name'
    return k, l

@app.callback(
    Output('Name1', 'placeholder'),
    Input('Year1', 'value'),
)
def first(p):
    if p is None:
        k='Please select a year'
    else:
        k='Gathering Data...'
    return k

##################################################################ピッチ

@app.callback(
    Output('PitchCount0', 'options'),
    Output('PitchCount0', 'value'),
    Input('Year0', 'value'),
    Input('Name0', 'value')
)
def first(p,q):
    if p is None:
        k = ''
        l = None
    elif q is None or q == 'Name':
        k = ''
        l = None
    else:
        global ef
        ef = pd.DataFrame()
        ef_0 = statcast_pitcher(str(p)+'-01-01', str(p)+'-12-31', playerid_lookup(q.split()[1], q.split()[0], fuzzy=True).iloc[0,2])
        if Game_Type == 'R':
            ef_1 = ef_0[ef_0['game_type']== 'R']
        elif Game_Type == 'P':
            ef_1 = ef_0[ef_0['game_type'].isin(['F', 'D', 'L', 'W'])]
        global length
        length = ef_1.shape[0]
        c=[]
        for t in range(length,0,-1):
            c.append(t)
        ef = ef_1.assign(n=c)
        if 'ef1' in globals():
            pass
        else:
            global vx0_n
            vx0_n = ef.columns.get_loc('vx0')
            global vy0_n
            vy0_n = ef.columns.get_loc('vy0')
            global vz0_n
            vz0_n = ef.columns.get_loc('vz0')
            global ax_n
            ax_n = ef.columns.get_loc('ax')
            global ay_n
            ay_n = ef.columns.get_loc('ay')
            global sz_top_n
            sz_top_n = ef.columns.get_loc('sz_top')
            global sz_bot_n
            sz_bot_n = ef.columns.get_loc('sz_bot')
            global az_n
            az_n = ef.columns.get_loc('az')
            global release_pos_y_n
            release_pos_y_n = ef.columns.get_loc('release_pos_y')
            global pitch_name_n
            pitch_name_n = ef.columns.get_loc('pitch_name')
        k=[{'label': 'Pitch Count', 'value': 'Pitch Count', 'disabled':True}]
        k.extend(reversed([{'label': str(x)+','+str(ef.iloc[length-x,0])+','+str(ef.iloc[length-x,1])+',Inning:'+str(ef.iloc[length-x,35])+',Ball-Strike-Out:'+str(ef.iloc[length-x,24])+'-'+str(ef.iloc[length-x,25])+'-'+str(ef.iloc[length-x,34])+','+str(ef.iloc[length-x,5])+','+str(ef.iloc[length-x,2])+'(mph)', 'value':x} for x in ef['n']]))
        l = 'Pitch Count'
    return k, l

@app.callback(
    Output('PitchCount0', 'placeholder'),
    Input('Year0', 'value'),
    Input('Name0', 'value'),
)
def first(p,q):
    if p is None:
        k='Please select a year'
    elif q is None or q == 'Name':
        k='Please select player'
    else:
        k='Gathering Data...'
    return k

@app.callback(
    Output('PitchCount1', 'options'),
    Output('PitchCount1', 'value'),
    State('Year1', 'value'),
    Input('Name1', 'value')
)
def first(p,q):
    if p is None:
        k = ''
        l = None
    elif q is None or q == 'Name':
        k = ''
        l = None
    else:
        global ef1
        ef1 = pd.DataFrame()
        ef1_0 = statcast_pitcher(str(p)+'-01-01', str(p)+'-12-31', playerid_lookup(q.split()[1], q.split()[0], fuzzy=True).iloc[0,2])
        if Game_Type == 'R':
            ef1_1 = ef1_0[ef1_0['game_type']== 'R']
        elif Game_Type == 'P':
            ef1_1 = ef1_0[ef1_0['game_type'].isin(['F', 'D', 'L', 'W'])]
        global d1
        d1=ef1_1.shape[0]
        c=[]
        for t in range(d1,0,-1):
            c.append(t)
        ef1 = ef1_1.assign(n=c)
        if 'ef' in globals():
            pass
        else:
            global vx0_n
            vx0_n = ef1.columns.get_loc('vx0')
            global vy0_n
            vy0_n = ef1.columns.get_loc('vy0')
            global vz0_n
            vz0_n = ef1.columns.get_loc('vz0')
            global ax_n
            ax_n = ef1.columns.get_loc('ax')
            global ay_n
            ay_n = ef1.columns.get_loc('ay')
            global sz_top_n
            sz_top_n = ef1.columns.get_loc('sz_top')
            global sz_bot_n
            sz_bot_n = ef1.columns.get_loc('sz_bot')
            global az_n
            az_n = ef1.columns.get_loc('az')
            global release_pos_y_n
            release_pos_y_n = ef1.columns.get_loc('release_pos_y')
            global pitch_name_n
            pitch_name_n = ef1.columns.get_loc('pitch_name')
        k=[{'label': 'Pitch Count', 'value': 'Pitch Count', 'disabled':True}]
        k.extend(reversed([{'label': str(x)+','+str(ef1.iloc[d1-x,0])+','+str(ef1.iloc[d1-x,1])+',Inning:'+str(ef1.iloc[d1-x,35])+',Ball-Strike-Out:'+str(ef1.iloc[d1-x,24])+'-'+str(ef1.iloc[d1-x,25])+'-'+str(ef1.iloc[d1-x,34])+','+str(ef1.iloc[d1-x,5])+','+str(ef1.iloc[d1-x,2])+'(mph)', 'value':x} for x in ef1['n']]))
        l = 'Pitch Count'
    return k, l

@app.callback(
    Output('PitchCount1', 'placeholder'),
    Input('Year1', 'value'),
    Input('Name1', 'value'),
)
def first(p,q):
    if p is None:
        k='Please select a year'
    elif q is None or q == 'Name':
        k='Please select player'
    else:
        k='Gathering Data...'
    return k

###################################################################説明

@app.callback(
    Output('account0', 'children'),
    Input('PitchCount0', 'value'),
)
def first(p):
    if p is None or p == 'Pitch Count':
        k = None
    else:
        vy_f0 = -np.sqrt(ef.iloc[length-p,vy0_n]**2-(2*ef.iloc[length-p,ay_n]*(50-17/12)))
        t0 = (vy_f0-ef.iloc[length-p,vy0_n])/ef.iloc[length-p,ay_n]
        vz_f0 = ef.iloc[length-p,vz0_n]+ef.iloc[length-p,az_n]*t0
        vaa0 = (np.arctan(vz_f0/vy_f0))*(180/(math.pi))
        k = 'Piych Info--Pitch Name:'+str(ef.iloc[length-p,pitch_name_n])+',Description:'+str(ef.iloc[length-p,9])+',VAA:'+str(vaa0)
    return k

@app.callback(
    Output('account1', 'children'),
    Input('PitchCount1', 'value'),
)
def first(p):
    if p is None or p == 'Pitch Count':
        k = None
    else:
        vy_f1 = -np.sqrt(ef.iloc[length-p,vy0_n]**2-(2*ef.iloc[length-p,ay_n]*(50-17/12)))
        t1 = (vy_f1-ef.iloc[length-p,vy0_n])/ef.iloc[length-p,ay_n]
        vz_f1 = ef.iloc[length-p,vz0_n]+ef.iloc[length-p,az_n]*t1
        vaa1 = (np.arctan(vz_f1/vy_f1))*(180/(math.pi))
        k = 'Piych Info--Pitch Name:'+str(ef1.iloc[d1-p,pitch_name_n])+',Description:'+str(ef1.iloc[d1-p,9])+',VAA:'+str(vaa1)
    return k



##################################################################本体

@app.callback(
        Output('Chart0', 'figure'),
        Output('Chart1', 'figure'),
        Output('Chart2', 'figure'),
        Input('PitchCount0', 'value'),
        Input('PitchCount1', 'value')
)
def update_graph0(r,r1):
    fig_0 = go.Figure()
    fig_1 = go.Figure()
    fig_2 = go.Figure()

    def time_50_0(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n]**2-(2*a.iloc[b-c,ay_n]*50))-a.iloc[b-c,vy0_n])/a.iloc[b-c,ay_n]
    def time_50_1712(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n]**2-(2*a.iloc[b-c,ay_n]*(50-17/12)))-a.iloc[b-c,vy0_n])/a.iloc[b-c,ay_n]
    def time_start(a,b,c):
        return (-a.iloc[b-c,vy0_n]-np.sqrt(a.iloc[b-c,vy0_n]**2-a.iloc[b-c,ay_n]*(100-2*a.iloc[b-c,release_pos_y_n])))/a.iloc[b-c,ay_n]
    def time_whole(a,b,c):
        return time_50_0(a,b,c)-time_start(a,b,c)
    def velx0_start(a,b,c):
        return a.iloc[b-c,vx0_n]+a.iloc[b-c,ax_n]*time_start(a,b,c)
    def vely0_start(a,b,c):
        return a.iloc[b-c,vy0_n]+a.iloc[b-c,ay_n]*time_start(a,b,c)
    def velz0_start(a,b,c):
        return a.iloc[b-c,vz0_n]+a.iloc[b-c,az_n]*time_start(a,b,c)
    def relese_x_correction0(a,b,c):
        return a.iloc[b-c,29]-(a.iloc[b-c,vx0_n]*time_50_1712(a,b,c)+(1/2)*a.iloc[b-c,ax_n]*time_50_1712(a,b,c)**2)
    def relese_z_correction0(a,b,c):
        return a.iloc[b-c,30]-(a.iloc[b-c,vz0_n]*time_50_1712(a,b,c)+(1/2)*a.iloc[b-c,az_n]*time_50_1712(a,b,c)**2)
    def relese_x_start0(a,b,c):
        return relese_x_correction0(a,b,c)+a.iloc[b-c,vx0_n]*time_start(a,b,c)+(1/2)*a.iloc[b-c,ax_n]*time_start(a,b,c)**2
    def relese_y_start0(a,b,c):
        return 50+a.iloc[b-c,vy0_n]*time_start(a,b,c)+(1/2)*a.iloc[b-c,ay_n]*time_start(a,b,c)**2
    def relese_z_start0(a,b,c):
        return relese_z_correction0(a,b,c)+a.iloc[b-c,vz0_n]*time_start(a,b,c)+(1/2)*a.iloc[b-c,az_n]*time_start(a,b,c)**2
    if r is None or r == 'Pitch Count':
        pass
    else:
        ax = ef.iloc[length-r,ax_n]
        ay = ef.iloc[length-r,ay_n]
        az = ef.iloc[length-r,az_n]
        t_50_0 = time_50_0(ef,length,r)
        t_50_1712 = time_50_1712(ef,length,r)
        t_start = time_start(ef,length,r)
        t_whole = time_whole(ef,length,r)
        v_x0_s = velx0_start(ef,length,r)
        v_y0_s = vely0_start(ef,length,r)
        v_z0_s = velz0_start(ef,length,r)
        r_x_s = relese_x_start0(ef,length,r)
        r_y_s = relese_y_start0(ef,length,r)
        r_z_s = relese_z_start0(ef,length,r)
        x0_1=[]
        y0_1=[]
        z0_1=[]
        for u in frange(0,t_whole,Per):
            x0_1.append(r_x_s+v_x0_s*u+(1/2)*ax*u**2)
            y0_1.append(r_y_s+v_y0_s*u+(1/2)*ay*u**2)
            z0_1.append(r_z_s+v_z0_s*u+(1/2)*az*u**2)
        fig_0.add_trace(go.Scatter3d(
            x=x0_1,
            y=y0_1,
            z=z0_1,
            mode='markers',
            marker=dict(
            size=5,
            color='blue'
        ),
        opacity=0.5,
        name='The Picth Trajectory'
        ))
        fig_1.add_trace(go.Scatter(
            x=y0_1,
            y=z0_1,
            mode='markers',
            marker=dict(
            size=3,
            color='blue'
        ),
        opacity=0.5,
        name='1'
        ))
        fig_2.add_trace(go.Scatter(
            x=y0_1,
            y=x0_1,
            mode='markers',
            marker=dict(
            size=3,
            color='blue'
        ),
        opacity=0.5,
        name='1'
        ))

        x1_1=[]
        y1_1=[]
        z1_1=[]
        for u in frange(0,t_whole,Per):
            x1_1.append(r_x_s+v_x0_s*(0.1)+(1/2)*ax*(0.1)**2+(v_x0_s+ax*0.1)*u)
            y1_1.append(r_y_s+v_y0_s*(0.1)+(1/2)*ay*(0.1)**2+(v_y0_s+ay*0.1)*u+(1/2)*ay*(u)**2)
            z1_1.append(r_z_s+v_z0_s*(0.1)+(1/2)*az*(0.1)**2+(v_z0_s+az*0.1)*u+(1/2)*g_acceleretion*(u)**2)
        fig_0.add_trace(go.Scatter3d(
            x=x1_1,
            y=y1_1,
            z=z1_1,
            mode='markers',
            marker=dict(
                size=3,
                color='rgb(49, 140, 231)'
            ),
            opacity=0.5,
            name='Without Movement from RP'
        ))

        x2_1=[]
        y2_1=[]
        z2_1=[]
        for p in frange(0,t_50_0-t_50_1712+0.167,Per):
            x2_1.append(r_x_s+v_x0_s*(t_50_1712-t_start-0.167)+(1/2)*ax*(t_50_1712-t_start-0.167)**2+(v_x0_s+ax*(t_50_1712-t_start-0.167))*p)
            y2_1.append(r_y_s+v_y0_s*(t_50_1712-t_start-0.167)+(1/2)*ay*(t_50_1712-t_start-0.167)**2+(v_y0_s+ay*(t_50_1712-t_start-0.167))*p+(1/2)*ay*(p)**2)
            z2_1.append(r_z_s+v_z0_s*(t_50_1712-t_start-0.167)+(1/2)*az*(t_50_1712-t_start-0.167)**2+(v_z0_s+az*(t_50_1712-t_start-0.167))*p+(1/2)*g_acceleretion*(p)**2)
        fig_0.add_trace(go.Scatter3d(x=x2_1,
            y=y2_1,
            z=z2_1,
            mode='markers',
            marker=dict(
                size=3,
                color='rgb(49, 140, 231)'
            ),
            opacity=0.5,
            name='Without Movement from CP'
        ))

        x_rp_1=[]
        y_rp_1=[]
        z_rp_1=[]
        x_rp_1.append(r_x_s+v_x0_s*(0.1)+(1/2)*ax*(0.1)**2)
        y_rp_1.append(r_y_s+v_y0_s*(0.1)+(1/2)*ay*(0.1)**2)
        z_rp_1.append(r_z_s+v_z0_s*(0.1)+(1/2)*az*(0.1)**2)
        fig_0.add_trace(go.Scatter3d(x=x_rp_1,
            y=y_rp_1,
            z=z_rp_1,
            mode='markers',
            marker=dict(
                size=7,
                color='black'
            ),
            opacity=1,
            name='Recognition Point'
        ))

        x_cp_1=[]
        y_cp_1=[]
        z_cp_1=[]
        x_cp_1.append(r_x_s+v_x0_s*(t_50_1712-t_start-0.167)+(1/2)*ax*(t_50_1712-t_start-0.167)**2)
        y_cp_1.append(r_y_s+v_y0_s*(t_50_1712-t_start-0.167)+(1/2)*ay*(t_50_1712-t_start-0.167)**2)
        z_cp_1.append(r_z_s+v_z0_s*(t_50_1712-t_start-0.167)+(1/2)*az*(t_50_1712-t_start-0.167)**2)
        fig_0.add_trace(go.Scatter3d(x=x_cp_1,
            y=y_cp_1,
            z=z_cp_1,
            mode='markers',
            marker=dict(
                size=7,
                color='black'
            ),
        opacity=1,
        name='Commit Point'
        ))

        x_sz=[]
        y_sz=[]
        z_sz=[]
        x_sz.append(17/24)
        y_sz.append(17/12)
        z_sz.append(ef.iloc[length-r,sz_bot_n])
        x_sz.append(-17/24)
        y_sz.append(17/12)
        z_sz.append(ef.iloc[length-r,sz_bot_n])
        x_sz.append(-17/24)
        y_sz.append(17/12)
        z_sz.append(ef.iloc[length-r,sz_top_n])
        x_sz.append(17/24)
        y_sz.append(17/12)
        z_sz.append(ef.iloc[length-r,sz_top_n])
        x_sz.append(17/24)
        y_sz.append(17/12)
        z_sz.append(ef.iloc[length-r,sz_bot_n])
        fig_0.add_trace(go.Scatter3d(
            x=x_sz,
            y=y_sz,
            z=z_sz,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            opacity=1,
            name='Strike Zone(1)'
        ))

        fig_0.update_scenes(aspectratio_x=1,
                  aspectratio_y=2.5,
                  aspectratio_z=1
                  )
        fig_0.update_layout(
            scene = dict(
                xaxis = dict(nticks=10, range=[-3.5,3.5],),
                yaxis = dict(nticks=20, range=[0,60],),
                zaxis = dict(nticks=10, range=[0,7],),),
            scene_aspectmode = 'manual'
        )
    if r1 is None or r1 == 'Pitch Count':
        pass
    else:
        ax_w = ef1.iloc[d1-r1,ax_n]
        ay_w = ef1.iloc[d1-r1,ay_n]
        az_w = ef1.iloc[d1-r1,az_n]
        t_50_0_w = time_50_0(ef1,d1,r1)
        t_50_1712_w = time_50_1712(ef1,d1,r1)
        t_start_w = time_start(ef1,d1,r1)
        t_whole_w = time_whole(ef1,d1,r1)
        v_x0_s_w = velx0_start(ef1,d1,r1)
        v_y0_s_w = vely0_start(ef1,d1,r1)
        v_z0_s_w = velz0_start(ef1,d1,r1)
        r_x_s_w = relese_x_start0(ef1,d1,r1)
        r_y_s_w = relese_y_start0(ef1,d1,r1)
        r_z_s_w = relese_z_start0(ef1,d1,r1)
        x0_2=[]
        y0_2=[]
        z0_2=[]
        for u in frange(0,t_whole_w,Per):
            x0_2.append(r_x_s_w+v_x0_s_w*u+(1/2)*ax_w*u**2)
            y0_2.append(r_y_s_w+v_y0_s_w*u+(1/2)*ay_w*u**2)
            z0_2.append(r_z_s_w+v_z0_s_w*u+(1/2)*az_w*u**2)
        fig_0.add_trace(go.Scatter3d(
        x=x0_2,
        y=y0_2,
        z=z0_2,
        mode='markers',
        marker=dict(
        size=5,
        color='red'
        ),
        opacity=0.5,
        name='The Picth Trajectory2'
        ))
        fig_1.add_trace(go.Scatter(
            x=y0_2,
            y=z0_2,
            mode='markers',
            marker=dict(
            size=3,
            color='red'
        ),
        opacity=0.5,
        name='2'
        ))
        fig_2.add_trace(go.Scatter(
            x=y0_2,
            y=x0_2,
            mode='markers',
            marker=dict(
            size=3,
            color='red'
        ),
        opacity=0.5,
        name='2'
        ))


        x1_2=[]
        y1_2=[]
        z1_2=[]
        for u in frange(0,t_whole_w,Per):
            x1_2.append(r_x_s_w+v_x0_s_w*(0.1)+(1/2)*ax_w*(0.1)**2+(v_x0_s_w+ax_w*0.1)*u)
            y1_2.append(r_y_s_w+v_y0_s_w*(0.1)+(1/2)*ay_w*(0.1)**2+(v_y0_s_w+ay_w*0.1)*u+(1/2)*ay_w*(u)**2)
            z1_2.append(r_z_s_w+v_z0_s_w*(0.1)+(1/2)*az_w*(0.1)**2+(v_z0_s_w+az_w*0.1)*u+(1/2)*g_acceleretion*(u)**2)
        fig_0.add_trace(go.Scatter3d(
            x=x1_2,
            y=y1_2,
            z=z1_2,
            mode='markers',
            marker=dict(
                size=3,
                color='orange'
            ),
            opacity=0.5,
            name='Without Movement from RP'
        ))

        x2_2=[]
        y2_2=[]
        z2_2=[]
        for p in frange(0,t_50_0_w-t_50_1712_w+0.167,Per):
            x2_2.append(r_x_s_w+v_x0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*ax_w*(t_50_1712_w-t_start_w-0.167)**2+(v_x0_s_w+ax_w*(t_50_1712_w-t_start_w-0.167))*p)
            y2_2.append(r_y_s_w+v_y0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*ay_w*(t_50_1712_w-t_start_w-0.167)**2+(v_y0_s_w+ay_w*(t_50_1712_w-t_start_w-0.167))*p+(1/2)*ay_w*(p)**2)
            z2_2.append(r_z_s_w+v_z0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*az_w*(t_50_1712_w-t_start_w-0.167)**2+(v_z0_s_w+az_w*(t_50_1712_w-t_start_w-0.167))*p+(1/2)*g_acceleretion*(p)**2)
        fig_0.add_trace(go.Scatter3d(
            x=x2_2,
            y=y2_2,
            z=z2_2,
            mode='markers',
            marker=dict(
                size=3,
                color='orange'
            ),
            opacity=0.5,
            name='Without Movement from CP'
        ))

        x_rp_2=[]
        y_rp_2=[]
        z_rp_2=[]
        x_rp_2.append(r_x_s_w+v_x0_s_w*(0.1)+(1/2)*ax_w*(0.1)**2)
        y_rp_2.append(r_y_s_w+v_y0_s_w*(0.1)+(1/2)*ay_w*(0.1)**2)
        z_rp_2.append(r_z_s_w+v_z0_s_w*(0.1)+(1/2)*az_w*(0.1)**2)
        fig_0.add_trace(go.Scatter3d(
            x=x_rp_2,
            y=y_rp_2,
            z=z_rp_2,
            mode='markers',
            marker=dict(
                size=7,
                color='black'
            ),
            opacity=1,
            name='Recognition Point'
        ))

        x_cp_2=[]
        y_cp_2=[]
        z_cp_2=[]
        x_cp_2.append(r_x_s_w+v_x0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*ax_w*(t_50_1712_w-t_start_w-0.167)**2)
        y_cp_2.append(r_y_s_w+v_y0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*ay_w*(t_50_1712_w-t_start_w-0.167)**2)
        z_cp_2.append(r_z_s_w+v_z0_s_w*(t_50_1712_w-t_start_w-0.167)+(1/2)*az_w*(t_50_1712_w-t_start_w-0.167)**2)
        fig_0.add_trace(go.Scatter3d(x=x_cp_2,
            y=y_cp_2,
            z=z_cp_2,
            mode='markers',
            marker=dict(
                size=7,
                color='black'
            ),
        opacity=1,
        name='Commit Point'
        ))

        x_sz_2=[]
        y_sz_2=[]
        z_sz_2=[]
        x_sz_2.append(17/24)
        y_sz_2.append(17/12)
        z_sz_2.append(ef1.iloc[d1-r1,sz_bot_n])
        x_sz_2.append(-17/24)
        y_sz_2.append(17/12)
        z_sz_2.append(ef1.iloc[d1-r1,sz_bot_n])
        x_sz_2.append(-17/24)
        y_sz_2.append(17/12)
        z_sz_2.append(ef1.iloc[d1-r1,sz_top_n])
        x_sz_2.append(17/24)
        y_sz_2.append(17/12)
        z_sz_2.append(ef1.iloc[d1-r1,sz_top_n])
        x_sz_2.append(17/24)
        y_sz_2.append(17/12)
        z_sz_2.append(ef1.iloc[d1-r1,sz_bot_n])
        fig_0.add_trace(go.Scatter3d(
            x=x_sz_2,
            y=y_sz_2,
            z=z_sz_2,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            opacity=1,
            name='Strike Zone(2)',
            visible='legendonly'
        ))

    fig_0.update_scenes(aspectratio_x=1,
        aspectratio_y=2.5,
        aspectratio_z=1
        )
    fig_0.update_layout(
        scene = dict(
            xaxis = dict(nticks=10, range=[-3.5,3.5],),
            yaxis = dict(nticks=20, range=[0,60],),
            zaxis = dict(nticks=10, range=[0,7],),),
        scene_aspectmode = 'manual'
    )

    fig_1.update_layout(
        xaxis = dict(range=(-2,60)),
        yaxis = dict(range=(0,10)),
        legend = dict(
            xanchor='left',
            yanchor='bottom',
            x=0.02,
            y=0.9,
            orientation='h'
            ),
        margin=dict(t=80, b=40, l=5, r=5)
    )

    fig_2.update_layout(
        xaxis = dict(range=(-2,60)),
        yaxis = dict(range=(5,-5)),
        legend = dict(
            xanchor='left',
            yanchor='bottom',
            x=0.02,
            y=0.9,
            orientation='h'
            ),
        margin=dict(t=40, b=80, l=5, r=5)
    )

    return fig_0, fig_1, fig_2

app.run_server(mode="inline")

'''・これは軌道計算機です。加速度を一定としているため、簡易的なものでありますが、お役に立てば幸いです。（データが少ない上、実際は海抜や湿度、風、空気抵抗による加速度の変化なども考慮しなければならないところ、等加速度運動とみなしているため、誤差があります。）
・Perは時間間隔(s)です。任意で変えられます。
・距離の単位は統一でfeetです。
・キャッチャー視点で後ろから前がyの正方向、左から右がxの正方向、下から上がzの正方向です。
・原点はホームプレート上の後方の角です。
・VAA、HAAは共にホームプレート上前方通過時(y=17/12)での角度です。。
・これを利用して、記事、論文等を一般に公開、発表する場合、X(Twitter)@empty8128まで連絡をお願いします。
・同様に不具合、間違い、質問なのがあれば連絡をお願いします。'''