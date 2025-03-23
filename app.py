import streamlit as st
from pybaseball import pitching_stats
import pandas as pd
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import plotly.graph_objects as go
import numpy as np
import json

# データの作成

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

def load_data():
    file_path = 'data/player.json' 
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
data = load_data()

st.set_page_config(layout="wide")

st.markdown("## Pitch Trajector")
st.sidebar.markdown("Please select in the order of year-playername-pitch")


###年指定0

y0 = [var for var in range(2015,2026,1)]

y0_1 = st.sidebar.selectbox(
    'Year0',
    y0,
    index = None,
    placeholder='Please select a year.')

###選手指定0

if y0_1 is None:
    pl0 = ''
else:
    pl0 = []
    for i in range(0,len(data[str(y0_1)])):
        pl0.append(str((data[str(y0_1)][i]["name"]).split(' ',1)[0])+' '+str((data[str(y0_1)][i]["name"]).split(' ',1)[1]))

pl0_1 = st.sidebar.selectbox(
        'Player Name0',
        pl0,
        index = None,
        placeholder='Please select a player.'
        )

###球指定0
if y0_1 is None or pl0_1 is None:
    pi0=''
else:
    with st.spinner('Wait a minute'):
        for i in range(len(data[str(y0_1)])):
            name= str((data[str(y0_1)][i]["name"]).split(' ',1)[0])+' '+str((data[str(y0_1)][i]["name"]).split(' ',1)[1])
            if pl0_1==name:
                name_id=str(data[str(y0_1)][i]["id"])
                break
            else:
                pass
        pf0 = pd.DataFrame()
        pf0_0 = statcast_pitcher(str(y0_1)+'-01-01', str(y0_1)+'-12-31', name_id)
        if Game_Type == 'R':
            pf0_1 = pf0_0[pf0_0['game_type']== 'R']
        elif Game_Type == 'P':
            pf0_1 = pf0_0[pf0_0['game_type'].isin(['F', 'D', 'L', 'W'])]
        len0 = pf0_1.shape[0]
        num=[]
        for t in range(len0,0,-1):
            num.append(t)
        pf0 = pf0_1.assign(n=num)

        p_t_n0 = pf0.columns.get_loc('pitch_type')
        g_d_n0 = pf0.columns.get_loc('game_date')
        r_s_n0 = pf0.columns.get_loc('release_speed')
        b_n0 = pf0.columns.get_loc('balls')
        s_n0 = pf0.columns.get_loc('strikes')
        o_w_u_n0 = pf0.columns.get_loc('outs_when_up')
        inn_n0 = pf0.columns.get_loc('inning')
        vx0_n0 = pf0.columns.get_loc('vx0')
        vy0_n0 = pf0.columns.get_loc('vy0')
        vz0_n0 = pf0.columns.get_loc('vz0')
        ax_n0 = pf0.columns.get_loc('ax')
        ay_n0 = pf0.columns.get_loc('ay')
        sz_top_n0 = pf0.columns.get_loc('sz_top')
        sz_bot_n0 = pf0.columns.get_loc('sz_bot')
        az_n0 = pf0.columns.get_loc('az')
        r_p_y_n0 = pf0.columns.get_loc('release_pos_y')

        pi0=[]
        pi0.extend(reversed([str('{:0=4}'.format(x))+','+str(pf0.iloc[len0-x,g_d_n0])+','+str(pf0.iloc[len0-x,p_t_n0])+',IP:'+str(pf0.iloc[len0-x,inn_n0])+',B-S-O:'+str(pf0.iloc[len0-x,b_n0])+'-'+str(pf0.iloc[len0-x,s_n0])+'-'+str(pf0.iloc[len0-x,o_w_u_n0])+','+str(pf0.iloc[len0-x,r_s_n0])+'(mph)' for x in pf0['n']]))

pi0_1 = st.sidebar.selectbox(
    'Pitch0',
    pi0,
    index = None,
    placeholder='Please select a pitch.')

###グラフ

fig_0 = go.Figure()

###グラフ0


if y0_1 is None or pl0_1 is None or pi0_1 is None:
    pass
else:
    def t_50_0(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n0]**2-(2*a.iloc[b-c,ay_n0]*50))-a.iloc[b-c,vy0_n0])/a.iloc[b-c,ay_n0]
    def t_50_1712(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n0]**2-(2*a.iloc[b-c,ay_n0]*(50-17/12)))-a.iloc[b-c,vy0_n0])/a.iloc[b-c,ay_n0]
    def t_s(a,b,c):
        return (-a.iloc[b-c,vy0_n0]-np.sqrt(a.iloc[b-c,vy0_n0]**2-a.iloc[b-c,ay_n0]*(100-2*a.iloc[b-c,r_p_y_n0])))/a.iloc[b-c,ay_n0]
    def t_w(a,b,c):
        return t_50_0(a,b,c)-t_s(a,b,c)
    def v_x0_s(a,b,c):
        return a.iloc[b-c,vx0_n0]+a.iloc[b-c,ax_n0]*t_s(a,b,c)
    def v_y0_s(a,b,c):
        return a.iloc[b-c,vy0_n0]+a.iloc[b-c,ay_n0]*t_s(a,b,c)
    def v_z0_s(a,b,c):
        return a.iloc[b-c,vz0_n0]+a.iloc[b-c,az_n0]*t_s(a,b,c)
    def r_x_c0(a,b,c):
        return a.iloc[b-c,29]-(a.iloc[b-c,vx0_n0]*t_50_1712(a,b,c)+(1/2)*a.iloc[b-c,ax_n0]*t_50_1712(a,b,c)**2)
    def r_z_c0(a,b,c):
        return a.iloc[b-c,30]-(a.iloc[b-c,vz0_n0]*t_50_1712(a,b,c)+(1/2)*a.iloc[b-c,az_n0]*t_50_1712(a,b,c)**2)
    def r_x_s0(a,b,c):
        return r_x_c0(a,b,c)+a.iloc[b-c,vx0_n0]*t_s(a,b,c)+(1/2)*a.iloc[b-c,ax_n0]*t_s(a,b,c)**2
    def r_y_s0(a,b,c):
        return 50+a.iloc[b-c,vy0_n0]*t_s(a,b,c)+(1/2)*a.iloc[b-c,ay_n0]*t_s(a,b,c)**2
    def r_z_s0(a,b,c):
        return r_z_c0(a,b,c)+a.iloc[b-c,vz0_n0]*t_s(a,b,c)+(1/2)*a.iloc[b-c,az_n0]*t_s(a,b,c)**2

    n0 = int(pi0_1[0:4])

    ax0 = pf0.iloc[len0-n0,ax_n0]
    ay0 = pf0.iloc[len0-n0,ay_n0]
    az0 = pf0.iloc[len0-n0,az_n0]
    t_50_00 = t_50_0(pf0,len0,n0)
    t_50_17120 = t_50_1712(pf0,len0,n0)
    t_start0 = t_s(pf0,len0,n0)
    t_whole0 = t_w(pf0,len0,n0)
    v_x0_s0 = v_x0_s(pf0,len0,n0)
    v_y0_s0 = v_y0_s(pf0,len0,n0)
    v_z0_s0 = v_z0_s(pf0,len0,n0)
    r_x_s0 = r_x_s0(pf0,len0,n0)
    r_y_s0 = r_y_s0(pf0,len0,n0)
    r_z_s0 = r_z_s0(pf0,len0,n0)
    x0_1=[]
    y0_1=[]
    z0_1=[]
    for u in frange(0,t_whole0,Per):
        x0_1.append(r_x_s0+v_x0_s0*u+(1/2)*ax0*u**2)
        y0_1.append(r_y_s0+v_y0_s0*u+(1/2)*ay0*u**2)
        z0_1.append(r_z_s0+v_z0_s0*u+(1/2)*az0*u**2)
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

    x0_2=[]
    y0_2=[]
    z0_2=[]
    for u in frange(0,t_whole0,Per):
        x0_2.append(r_x_s0+v_x0_s0*(0.1)+(1/2)*ax0*(0.1)**2+(v_x0_s0+ax0*0.1)*u)
        y0_2.append(r_y_s0+v_y0_s0*(0.1)+(1/2)*ay0*(0.1)**2+(v_y0_s0+ay0*0.1)*u+(1/2)*ay0*(u)**2)
        z0_2.append(r_z_s0+v_z0_s0*(0.1)+(1/2)*az0*(0.1)**2+(v_z0_s0+az0*0.1)*u+(1/2)*g_acceleretion*(u)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x0_2,
        y=y0_2,
        z=z0_2,
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(49, 140, 231)'
        ),
        opacity=0.5,
        name='Without Movement from RP'
    ))

    x0_2=[]
    y0_2=[]
    z0_2=[]
    for p in frange(0,t_50_00-t_50_17120+0.167,Per):
        x0_2.append(r_x_s0+v_x0_s0*(t_50_17120-t_start0-0.167)+(1/2)*ax0*(t_50_17120-t_start0-0.167)**2+(v_x0_s0+ax0*(t_50_17120-t_start0-0.167))*p)
        y0_2.append(r_y_s0+v_y0_s0*(t_50_17120-t_start0-0.167)+(1/2)*ay0*(t_50_17120-t_start0-0.167)**2+(v_y0_s0+ay0*(t_50_17120-t_start0-0.167))*p+(1/2)*ay0*(p)**2)
        z0_2.append(r_z_s0+v_z0_s0*(t_50_17120-t_start0-0.167)+(1/2)*az0*(t_50_17120-t_start0-0.167)**2+(v_z0_s0+az0*(t_50_17120-t_start0-0.167))*p+(1/2)*g_acceleretion*(p)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x0_2,
        y=y0_2,
        z=z0_2,
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(49, 140, 231)'
        ),
        opacity=0.5,
        name='Without Movement from CP'
    ))

    x0_rp=[]
    y0_rp=[]
    z0_rp=[]
    x0_rp.append(r_x_s0+v_x0_s0*(0.1)+(1/2)*ax0*(0.1)**2)
    y0_rp.append(r_y_s0+v_y0_s0*(0.1)+(1/2)*ay0*(0.1)**2)
    z0_rp.append(r_z_s0+v_z0_s0*(0.1)+(1/2)*az0*(0.1)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x0_rp,
        y=y0_rp,
        z=z0_rp,
        mode='markers',
        marker=dict(
            size=7,
            color='black'
        ),
        opacity=1,
        name='Recognition Point'
    ))

    x0_cp=[]
    y0_cp=[]
    z0_cp=[]
    x0_cp.append(r_x_s0+v_x0_s0*(t_50_17120-t_start0-0.167)+(1/2)*ax0*(t_50_17120-t_start0-0.167)**2)
    y0_cp.append(r_y_s0+v_y0_s0*(t_50_17120-t_start0-0.167)+(1/2)*ay0*(t_50_17120-t_start0-0.167)**2)
    z0_cp.append(r_z_s0+v_z0_s0*(t_50_17120-t_start0-0.167)+(1/2)*az0*(t_50_17120-t_start0-0.167)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x0_cp,
        y=y0_cp,
        z=z0_cp,
        mode='markers',
        marker=dict(
            size=7,
            color='black'
        ),
    opacity=1,
    name='Commit Point'
    ))

    x0_sz=[]
    y0_sz=[]
    z0_sz=[]
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[len0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[len0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[len0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[len0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[len0-n0,sz_bot_n0])
    fig_0.add_trace(go.Scatter3d(
        x=x0_sz,
        y=y0_sz,
        z=z0_sz,
        mode='lines',
        line=dict(
            color='black',
            width=3
        ),
        opacity=1,
        name='Strike Zone(0)'
    ))


    fig_0.update_scenes(
        aspectratio_x=1,
        aspectratio_y=2.5,
        aspectratio_z=1
        )
    fig_0.update_layout(
        scene = dict(
            xaxis = dict(nticks=10, range=[-3.5,3.5],),
            yaxis = dict(nticks=20, range=[0,60],),
            zaxis = dict(nticks=10, range=[0,7],),),
        height=800,
        width=1000,
        scene_aspectmode = 'manual',
        legend=dict(
            xanchor='left',
            yanchor='top',
            x=0.01,
            y=1,
            orientation='h',
        )
    )

###表示
st.plotly_chart(fig_0)
