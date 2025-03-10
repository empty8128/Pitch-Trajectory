import streamlit as st
from datetime import datetime
from pybaseball import pitching_stats
import pandas as pd
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import plotly.graph_objects as go
import numpy as np

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

st.set_page_config(layout="wide")

st.markdown("## Pitch Trajector")

###年指定0

year0 = [var for var in range(2015,datetime.now().year+1,1)]

year0_1 = st.sidebar.selectbox(
    'Year0',
    year0,
    index = None,
    placeholder='Please select a year.')

###選手指定0

if year0_1 is None:
    player0 = ''
else:
    player0 = [var for var in sorted(pitching_stats(year0_1, qual=1)['Name'].unique())]

player0_1 = st.sidebar.selectbox(
    'Player Name0',
    player0,
    index = None,
    placeholder='Please select a player.'
    )

###球指定0
if year0_1 is None or player0_1 is None:
    pitch0=''
else:
    pf0 = pd.DataFrame()
    pf0_0 = statcast_pitcher(str(year0_1)+'-01-01', str(year0_1)+'-12-31', playerid_lookup(player0_1.split()[1], player0_1.split()[0], fuzzy=True).iloc[0,2])
    if Game_Type == 'R':
        pf0_1 = pf0_0[pf0_0['game_type']== 'R']
    elif Game_Type == 'P':
        pf0_1 = pf0_0[pf0_0['game_type'].isin(['F', 'D', 'L', 'W'])]
    length0 = pf0_1.shape[0]
    num=[]
    for t in range(length0,0,-1):
        num.append(t)
    pf0 = pf0_1.assign(n=num)

    pitch_type_n0 = pf0.columns.get_loc('pitch_type')
    game_date_n0 = pf0.columns.get_loc('game_date')
    release_speed_n0 = pf0.columns.get_loc('release_speed')
    player_name_n0 = pf0.columns.get_loc('player_name')
    balls_n0 = pf0.columns.get_loc('balls')
    strikes_n0 = pf0.columns.get_loc('strikes')
    outs_when_up_n0 = pf0.columns.get_loc('outs_when_up')
    inning_n0 = pf0.columns.get_loc('inning')
    vx0_n0 = pf0.columns.get_loc('vx0')
    vy0_n0 = pf0.columns.get_loc('vy0')
    vz0_n0 = pf0.columns.get_loc('vz0')
    ax_n0 = pf0.columns.get_loc('ax')
    ay_n0 = pf0.columns.get_loc('ay')
    sz_top_n0 = pf0.columns.get_loc('sz_top')
    sz_bot_n0 = pf0.columns.get_loc('sz_bot')
    az_n0 = pf0.columns.get_loc('az')
    release_pos_y_n = pf0.columns.get_loc('release_pos_y')
    pitch_name_n0 = pf0.columns.get_loc('pitch_name')

    pitch0=[]
    pitch0.extend(reversed([str('{:0=4}'.format(x))+','+str(pf0.iloc[length0-x,game_date_n0])+','+str(pf0.iloc[length0-x,pitch_type_n0])+',IP:'+str(pf0.iloc[length0-x,inning_n0])+',B-S-O:'+str(pf0.iloc[length0-x,balls_n0])+'-'+str(pf0.iloc[length0-x,strikes_n0])+'-'+str(pf0.iloc[length0-x,outs_when_up_n0])+','+str(pf0.iloc[length0-x,release_speed_n0])+'(mph)' for x in pf0['n']]))

pitch0_1 = st.sidebar.selectbox(
    'Pitch0',
    pitch0,
    index = None,
    placeholder='Please select a pitch.')

###年指定1

year1 = [var for var in range(2015,datetime.now().year+1,1)]

year1_1 = st.sidebar.selectbox(
    'Year1',
    year1,
    index = None,
    placeholder='Please select a year.'
    )

###選手指定1

if year1_1 is None:
    player1 = ''
else:
    player1 = [var for var in sorted(pitching_stats(year1_1, qual=1)['Name'].unique())]

player1_1 = st.sidebar.selectbox(
    'Player Name1',
    player1,
    index = None,
    placeholder='Please select a player.'
    )

###球指定1

if year1_1 is None or player1_1 is None:
    pitch1=''
else:
    pf1 = pd.DataFrame()
    pf1_0 = statcast_pitcher(str(year1_1)+'-01-01', str(year1_1)+'-12-31', playerid_lookup(player1_1.split()[1], player1_1.split()[0], fuzzy=True).iloc[0,2])
    if Game_Type == 'R':
        pf1_1 = pf1_0[pf1_0['game_type']== 'R']
    elif Game_Type == 'P':
        pf1_1 = pf1_0[pf1_0['game_type'].isin(['F', 'D', 'L', 'W'])]
    length1 = pf1_1.shape[0]
    num1=[]
    for t in range(length1,0,-1):
        num1.append(t)
    pf1 = pf1_1.assign(n=num1)

    pitch_type_n1 = pf1.columns.get_loc('pitch_type')
    game_date_n1 = pf1.columns.get_loc('game_date')
    release_speed_n1 = pf1.columns.get_loc('release_speed')
    player_name_n1 = pf1.columns.get_loc('player_name')
    balls_n1 = pf1.columns.get_loc('balls')
    strikes_n1 = pf1.columns.get_loc('strikes')
    outs_when_up_n1 = pf1.columns.get_loc('outs_when_up')
    inning_n1 = pf1.columns.get_loc('inning')
    vx0_n1 = pf1.columns.get_loc('vx0')
    vy0_n1 = pf1.columns.get_loc('vy0')
    vz0_n1 = pf1.columns.get_loc('vz0')
    ax_n1 = pf1.columns.get_loc('ax')
    ay_n1 = pf1.columns.get_loc('ay')
    sz_top_n1 = pf1.columns.get_loc('sz_top')
    sz_bot_n1 = pf1.columns.get_loc('sz_bot')
    az_n1 = pf1.columns.get_loc('az')
    release_pos_y_n = pf1.columns.get_loc('release_pos_y')
    pitch_name_n1 = pf1.columns.get_loc('pitch_name')

    pitch1=[]
    pitch1.extend(reversed([str('{:0=4}'.format(x))+','+str(pf1.iloc[length1-x,game_date_n1])+','+str(pf1.iloc[length1-x,pitch_type_n1])+',IP:'+str(pf1.iloc[length1-x,inning_n1])+',B-S-O:'+str(pf1.iloc[length1-x,balls_n1])+'-'+str(pf1.iloc[length1-x,strikes_n1])+'-'+str(pf1.iloc[length1-x,outs_when_up_n1])+','+str(pf1.iloc[length1-x,release_speed_n1])+'(mph)' for x in pf1['n']]))

pitch1_1 = st.sidebar.selectbox(
    'Pitch1',
    pitch1,
    index = None,
    placeholder='Please select a pitch.')


###グラフ

fig_0 = go.Figure()
fig_1 = go.Figure()
fig_2 = go.Figure()

###グラフ0


if year0_1 is None or player0_1 is None or pitch0_1 is None:
    pass
else:
    def t_50_0(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n0]**2-(2*a.iloc[b-c,ay_n0]*50))-a.iloc[b-c,vy0_n0])/a.iloc[b-c,ay_n0]
    def t_50_1712(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n0]**2-(2*a.iloc[b-c,ay_n0]*(50-17/12)))-a.iloc[b-c,vy0_n0])/a.iloc[b-c,ay_n0]
    def t_s(a,b,c):
        return (-a.iloc[b-c,vy0_n0]-np.sqrt(a.iloc[b-c,vy0_n0]**2-a.iloc[b-c,ay_n0]*(100-2*a.iloc[b-c,release_pos_y_n])))/a.iloc[b-c,ay_n0]
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

    n0 = int(pitch0_1[0:4])

    ax0 = pf0.iloc[length0-n0,ax_n0]
    ay0 = pf0.iloc[length0-n0,ay_n0]
    az0 = pf0.iloc[length0-n0,az_n0]
    t_50_00 = t_50_0(pf0,length0,n0)
    t_50_17120 = t_50_1712(pf0,length0,n0)
    t_start0 = t_s(pf0,length0,n0)
    t_whole0 = t_w(pf0,length0,n0)
    v_x0_s0 = v_x0_s(pf0,length0,n0)
    v_y0_s0 = v_y0_s(pf0,length0,n0)
    v_z0_s0 = v_z0_s(pf0,length0,n0)
    r_x_s0 = r_x_s0(pf0,length0,n0)
    r_y_s0 = r_y_s0(pf0,length0,n0)
    r_z_s0 = r_z_s0(pf0,length0,n0)
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
    z0_sz.append(pf0.iloc[length0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[length0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[length0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[length0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iloc[length0-n0,sz_bot_n0])
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
        name='Strike Zone(1)'
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

###グラフ1

if year1_1 is None or player1_1 is None or pitch1_1 is None:
    pass
else:
    def t_50_0(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n1]**2-(2*a.iloc[b-c,ay_n1]*50))-a.iloc[b-c,vy0_n1])/a.iloc[b-c,ay_n1]
    def t_50_1712(a,b,c):
        return (-np.sqrt(a.iloc[b-c,vy0_n1]**2-(2*a.iloc[b-c,ay_n1]*(50-17/12)))-a.iloc[b-c,vy0_n1])/a.iloc[b-c,ay_n1]
    def t_s(a,b,c):
        return (-a.iloc[b-c,vy0_n1]-np.sqrt(a.iloc[b-c,vy0_n1]**2-a.iloc[b-c,ay_n1]*(100-2*a.iloc[b-c,release_pos_y_n])))/a.iloc[b-c,ay_n1]
    def t_w(a,b,c):
        return t_50_0(a,b,c)-t_s(a,b,c)
    def v_x0_s(a,b,c):
        return a.iloc[b-c,vx0_n1]+a.iloc[b-c,ax_n1]*t_s(a,b,c)
    def v_y0_s(a,b,c):
        return a.iloc[b-c,vy0_n1]+a.iloc[b-c,ay_n1]*t_s(a,b,c)
    def v_z0_s(a,b,c):
        return a.iloc[b-c,vz0_n1]+a.iloc[b-c,az_n1]*t_s(a,b,c)
    def r_x_c1(a,b,c):
        return a.iloc[b-c,29]-(a.iloc[b-c,vx0_n1]*t_50_1712(a,b,c)+(1/2)*a.iloc[b-c,ax_n1]*t_50_1712(a,b,c)**2)
    def r_z_c1(a,b,c):
        return a.iloc[b-c,30]-(a.iloc[b-c,vz0_n1]*t_50_1712(a,b,c)+(1/2)*a.iloc[b-c,az_n1]*t_50_1712(a,b,c)**2)
    def r_x_s0(a,b,c):
        return r_x_c1(a,b,c)+a.iloc[b-c,vx0_n1]*t_s(a,b,c)+(1/2)*a.iloc[b-c,ax_n1]*t_s(a,b,c)**2
    def r_y_s0(a,b,c):
        return 50+a.iloc[b-c,vy0_n1]*t_s(a,b,c)+(1/2)*a.iloc[b-c,ay_n1]*t_s(a,b,c)**2
    def r_z_s0(a,b,c):
        return r_z_c1(a,b,c)+a.iloc[b-c,vz0_n1]*t_s(a,b,c)+(1/2)*a.iloc[b-c,az_n1]*t_s(a,b,c)**2

    n1 = int(pitch1_1[0:4])

    ax1 = pf1.iloc[length1-n1,ax_n1]
    ay1 = pf1.iloc[length1-n1,ay_n1]
    az1 = pf1.iloc[length1-n1,az_n1]
    t_50_01 = t_50_0(pf1,length1,n1)
    t_50_17121 = t_50_1712(pf1,length1,n1)
    t_start1 = t_s(pf1,length1,n1)
    t_whole1 = t_w(pf1,length1,n1)
    v_x0_s1 = v_x0_s(pf1,length1,n1)
    v_y0_s1 = v_y0_s(pf1,length1,n1)
    v_z0_s1 = v_z0_s(pf1,length1,n1)
    r_x_s1 = r_x_s0(pf1,length1,n1)
    r_y_s1 = r_y_s0(pf1,length1,n1)
    r_z_s1 = r_z_s0(pf1,length1,n1)
    x1_1=[]
    y1_1=[]
    z1_1=[]
    for u in frange(0,t_whole1,Per):
        x1_1.append(r_x_s1+v_x0_s1*u+(1/2)*ax1*u**2)
        y1_1.append(r_y_s1+v_y0_s1*u+(1/2)*ay1*u**2)
        z1_1.append(r_z_s1+v_z0_s1*u+(1/2)*az1*u**2)
    fig_0.add_trace(go.Scatter3d(
        x=x1_1,
        y=y1_1,
        z=z1_1,
        mode='markers',
        marker=dict(
        size=5,
        color='red'
        ),
        opacity=0.5,
        name='The Picth Trajectory'
    ))

    x1_2=[]
    y1_2=[]
    z1_2=[]
    for u in frange(0,t_whole1,Per):
        x1_2.append(r_x_s1+v_x0_s1*(0.1)+(1/2)*ax1*(0.1)**2+(v_x0_s1+ax1*0.1)*u)
        y1_2.append(r_y_s1+v_y0_s1*(0.1)+(1/2)*ay1*(0.1)**2+(v_y0_s1+ay1*0.1)*u+(1/2)*ay1*(u)**2)
        z1_2.append(r_z_s1+v_z0_s1*(0.1)+(1/2)*az1*(0.1)**2+(v_z0_s1+az1*0.1)*u+(1/2)*g_acceleretion*(u)**2)
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

    x1_2=[]
    y1_2=[]
    z1_2=[]
    for p in frange(0,t_50_01-t_50_17121+0.167,Per):
        x1_2.append(r_x_s1+v_x0_s1*(t_50_17121-t_start1-0.167)+(1/2)*ax1*(t_50_17121-t_start1-0.167)**2+(v_x0_s1+ax1*(t_50_17121-t_start1-0.167))*p)
        y1_2.append(r_y_s1+v_y0_s1*(t_50_17121-t_start1-0.167)+(1/2)*ay1*(t_50_17121-t_start1-0.167)**2+(v_y0_s1+ay1*(t_50_17121-t_start1-0.167))*p+(1/2)*ay1*(p)**2)
        z1_2.append(r_z_s1+v_z0_s1*(t_50_17121-t_start1-0.167)+(1/2)*az1*(t_50_17121-t_start1-0.167)**2+(v_z0_s1+az1*(t_50_17121-t_start1-0.167))*p+(1/2)*g_acceleretion*(p)**2)
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
        name='Without Movement from CP'
    ))

    x1_rp=[]
    y1_rp=[]
    z1_rp=[]
    x1_rp.append(r_x_s1+v_x0_s1*(0.1)+(1/2)*ax1*(0.1)**2)
    y1_rp.append(r_y_s1+v_y0_s1*(0.1)+(1/2)*ay1*(0.1)**2)
    z1_rp.append(r_z_s1+v_z0_s1*(0.1)+(1/2)*az1*(0.1)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x1_rp,
        y=y1_rp,
        z=z1_rp,
        mode='markers',
        marker=dict(
            size=7,
            color='black'
        ),
        opacity=1,
        name='Recognition Point'
    ))

    x1_cp=[]
    y1_cp=[]
    z1_cp=[]
    x1_cp.append(r_x_s1+v_x0_s1*(t_50_17121-t_start1-0.167)+(1/2)*ax1*(t_50_17121-t_start1-0.167)**2)
    y1_cp.append(r_y_s1+v_y0_s1*(t_50_17121-t_start1-0.167)+(1/2)*ay1*(t_50_17121-t_start1-0.167)**2)
    z1_cp.append(r_z_s1+v_z0_s1*(t_50_17121-t_start1-0.167)+(1/2)*az1*(t_50_17121-t_start1-0.167)**2)
    fig_0.add_trace(go.Scatter3d(
        x=x1_cp,
        y=y1_cp,
        z=z1_cp,
        mode='markers',
        marker=dict(
            size=7,
            color='black'
        ),
    opacity=1,
    name='Commit Point'
    ))

    x1_sz=[]
    y1_sz=[]
    z1_sz=[]
    x1_sz.append(17/24)
    y1_sz.append(17/12)
    z1_sz.append(pf1.iloc[length1-n1,sz_bot_n1])
    x1_sz.append(-17/24)
    y1_sz.append(17/12)
    z1_sz.append(pf1.iloc[length1-n1,sz_bot_n1])
    x1_sz.append(-17/24)
    y1_sz.append(17/12)
    z1_sz.append(pf1.iloc[length1-n1,sz_top_n1])
    x1_sz.append(17/24)
    y1_sz.append(17/12)
    z1_sz.append(pf1.iloc[length1-n1,sz_top_n1])
    x1_sz.append(17/24)
    y1_sz.append(17/12)
    z1_sz.append(pf1.iloc[length1-n1,sz_bot_n1])

    if pitch0_1 is None:
        fig_0.add_trace(go.Scatter3d(
            x=x1_sz,
            y=y1_sz,
            z=z1_sz,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            opacity=1,
            name='Strike Zone(1)'
        ))
    else:
        fig_0.add_trace(go.Scatter3d(
            x=x1_sz,
            y=y1_sz,
            z=z1_sz,
            mode='lines',
            line=dict(
                color='black',
                width=3
            ),
            opacity=1,
            name='Strike Zone(1)',
            visible='legendonly'
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