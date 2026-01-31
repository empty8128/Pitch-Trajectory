import streamlit as st
from pybaseball import pitching_stats
import pandas as pd
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import plotly.graph_objects as go
import numpy as np
import json
import dash
import math
from numpy import pi, sin, cos

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

active_spin = 0

ok = 0

st.set_page_config(layout="wide")

#st.markdown('<div class="con"><h1>Pitch Trajector</h1><p class="twitter">Who made this? <a href= "https://x.com/empty8128">empty8128(𝕏(Twitter))</a></p></div>',unsafe_allow_html=True)
#st.sidebar.markdown("Please select in the order of year-playername-pitch")

st.markdown("""
    <style>
        .con{
            display: flex;
            align-items: center;
            }
    </style>
""", unsafe_allow_html=True)

#回転

theta = np.linspace(0, 2*pi, 120)
phi = np.linspace(0, pi, 60)
u , v = np.meshgrid(theta, phi)
xs = cos(u)*sin(v)
ys = sin(u)*sin(v)
zs = cos(v)


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

spinner_placeholder = st.sidebar.empty()

###球指定0
if y0_1 is None or pl0_1 is None:
    pi0=''
else:
    with spinner_placeholder:
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
            p_n_n0 = pf0.columns.get_loc('pitch_name')
            de_n0 = pf0.columns.get_loc('description')
            r_s_n0 = pf0.columns.get_loc('release_speed')
            h_t_n0 = pf0.columns.get_loc('home_team')
            plate_x_n0 = pf0.columns.get_loc('plate_x')
            plate_z_n0 = pf0.columns.get_loc('plate_z')
            a_t_n0 = pf0.columns.get_loc('away_team')
            zone_n0 = pf0.columns.get_loc('zone') 
            r_s_r_n0 = pf0.columns.get_loc('release_spin_rate')
            p_th_n0 = pf0.columns.get_loc('p_throws')
            s_a_n0 = pf0.columns.get_loc('spin_axis')
            b_n0 = pf0.columns.get_loc('balls')
            s_n0 = pf0.columns.get_loc('strikes')
            o_w_u_n0 = pf0.columns.get_loc('outs_when_up')
            inn_n0 = pf0.columns.get_loc('inning')
            inntb_n0 = pf0.columns.get_loc('inning_topbot')
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
            pi0.extend(reversed([str('{:0=4}'.format(x))+','+str(pf0.iat[len0-x,g_d_n0])+','+str(pf0.iat[len0-x,p_t_n0])+',IP:'+str(pf0.iat[len0-x,inn_n0])+',B-S-O:'+str(pf0.iat[len0-x,b_n0])+'-'+str(pf0.iat[len0-x,s_n0])+'-'+str(pf0.iat[len0-x,o_w_u_n0])+','+str(pf0.iat[len0-x,r_s_n0])+'(mph)' for x in pf0['n']]))

pi0_1 = st.sidebar.selectbox(
    'Pitch0',
    pi0,
    index = None,
    placeholder='Please select a pitch.')

###グラフ

fig_0 = go.Figure()
fig = go.Figure()

###グラフ0


if y0_1 is None or pl0_1 is None or pi0_1 is None:
    pass
else:

    def t_50_0(a,b,c):
        return (-np.sqrt(a.iat[b-c,vy0_n0]**2-(2*a.iat[b-c,ay_n0]*50))-a.iat[b-c,vy0_n0])/a.iat[b-c,ay_n0]
    def t_50_1712(a,b,c):
        return (-np.sqrt(a.iat[b-c,vy0_n0]**2-(2*a.iat[b-c,ay_n0]*(50-17/12)))-a.iat[b-c,vy0_n0])/a.iat[b-c,ay_n0]
    def t_s(a,b,c):
        return (-a.iat[b-c,vy0_n0]-np.sqrt(a.iat[b-c,vy0_n0]**2-a.iat[b-c,ay_n0]*(100-2*a.iat[b-c,r_p_y_n0])))/a.iat[b-c,ay_n0]
    def t_w(a,b,c):
        return t_50_0(a,b,c)-t_s(a,b,c)
    def v_x0_s(a,b,c):
        return a.iat[b-c,vx0_n0]+a.iat[b-c,ax_n0]*t_s(a,b,c)
    def v_y0_s(a,b,c):
        return a.iat[b-c,vy0_n0]+a.iat[b-c,ay_n0]*t_s(a,b,c)
    def v_z0_s(a,b,c):
        return a.iat[b-c,vz0_n0]+a.iat[b-c,az_n0]*t_s(a,b,c)
    def r_x_c0(a,b,c):
        return a.iat[b-c,plate_x_n0]-(a.iat[b-c,vx0_n0]*t_50_1712(a,b,c)+(1/2)*a.iat[b-c,ax_n0]*t_50_1712(a,b,c)**2)
    def r_z_c0(a,b,c):
        return a.iat[b-c,plate_z_n0]-(a.iat[b-c,vz0_n0]*t_50_1712(a,b,c)+(1/2)*a.iat[b-c,az_n0]*t_50_1712(a,b,c)**2)
    def r_x0_s0(a,b,c):
        return r_x_c0(a,b,c)+a.iat[b-c,vx0_n0]*t_s(a,b,c)+(1/2)*a.iat[b-c,ax_n0]*t_s(a,b,c)**2
    def r_y0_s0(a,b,c):
        return 50+a.iat[b-c,vy0_n0]*t_s(a,b,c)+(1/2)*a.iat[b-c,ay_n0]*t_s(a,b,c)**2
    def r_z0_s0(a,b,c):
        return r_z_c0(a,b,c)+a.iat[b-c,vz0_n0]*t_s(a,b,c)+(1/2)*a.iat[b-c,az_n0]*t_s(a,b,c)**2

    n0 = int(pi0_1[0:4])

    for i in range(len(data[str(y0_1)])):
        t= str((data[str(y0_1)][i]["name"]).split(' ',1)[0])+' '+str((data[str(y0_1)][i]["name"]).split(' ',1)[1])
        if pl0_1==t:
            if pf0.iat[len0-n0,p_t_n0] == 'KC':
                pass
            elif pf0.iat[len0-n0,p_t_n0] in data[str(y0_1)][i]:
                active_spin = str(data[str(y0_1)][i][pf0.iat[len0-n0,p_t_n0]])
            else:
                active_spin= -1
            break
        else:
            pass

    vy_f0 = (-np.sqrt(pf0.iat[len0-n0,vy0_n0]**2-(2*pf0.iat[len0-n0,ay_n0]*(50-17/12))))
    t0 = (vy_f0-pf0.iat[len0-n0,vy0_n0])/pf0.iat[len0-n0,ay_n0]
    vz_f0 = pf0.iat[len0-n0,vz0_n0]+pf0.iat[len0-n0,az_n0]*t0
    vx_f0 = pf0.iat[len0-n0,vx0_n0]+pf0.iat[len0-n0,ax_n0]*t0
    vaa0 = round((np.arctan(vz_f0/vy_f0))*(180/(math.pi)),2)
    haa0 = round((np.arctan(vx_f0/vy_f0))*(180/(math.pi)),2)
    k = 'Piych Info--\r\nPitch Name:'+str(pf0.iat[len0-n0,p_n_n0])+',\r\nDescription:'+str(pf0.iat[len0-n0,de_n0])+',\r\nVAA:'+str(vaa0)+',\r\nactive_spin:' +str(active_spin)
    st.sidebar.markdown(k)
    l = str(pf0.iat[len0-n0,r_s_n0]) + '(MPH)  ' + str(pf0.iat[len0-n0,r_s_r_n0])+ '(RPM)'
    st.sidebar.markdown("<p style='font-size:32px; border-bottom: solid 3px; text-align: center;'></p>", unsafe_allow_html=True)

    ax0 = pf0.iat[len0-n0,ax_n0]
    ay0 = pf0.iat[len0-n0,ay_n0]
    az0 = pf0.iat[len0-n0,az_n0]
    t_50_00 = t_50_0(pf0,len0,n0)
    t_50_17120 = t_50_1712(pf0,len0,n0)
    t_start0 = t_s(pf0,len0,n0)
    t_whole0 = t_w(pf0,len0,n0)
    v_x0_s0 = v_x0_s(pf0,len0,n0)
    v_y0_s0 = v_y0_s(pf0,len0,n0)
    v_z0_s0 = v_z0_s(pf0,len0,n0)
    r_x_s0 = r_x0_s0(pf0,len0,n0)
    r_y_s0 = r_y0_s0(pf0,len0,n0)
    r_z_s0 = r_z0_s0(pf0,len0,n0)

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
    z0_sz.append(pf0.iat[len0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iat[len0-n0,sz_bot_n0])
    x0_sz.append(-17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iat[len0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iat[len0-n0,sz_top_n0])
    x0_sz.append(17/24)
    y0_sz.append(17/12)
    z0_sz.append(pf0.iat[len0-n0,sz_bot_n0])
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

    xp0 = []
    yp0 = []
    zp0 = []

    if pf0.iat[len0-n0,zone_n0] <= 9:
        bs = 'Strike'
        zone0 = (pf0.iat[len0-n0,zone_n0]-1) % 3 #2
        zone1 = (pf0.iat[len0-n0,zone_n0]-1) // 3 #2
        xp0.append(-17/24+zone0*(17/36))
        yp0.append(17/12)
        zp0.append(pf0.iat[len0-n0,sz_top_n0]-(zone1)*(1/3)*(pf0.iat[len0-n0,sz_top_n0]-pf0.iat[len0-n0,sz_bot_n0]))
        xp0.append(-17/24+(zone0+1)*(17/36))
        yp0.append(17/12)
        zp0.append(pf0.iat[len0-n0,sz_top_n0]-(zone1)*(1/3)*(pf0.iat[len0-n0,sz_top_n0]-pf0.iat[len0-n0,sz_bot_n0]))
        xp0.append(-17/24+(zone0+1)*(17/36))
        yp0.append(17/12)
        zp0.append(pf0.iat[len0-n0,sz_top_n0]-(zone1+1)*(1/3)*(pf0.iat[len0-n0,sz_top_n0]-pf0.iat[len0-n0,sz_bot_n0]))
        xp0.append(-17/24+zone0*(17/36))
        yp0.append(17/12)
        zp0.append(pf0.iat[len0-n0,sz_top_n0]-(zone1+1)*(1/3)*(pf0.iat[len0-n0,sz_top_n0]-pf0.iat[len0-n0,sz_bot_n0]))
        fig_0.add_trace(go.Mesh3d(
            x=xp0,
            y=yp0,
            z=zp0,
            i=[0, 1, 2, 0],
            j=[1, 2, 3, 3],  
            k=[2, 3, 0, 1], 
            color='lightblue',
            opacity=0.2)
        )
    elif pf0.iat[len0-n0,zone_n0] >= 11:
        bs = 'Ball'
        if pf0.iat[len0-n0,zone_n0] == 11:
            xp0.append(-17/24-1/3)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(-17/24)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(-17/24)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0]+1/3)
            xp0.append(-17/24-1/3)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0]+1/3)
        if pf0.iat[len0-n0,zone_n0] == 12:
            xp0.append(17/24+1/3)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(17/24)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(17/24)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0]+1/3)
            xp0.append(17/24+1/3)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_top_n0]+1/3)
        if pf0.iat[len0-n0,zone_n0] == 13:
            xp0.append(-17/24-1/3)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(-17/24)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(-17/24)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0]-1/3)
            xp0.append(-17/24-1/3)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0]-1/3)
        if pf0.iat[len0-n0,zone_n0] == 14:
            xp0.append(17/24+1/3)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(17/24)
            yp0.append(17/12)
            zp0.append((pf0.iat[len0-n0,sz_top_n0]+pf0.iat[len0-n0,sz_bot_n0])/2)
            xp0.append(17/24)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0])
            xp0.append(0)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0]-1/3)
            xp0.append(17/24+1/3)
            yp0.append(17/12)
            zp0.append(pf0.iat[len0-n0,sz_bot_n0]-1/3)
        fig_0.add_trace(go.Mesh3d(
            x=xp0,
            y=yp0,
            z=zp0,
            i = [0, 2, 4, 5], 
            j = [1, 3, 5, 0], 
            k = [2, 4, 2, 2], 
            color='lightblue',
            opacity=0.3)
        )

    x_ball0 = []
    y_ball0 = []
    z_ball0 = []

    x_cen = pf0.iat[len0-n0,plate_x_n0]
    z_cen = pf0.iat[len0-n0,plate_z_n0]
    for t in frange(0, 2*pi + pi/20, pi/20):
        x_ball0.append(x_cen+0.111549*cos(t))
        y_ball0.append(17/12)
        z_ball0.append(z_cen+0.111549*sin(t))
    fig_0.add_trace(go.Scatter3d(
        x=x_ball0,
        y=y_ball0,
        z=z_ball0,
        mode='lines',
        line=dict(
            color='#243250',
            width=3
        ),
        opacity=1,
        name = 'Passage Course'
        )
    )


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
        height=700,
        width=1000,
        scene_aspectmode = 'manual',
        legend=dict(
            xanchor='left',
            yanchor='top',
            x=0.01,
            y=1,
            orientation='h',
        ),
        title='Passage Course'
    )
    
    alpha = (2*pi)*((180+(pf0.iat[len0-n0,s_a_n0]))/360)
    if pf0.iat[len0-n0,p_th_n0] == 'R':
        beta = -math.acos(float(active_spin)/100)
    else:
        beta = math.acos(float(active_spin)/100)

    xa = []
    ya = []
    za = []

    interval = (2*pi)/60
    for t in frange(0, 2*pi+interval, 2*interval):  # meridians:
        xa.append(1.1*(cos(t)+0.1*sin(t)))
        ya.append(1.1*((cos(alpha)*(sin(t)-0.1*cos(t))+0.1*sin(alpha))))
        za.append(1.1*(0.1*cos(alpha)-(sin(t)-0.1*cos(t))*sin(alpha)))
        xa.append(1.1*(cos(t)))
        ya.append(1.1*(sin(t)*cos(alpha)))
        za.append(1.1*(-sin(t)*sin(alpha)))
        xa.append(1.1*(cos(t)+0.1*sin(t)))
        ya.append(1.1*((cos(alpha))*(sin(t)-0.1*cos(t))-0.1*sin(alpha)))
        za.append(1.1*(-0.1*cos(alpha)-(sin(t)-0.1*cos(t))*sin(alpha)))

        xa.append(1.1*(cos(t+interval)+0.1*sin(t+interval)))
        ya.append(1.1*((cos(alpha))*(sin(t+interval)-0.1*cos(t+interval))-0.1*sin(alpha)))
        za.append(1.1*(-0.1*cos(alpha)-(sin(t+interval)-0.1*cos(t+interval))*sin(alpha)))
        xa.append(1.1*(cos(t+interval)))
        ya.append(1.1*(sin(t+interval)*cos(alpha)))
        za.append(1.1*(-sin(t+interval)*sin(alpha)))
        xa.append(1.1*(cos(t+interval)+0.1*sin(t+interval)))
        ya.append(1.1*((cos(alpha)*(sin(t+interval)-0.1*cos(t+interval))+0.1*sin(alpha))))
        za.append(1.1*(0.1*cos(alpha)-(sin(t+interval)-0.1*cos(t+interval))*sin(alpha)))

        xa.append(1.1*(cos(t)+0.1*sin(t)))
        ya.append(1.1*((cos(alpha)*(sin(t)-0.1*cos(t))+0.1*sin(alpha))))
        za.append(1.1*(0.1*cos(alpha)-(sin(t)-0.1*cos(t))*sin(alpha)))

        xa.append([None])
        ya.append([None])
        za.append([None])


    xb = []
    yb = []
    zb = []

    xb.append(0)
    yb.append(1.4*(sin(alpha)))
    zb.append(1.4*(cos(alpha)))

    xb.append(0)
    yb.append(1.4*(-sin(alpha)))
    zb.append(1.4*(-cos(alpha)))

    xb.append([None])
    yb.append([None])
    zb.append([None])

    fig.add_surface(x=xs, y=ys, z=zs, 
                    colorscale=[[0, '#ffffff' ], [1, '#9D9087']],
                    showscale=False, opacity=0.5)  # or opacity=1
    fig.add_scatter3d(x=xa, y=ya, z=za, mode='lines', line_width=3, line_color='#B22D35')
    fig.add_scatter3d(x=xb, y=yb, z=zb, mode='lines', line_width=10, line_color='#4C444D')
    fig.update_layout(width=10, height=500)


    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=1.3*cos(beta), y=0, z=1.3*sin(beta)),  # 視点の位置
                up=dict(x=0, y=0, z=0)  # 上方向の向き
            ),
            xaxis=dict(
            tickangle=0 
            ),
            zaxis=dict(
            tickangle=0  
            ),
            dragmode=False,
            xaxis_showgrid=False,   
            yaxis_showgrid=False,
            zaxis_showgrid=False,   
            xaxis_visible=False,    
            yaxis_visible=False,
            zaxis_visible=False
        ),
        showlegend=False
    )

    if str(pf0.iat[len0-n0,inntb_n0]) == 'Top':
        index5_1 = 'Pitcher'
        index6_1 = 'Batter'
    else:
        index5_1 = 'Batter'
        index6_1 = 'Pitcher'
    index1 = str(pf0.iat[len0-n0,r_s_n0])
    index2 = str(pf0.iat[len0-n0,r_s_r_n0])
    index3_0 = pf0.iat[len0-n0,p_t_n0]
    index3_1 = str(pf0.iat[len0-n0,p_n_n0])
    index4_0 = int(pf0.iat[len0-n0,zone_n0])
    index4_1 = bs
    index5_0 = str(pf0.iat[len0-n0,h_t_n0])
    index6_0 = str(pf0.iat[len0-n0,a_t_n0])
    ok = 1

###表示

fig_0.update_layout(
        title=dict(
            text=('<span style = "display: flex;" ><span style = "font-size: 45px;">Pitch Trajectory</span><br><br><span  style = "color: orange; font-size: 24px; margin-top: 10px;"><a href="https://x.com/empty8128">empty8128 (𝕏/Twitter)</a></span></span>'),
        ),
)

st.markdown("""
    <style>
        .con0{
            display: flex;
            align-items: center;
            }
        .con1{
            font-size: 24px;
            color: black !impoortant;
            }
        .con2{
            font-size: 12px;
            color: orange !impoortant;
            }
    </style>
""", unsafe_allow_html=True)

st.plotly_chart(fig_0, key="unique_key_1")
if active_spin == -1:
    st.sidebar.markdown("This data displays only pitches from 2020 to 2024.このデータは2020年から2024年のピッチのみを表示しています。")
else:
    st.sidebar.plotly_chart(fig, key="unique_key_2")

if ok == 1:
    st.markdown(f'<div class="whole"><div class="title0"><div class="title1">SPEED</div></div><ul class="list"><li class="list1">{index1}</li><li class="list2">(MPH)</li></ul></div><div class="whole"><div class="title0"><div class="title1">RPM</div></div><ul class="list"><li class="list1">{index2}</li><li class="list2">(/s)</li></ul></div><div class="whole"><div class="title0"><div class="title1">VAA</div></div><ul class="list"><li class="list1">{vaa0}</li><li class="list2">(°)</li></ul></div><div class="whole"><div class="title0"><div class="title1">HAA</div></div><ul class="list"><li class="list1">{haa0}</li><li class="list2">(°)</li></ul></div><div class="whole"><div class="title0"><div class="title1">Pitch Type</div></div><ul class="list"><li class="list1">{index3_0}</li><li class="list2">{index3_1}</li></ul></div><div class="whole"><div class="title0"><div class="title1">ZONE</div></div><ul class="list"><li class="list1">{index4_0}</li><li class="list2">{index4_1}</li></ul></div><div class="whole"><div class="title0"><div class="title1">HOME</div></div><ul class="list"><li class="list1">{index5_0}</li><li class="list2">{index5_1}</li></ul></div><div class="whole"><div class="title0"><div class="title1">AWAY</div></div><ul class="list"><li class="list1">{index6_0}</li><li class="list2">{index6_1}</li></ul></div>',unsafe_allow_html=True)

st.sidebar.markdown('This data is still in the verification phase. Please contact me if you know more about this.ここのデータはまだ検証段階です。詳しい方は連絡をいただけると助かります。')
#<div class="whole"><div class="title0"><div class="title1">SPEED</div></div><ul class="list"><li class="list1">{index1}</li><li class="list2">(MPH)</li></ul></div>

st.markdown("""
    <style>
        .whole{
            border: solid 1px;
            border-radius: 10px 10px 10px 10px;
            width:152px;
            float: left;
            margin: 10px;
            }
        .title0{
            background:#D0B8BB !important;
            width:150px;
            height:50px;
            border-radius: 10px 10px 0px 0px;
            display: table;
            width: 100%;
            text-align: center;
            }
        .title1{
            display: table-cell;
            vertical-align: middle;
            font-size: 24px;
            color: black;
        }
        .list{
            background:#ffffff !important;
            height:100px;
            width:150px;
            margin: 0px;
            border-radius: 0px 0px 10px 10px;
            vertical-align: middle;
        }
        .list1 {
            margin: 0px !important;
            padding: 0px !important;
            list-style:none;
            font-size: 40px !important;
            text-align: center;
            color: black;
        }
        .list2 {
            margin: 0px !important;
            padding: 0px !important;
            list-style:none;
            font-size: 15px !important;
            text-align: center;
            color: black;
        }
        body {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)


#streamlit run app.py