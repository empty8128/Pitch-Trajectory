import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pybaseball import statcast_pitcher
from pybaseball import statcast_batter
from pybaseball import playerid_lookup
import streamlit as st
import dash
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

df = pd.read_csv('streamlit/example.csv')
vars_cat = [var for var in sorted(pitching_stats('2018', qual=1)['Name'].unique())]
st.set_page_config(layout="wide")

fig_0 = go.Figure()
st.plotly_chart(fig_0, use_container_width=True)