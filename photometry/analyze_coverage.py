# Analyze the coverage of the available observations

from pickle import MARK
from dash import Dash, dcc, html
import plotly.express as px
import numpy as np
import pandas as pd

COMPARISONS = (
    ('Mu', 'Mean Phase'),
    # ('Mu0', 'Mean Phase'),
    # ('Mu', 'Mu0'),

    # ('Mean Emission', 'Mean Phase'),
    # ('Incidence', 'Mean Phase'),
    # ('Incidence', 'Mean Emission'),

    # ('Date', 'Mean Phase'),
    # ('Date', 'Mu'),
    # ('Date', 'Mu0'),
    # ('Date', 'Mean Emission'),
    # ('Date', 'Incidence'),

    # ('Date', 'Mean Longitude')
)

RANGES = {'Mu': (0,1),
          'Mu0': (0,1),
          'Mean Phase': (0,180),
          'Mean Emission': (0,180),
          'Incidence': (50,90)}

COLUMNS = 3
ROW_STYLE = {'display': 'inline-block', 'width': '33%'}
PAPER_COLOR = '#000000'
BG_COLOR = '#202020'
MARKER_COLOR = '#00a0a0'
MARKER_SIZE = 2
FONT_COLOR = '#ffffff'

app = Dash(__name__)

obsdata = pd.read_csv('../data_files/good_qual_1deg.csv', parse_dates=['Date'])

print('** SUMMARY STATISTICS **')
print('Unique observation names:', len(obsdata.groupby('Observation')))
print('Total 1-degree slices:', len(obsdata))
print('Starting date:', obsdata['Date'].min())
print('Ending date:', obsdata['Date'].max())
print('Time span:', obsdata['Date'].max()-obsdata['Date'].min())
obsdata['Mu'] = np.abs(np.cos(np.radians(obsdata['Mean Emission'])))
obsdata['Mu0'] = np.abs(np.cos(np.radians(obsdata['Incidence'])))
obsdata['Mean Longitude'] = (obsdata['Min Long']+obsdata['Max Long'])/2
obsdata['Marker Color'] = MARKER_COLOR
obsdata['Marker Size'] = MARKER_SIZE

rows = []
cur_row = []

for plot_num, (src1, src2) in enumerate(COMPARISONS):

    fig = px.scatter(obsdata, x=src1, y=src2,
                     hover_name='Observation',
                     hover_data=('Date',))
    fig.update_traces(marker={'color': MARKER_COLOR,
                              'size': MARKER_SIZE})
    fig.update_layout(paper_bgcolor=PAPER_COLOR,
                      plot_bgcolor=BG_COLOR,
                      font_color=FONT_COLOR)
    fig_id = f'plot{plot_num}'

    div = html.Div(children=[
            dcc.Graph(
                id=fig_id,
                figure=fig
            )
        ], style=ROW_STYLE)
    cur_row.append(div)

    if (plot_num % COLUMNS) == COLUMNS-1:
        print('Yes')
        rows.append(html.Div(children=cur_row))
        cur_row = []

if cur_row:
    rows.append(html.Div(children=cur_row))

print(rows)
app.layout = html.Div(children=rows)

app.run_server()
