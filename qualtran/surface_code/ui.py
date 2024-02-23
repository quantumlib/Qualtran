from qualtran.surface_code import quantum_error_correction_scheme_summary as qecs
from qualtran.surface_code import fifteen_to_one, ccz2t_cost_model, magic_state_factory
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code import rotation_cost_model
from qualtran.surface_code.azure_cost_model import logical_qubits, minimum_time_steps, code_distance

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, ALL
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



app = Dash(__name__)

def get_objects(modules, object_type):
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj

algorithm = AlgorithmSummary(10, 10, 10, 10, 10, 10)
qec_schemes = tuple(get_objects([qecs], qecs.QuantumErrorCorrectionSchemeSummary))
magic_factories = tuple(get_objects([fifteen_to_one, ccz2t_cost_model], magic_state_factory.MagicStateFactory))
rotation_models = tuple(get_objects([rotation_cost_model], rotation_cost_model.RotationCostModel))

app.layout = html.Div([
    html.H4('Interactive Quantum Resource Estimation'),
    html.Table(
        [
            html.Tr([html.Td('Enter physical error rate'),  dcc.Input(id='physical_error_rate', type='number', min=0, max=1, style={'marginLeft':'10px'}, value=1e-4)]),
            html.Tr([html.Td('Enter error budget'), dcc.Input(id='error_budget', type='number', min=0, max=1, style={'marginLeft':'10px'}, value=1e-3)]),
        ]
    ),
    
    html.P('Enter a summary of the quantum algorithm/circuit'),
    html.Table(
        [
            html.Tr([html.Td('Qubits'), dcc.Input(id={'type':'algorithm', 'property': 'qubits'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Number of Qubits", value=10)]),
            html.Tr([html.Td('Measurements'), dcc.Input(id={'type':'algorithm', 'property': 'Measurements'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Number of Measurements", value=10)]),
            html.Tr([html.Td('Toffolis'), dcc.Input(id={'type':'algorithm', 'property': 'Toffolis'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Number of Toffolis", value=10)]),
            html.Tr([html.Td('Ts'), dcc.Input(id={'type':'algorithm', 'property': 'Ts'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Number of T gates", value=10)]),
            html.Tr([html.Td('Rotations'), dcc.Input(id={'type':'algorithm', 'property': 'Rotations'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Number of Rotations", value=10)]),
            html.Tr([html.Td('Rotation Circuit Depth'), dcc.Input(id={'type':'algorithm', 'property': 'Rotation Depth'}, type='number', min=0, style={'marginLight':'10px'}, placeholder="Rotation Circuit Depth", value=10)]),
        ]
    ),

    html.P("Select a QEC algorithm:"),
    dcc.RadioItems(
        id='QEC',
        options=[str(qec) for qec in qec_schemes],
        value=str(qec_schemes[0]),
    ),
    html.P("Enter number of magic state factories:"),
    html.Table(
        [html.Tr([html.Td(str(f)), dcc.Input(id={'type':'magic-factory', 'index':i}, type='number', min=0, style={'marginLeft':'10px'}, value=0)]) for i, f in enumerate(magic_factories)]
    ),
    html.P("Select Rotation Cost Model:"),
    dcc.RadioItems(
        id='RotationCostModel',
        options=[str(rcm) for rcm in rotation_models],
        value=str(rotation_models[0]),
    ),
    dcc.Loading(dcc.Graph(id="qubit-pie-chart"), type="cube"),
    dcc.Loading(dcc.Graph(id="runtime-plot"), type="cube"),
    html.P(id='hidden-p', hidden=True)
])


def parse_float(s):
    return float(s) if s else 0

@app.callback(
        Output('hidden-p', 'title'),
        Input({'type':'algorithm', 'property': ALL}, "value"),
)
def create_algorithm_summary(values):
    global algorithm
    qs, ms, toffs, ts, rs, rd = values
    algorithm = AlgorithmSummary(algorithm_qubits=parse_float(qs), measurements=parse_float(ms), toffoli_gates=parse_float(toffs), t_gates=parse_float(ts), rotation_gates=parse_float(rs), rotation_circuit_depth=parse_float(rd))
    return ''

memory_footprint = pd.DataFrame(
    [['logical qubits', 0],
    ['Magic State Distillation', 0]],
    columns=['source', 'qubits']
)

@app.callback(
    Output("qubit-pie-chart", "figure"), 
    Input({'type':'algorithm', 'property': 'qubits'}, "value"),
    Input({'type':'magic-factory', 'index':ALL}, 'value'),
)
def update_qubits(qubits, i_magic_counts):
    global memory_footprint
    magic_counts = [parse_float(c) for c in i_magic_counts]
    if sum(v != 0 for v in magic_counts) > 1:
        memory_footprint = pd.DataFrame(
            columns=['source', 'qubits']
        )
        memory_footprint['source'] = ['logical qubits'] + [str(f) for f, c in zip(magic_factories, magic_counts) if c != 0]
        memory_footprint['qubits'] = [logical_qubits(algorithm)] + [f.footprint()*c for f, c in zip(magic_factories, magic_counts) if c != 0]     
    else:
        memory_footprint = pd.DataFrame(
            columns=['source', 'qubits']
        )
        memory_footprint['source'] = ['logical qubits', 'Magic State Distillation']
        memory_footprint['qubits'] = [logical_qubits(algorithm), sum(f.footprint()*c for f, c in zip(magic_factories, magic_counts))]
    fig = px.pie(memory_footprint, values='qubits', names='source', title='Qubit Utilization')
    fig.update_traces(textinfo='value')
    return fig

@app.callback(
    Output("runtime-plot", "figure"), 
    Input('error_budget', 'value'),
    Input('RotationCostModel', 'value'),
    Input('QEC', 'value'),
    Input('physical_error_rate', 'value'),
    Input({'type':'algorithm', 'property': ALL}, "value"),
)
def update_runtime_plot(error_budget, rotation_model_name, qec_scheme_name, physical_error_rate, algorithm_data):
    rotation_model = [model for model in rotation_models if str(model) == rotation_model_name][0]
    qec = [qec for qec in qec_schemes if str(qec) == qec_scheme_name][0]
    c_min = minimum_time_steps(
        error_budget=parse_float(error_budget),
        alg=algorithm,
        rotation_model=rotation_model) * 3
    c_max = 10*c_min
    time_steps = np.linspace(c_min, c_max, 10)
    cds = [code_distance(
        error_budget=parse_float(error_budget),
        time_steps=t,
        alg=algorithm,
        qec=qec,
        physical_error_rate=parse_float(physical_error_rate),
    ) for t in time_steps]
    duration = [
        qec.error_detection_circuit_time_us(d) * t for t, d in zip(time_steps, cds)
    ]
    name = 'duration (us)'
    if duration[-1] > 1000:
        duration = [d / 1000 for d in duration]
        name = 'duration (ms)'
    if duration[-1] > 1000:
        duration = [d / 1000 for d in duration]
        name = 'duration (s)'
    if duration[-1] > 60:
        duration = [d / 60 for d in duration]
        name = 'duration (min)'

    if duration[-1] > 60:
        duration = [d / 60 for d in duration]
        name = 'duration (h)'
    if duration[-1] > 24:
        duration = [d / 24 for d in duration]
        name = 'duration (days)'
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time_steps, y=duration, name=name))
    fig.add_trace(go.Scatter(x=time_steps, y=cds, name='code_distance'))
    fig.update_layout(
        title_text="Code Distance and Duration vs Time Steps",
        xaxis_title="Time Steps",
    )
    return fig
    # df = pd.DataFrame({'time_steps': time_steps, 'code_distance': cds, name: duration})
    # return px.line(df, x='time_steps', y=[name, 'code_distance'], title='Code Distance vs Time Steps')


if __name__ == '__main__':
    app.run_server(debug=True)
app.run_server(debug=True)

