#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

from qualtran.surface_code import ccz2t_cost_model, fifteen_to_one, magic_state_factory
from qualtran.surface_code import quantum_error_correction_scheme_summary as qecs
from qualtran.surface_code import rotation_cost_model
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.azure_cost_model import code_distance, logical_qubits, minimum_time_steps
from qualtran.surface_code.ccz2t_cost_model import CCZ2TFactory, get_ccz2t_costs_from_error_budget
from qualtran.surface_code.magic_count import MagicCount
from qualtran.surface_code.multi_factory import MultiFactory

app = Dash(__name__)


def get_objects(modules, object_type):
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj_name, obj


QEC_SCHEMES = dict(get_objects([qecs], qecs.QuantumErrorCorrectionSchemeSummary))
MAGIC_FACTORIES = dict(
    get_objects([fifteen_to_one, ccz2t_cost_model], magic_state_factory.MagicStateFactory)
)
ROTATION_MODELS = dict(get_objects([rotation_cost_model], rotation_cost_model.RotationCostModel))


def input_components():
    return [
        html.Table(
            [
                html.Tr(
                    [
                        html.Td('Enter physical error rate'),
                        dcc.Input(
                            debounce=True,
                            id='physical_error_rate',
                            type='number',
                            min=0,
                            max=1,
                            style={'marginLeft': '10px'},
                            value=1e-4,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Enter error budget'),
                        dcc.Input(
                            debounce=True,
                            id='error_budget',
                            type='number',
                            min=0,
                            max=1,
                            style={'marginLeft': '10px'},
                            value=1e-3,
                        ),
                    ]
                ),
            ]
        ),
        html.P("Select Estimation Cost Model:"),
        dcc.RadioItems(
            id='estimation_model',
            options=['GidneyFolwer (arxiv:1812.01238)', 'Beverland et al (arxiv:2211.07629)'],
            value='GidneyFolwer (arxiv:1812.01238)',
        ),
        html.P('Enter a summary of the quantum algorithm/circuit'),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td('Qubits'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'qubits'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Number of Qubits",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Measurements'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'Measurements'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Number of Measurements",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Ts'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'Ts'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Number of T gates",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Toffolis'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'Toffolis'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Number of Toffolis",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Rotations'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'Rotations'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Number of Rotations",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Rotation Circuit Depth'),
                        dcc.Input(
                            debounce=True,
                            id={'type': 'algorithm', 'property': 'Rotation Depth'},
                            type='number',
                            min=0,
                            style={'marginLight': '10px'},
                            placeholder="Rotation Circuit Depth",
                            value=10,
                        ),
                    ]
                ),
            ]
        ),
        html.P("Select a QEC algorithm:"),
        dcc.RadioItems(id='QEC', options=sorted(QEC_SCHEMES), value=next(iter(QEC_SCHEMES.keys()))),
        html.Details(
            [
                html.Summary("QECs' information"),
                html.Table(
                    [
                        html.Tr([html.Td(k), html.Td(str(v), style={'padding': '10px'})])
                        for k, v in QEC_SCHEMES.items()
                    ]
                ),
            ]
        ),
        html.P("Enter number of magic state factories:"),
        dcc.RadioItems(
            id='magic_factories',
            options=sorted(MAGIC_FACTORIES),
            value=next(iter(MAGIC_FACTORIES.keys())),
        ),
        dcc.Input(debounce=True, id='magic_factories_number', type='number', min=0, value=0),
        html.Details(
            [
                html.Summary("Magic Factories' information"),
                html.Table(
                    [
                        html.Tr([html.Td(k), html.Td(str(v), style={'padding': '10px'})])
                        for k, v in MAGIC_FACTORIES.items()
                    ]
                ),
            ]
        ),
        html.P("Select Rotation Cost Model:"),
        dcc.RadioItems(
            id='rotation-cost-model',
            options=sorted(ROTATION_MODELS),
            value=next(iter(ROTATION_MODELS.keys())),
        ),
        html.Details(
            [
                html.Summary("Models' information"),
                html.Table(
                    [
                        html.Tr([html.Td(r), html.Td(str(v), style={'padding': '10px'})])
                        for r, v in ROTATION_MODELS.items()
                    ]
                ),
            ]
        ),
    ]


def create_ouputs():
    return [
        html.Table(
            [
                html.Tr(
                    [
                        html.Td(id='magic-name', children=['Total Number of Toffoli gates']),
                        dcc.Input(id='total_magic', type="text", value='', readOnly=True),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Minimum Number of magic factories'),
                        dcc.Input(id='min_factory_count', value='', readOnly=True),
                    ]
                ),
                html.Tr(
                    [
                        html.Td('Duration'),
                        html.Td(dcc.Input(id='duration', value='', readOnly=True)),
                    ],
                    id='duration-container',
                ),
            ]
        )
    ]


app.layout = html.Div(
    [
        html.H4('Interactive Quantum Resource Estimation'),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td(input_components(), style={'width': '30%'}),
                        html.Td(create_ouputs(), style={'width': '20%'}),
                        html.Td(
                            dcc.Loading(
                                dcc.Graph(id="qubit-pie-chart"),
                                type="cube",
                                style={'max-width': '80%', 'float': 'right'},
                            )
                        ),
                    ]
                )
            ],
            style={'width': '100%'},
        ),
        html.Div(
            id='runtime-container',
            children=[
                dcc.Loading(dcc.Graph(id="runtime-plot"), type="cube", style={'max-width': '80%'})
            ],
        ),
        html.P(id='hidden-p', hidden=True),
    ]
)


def parse_float(s):
    return float(s) if s else 0


@app.callback(
    Output("qubit-pie-chart", "figure"),
    Input({'type': 'algorithm', 'property': 'qubits'}, "value"),
    Input('magic_factories', 'value'),
    Input('magic_factories_number', 'value'),
    Input('estimation_model', 'value'),
    Input({'type': 'algorithm', 'property': 'Ts'}, "value"),
    Input({'type': 'algorithm', 'property': 'Toffolis'}, "value"),
    Input({'type': 'algorithm', 'property': 'Rotations'}, "value"),
    Input('QEC', 'value'),
    Input('rotation-cost-model', 'value'),
    Input('physical_error_rate', 'value'),
    Input('error_budget', 'value'),
)
def update_qubits(
    qubits,
    magic_factory,
    magic_count,
    estimation_model,
    ts,
    Toffolis,
    rotations,
    qec_name,
    rotation_name,
    physical_error_rate,
    error_budget,
):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        res = get_ccz2t_costs_from_error_budget(
            n_magic=MagicCount(n_t=parse_float(ts), n_ccz=parse_float(Toffolis)),
            n_algo_qubits=parse_float(qubits),
            phys_err=parse_float(physical_error_rate),
            error_budget=parse_float(error_budget),
            factory=MultiFactory(CCZ2TFactory(), int(parse_float(magic_count))),
        )
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [
            res.footprint - MAGIC_FACTORIES['GidneyFowlerCCZ'].footprint(),
            MAGIC_FACTORIES['GidneyFowlerCCZ'].footprint(),
        ]
        fig = px.pie(
            memory_footprint, values='qubits', names='source', title='Physical Qubit Utilization'
        )
        fig.update_traces(textinfo='value')
        return fig
    else:
        algorithm = AlgorithmSummary(algorithm_qubits=parse_float(qubits))
        magic_counts = parse_float(magic_count)
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [
            logical_qubits(algorithm),
            MAGIC_FACTORIES[magic_factory].footprint() * magic_counts,
        ]
        fig = px.pie(
            memory_footprint, values='qubits', names='source', title='Physical Qubit Utilization'
        )
        fig.update_traces(textinfo='value')
        return fig


@app.callback(
    Output("total_magic", "value"),
    Input({'type': 'algorithm', 'property': 'Toffolis'}, "value"),
    Input({'type': 'algorithm', 'property': 'Ts'}, "value"),
    Input({'type': 'algorithm', 'property': 'Rotations'}, "value"),
    Input('rotation-cost-model', 'value'),
    Input('estimation_model', 'value'),
)
def update_total_number_t_gates(toffs, ts, rots, rotation_model_name, estimation_model):
    rotation_cost = ROTATION_MODELS[rotation_model_name].rotation_cost(parse_float(rots))
    ts = (
        parse_float(ts)
        + rotation_cost.t_gates
        + 4 * (parse_float(toffs) + rotation_cost.toffoli_gates)
    )
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return ts / 4
    else:
        return ts


def format_duration(duration):
    unit = 'us'
    if duration[0] > 1000:
        duration = [d / 1000 for d in duration]
        unit = 'ms'
    if duration[0] > 1000:
        duration = [d / 1000 for d in duration]
        unit = 's'
    if duration[0] > 60:
        duration = [d / 60 for d in duration]
        unit = 'min'
    if duration[0] > 60:
        duration = [d / 60 for d in duration]
        unit = 'hours'
    if duration[0] > 24:
        duration = [d / 24 for d in duration]
        unit = 'days'
    return unit, duration


@app.callback(
    Output('min_factory_count', 'value'),
    Input('magic_factories', 'value'),
    Input({'type': 'algorithm', 'property': ALL}, "value"),
    Input('error_budget', 'value'),
    Input('rotation-cost-model', 'value'),
    Input('physical_error_rate', 'value'),
    Input('magic_factories_number', 'value'),
    Input('estimation_model', 'value'),
)
def update_magic_count(
    magic_factory,
    algorithm_data,
    error_budget,
    rotation_model_name,
    physical_error_rate,
    cv,
    estimation_model,
):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return '1'
    _, _, ts, toffs, rots, _ = map(parse_float, algorithm_data)
    rotation_model = ROTATION_MODELS[rotation_model_name]
    algorithm = AlgorithmSummary(*map(parse_float, algorithm_data))
    c_min = minimum_time_steps(
        error_budget=parse_float(error_budget), alg=algorithm, rotation_model=rotation_model
    )
    factory = MAGIC_FACTORIES[magic_factory]
    min_c = int(
        factory.n_cycles(
            MagicCount(
                n_ccz=toffs,
                n_t=ts + rots * rotation_model.rotation_cost(error_budget / (3 * rots)).t_gates,
            ),
            parse_float(physical_error_rate),
        )
        / c_min
        + 0.5
    )
    return f'{max(parse_float(cv), min_c):g}'


@app.callback(
    Output('magic_factories_number', 'value'),
    Input('magic_factories', 'value'),
    Input({'type': 'algorithm', 'property': ALL}, "value"),
    Input('error_budget', 'value'),
    Input('rotation-cost-model', 'value'),
    Input('physical_error_rate', 'value'),
    Input('estimation_model', 'value'),
)
def update_min_magic_count(
    magic_factory,
    algorithm_data,
    error_budget,
    rotation_model_name,
    physical_error_rate,
    estimation_model,
):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return 1
    _, _, ts, toffs, rots, _ = map(parse_float, algorithm_data)
    rotation_model = ROTATION_MODELS[rotation_model_name]
    algorithm = AlgorithmSummary(*map(parse_float, algorithm_data))
    c_min = minimum_time_steps(
        error_budget=parse_float(error_budget), alg=algorithm, rotation_model=rotation_model
    )
    factory = MAGIC_FACTORIES[magic_factory]
    return int(
        factory.n_cycles(
            MagicCount(
                n_ccz=toffs,
                n_t=ts + rots * rotation_model.rotation_cost(error_budget / (3 * rots)).t_gates,
            ),
            parse_float(physical_error_rate),
        )
        / c_min
        + 0.5
    )


@app.callback(
    Output("runtime-plot", "figure"),
    Input('error_budget', 'value'),
    Input('rotation-cost-model', 'value'),
    Input('QEC', 'value'),
    Input('physical_error_rate', 'value'),
    Input({'type': 'algorithm', 'property': ALL}, "value"),
    Input('estimation_model', 'value'),
)
def update_runtime_plot(
    error_budget,
    rotation_model_name,
    qec_scheme_name,
    physical_error_rate,
    algorithm_data,
    estimation_model,
):
    rotation_model = ROTATION_MODELS[rotation_model_name]
    qec = QEC_SCHEMES[qec_scheme_name]
    algorithm = AlgorithmSummary(*map(parse_float, algorithm_data))
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return go.Figure()
    else:
        c_min = minimum_time_steps(
            error_budget=parse_float(error_budget), alg=algorithm, rotation_model=rotation_model
        )
        c_max = 10 * c_min
        time_steps = np.linspace(c_min, c_max, 10)
        cds = [
            code_distance(
                error_budget=parse_float(error_budget),
                time_steps=t,
                alg=algorithm,
                qec=qec,
                physical_error_rate=parse_float(physical_error_rate),
            )
            for t in time_steps
        ]
        duration = [qec.error_detection_circuit_time_us(d) * t for t, d in zip(time_steps, cds)]
        unit, duration = format_duration(duration)
        name = f'Duration ({unit})'
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=time_steps, y=duration, name=name))
        fig.add_trace(go.Scatter(x=time_steps, y=cds, name='code_distance'))
        fig.update_layout(
            title_text="Code Distance and Duration vs Code Cycles", xaxis_title="Code Cycles"
        )
        return fig


@app.callback(Output('runtime-container', 'style'), Input('estimation_model', 'value'))
def toggle_runtime_plot(estimation_model):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(Output('magic-name', 'children'), Input('estimation_model', 'value'))
def toggle_magic_name(estimation_model):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return ['Total Number of Toffoli gates']
    else:
        return ['Total Number of T gates']


@app.callback(Output('duration-container', 'style'), Input('estimation_model', 'value'))
def toggle_duration_container(estimation_model):
    if estimation_model == 'GidneyFolwer (arxiv:1812.01238)':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output("duration", "value"),
    Input({'type': 'algorithm', 'property': ALL}, "value"),
    Input('error_budget', 'value'),
    Input('physical_error_rate', 'value'),
    Input('rotation-cost-model', 'value'),
    Input('magic_factories_number', 'value'),
    Input({'type': 'algorithm', 'property': 'qubits'}, "value"),
)
def update_duration(
    algorithm_data, error_budget, physical_error_rate, rotation_model_name, magic_count, qubits
):
    rotation_model = ROTATION_MODELS[rotation_model_name]
    algorithm = AlgorithmSummary(*map(parse_float, algorithm_data))
    rotation_cost = rotation_model.rotation_cost(error_budget / (3 * algorithm.rotation_gates))
    res = get_ccz2t_costs_from_error_budget(
        n_magic=MagicCount(
            n_t=algorithm.t_gates + rotation_cost.t_gates,
            n_ccz=algorithm.toffoli_gates + rotation_cost.toffoli_gates,
        ),
        n_algo_qubits=parse_float(qubits),
        phys_err=parse_float(physical_error_rate),
        error_budget=parse_float(error_budget),
        factory=MultiFactory(CCZ2TFactory(), int(parse_float(magic_count))),
    )
    unit, duration = format_duration([res.duration_hr * 60 * 60 * 10**6])
    return f'{duration[0]:g} {unit}'


if __name__ == '__main__':
    app.run_server(debug=True)
app.run_server(debug=True)
