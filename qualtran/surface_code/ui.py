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

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from qualtran.surface_code import ccz2t_cost_model, fifteen_to_one, magic_state_factory
from qualtran.surface_code import quantum_error_correction_scheme_summary as qecs
from qualtran.surface_code import rotation_cost_model
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.azure_cost_model import code_distance, logical_qubits, minimum_time_steps
from qualtran.surface_code.ccz2t_cost_model import (
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)
from qualtran.surface_code.magic_count import MagicCount
from qualtran.surface_code.multi_factory import MultiFactory


def get_objects(modules, object_type):
    """Get all objects of a given type from a list of modules."""
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj_name, obj


_QEC_SCHEMES = dict(get_objects([qecs], qecs.QuantumErrorCorrectionSchemeSummary))
_MAGIC_FACTORIES = dict(
    get_objects([fifteen_to_one, ccz2t_cost_model], magic_state_factory.MagicStateFactory)
)
_ROTATION_MODELS = dict(get_objects([rotation_cost_model], rotation_cost_model.RotationCostModel))

_GIDNEY_FOLWER_MODEL = 'GidneyFolwer (arxiv:1812.01238)'
_BEVERLAND_MODEL = 'Beverland et al (arxiv:2211.07629)'
_SUPPORTED_ESTIMATION_MODELS = [_GIDNEY_FOLWER_MODEL, _BEVERLAND_MODEL]

_ALGORITHM_INPUTS = [Input({'type': 'algorithm', 'property': ALL}, 'value')]
_QEC_INPUTS = [Input('QEC', 'value')]
_MAGIC_FACTORIES_INPUTS = [
    Input('_MAGIC_FACTORIES', 'value'),
    Input('_MAGIC_FACTORIES_number', 'value'),
]
_ROTATION_MODELS_INPUTS = [Input('rotation-cost-model', 'value')]
_ESTIMATION_INPUTS = [Input('estimation_model', 'value')]
_ERROR_INPUTS = [Input('physical_error_rate', 'value'), Input('error_budget', 'value')]

_ALL_INPUTS = [
    *_ERROR_INPUTS,
    *_ESTIMATION_INPUTS,
    *_ALGORITHM_INPUTS,
    *_QEC_INPUTS,
    *_MAGIC_FACTORIES_INPUTS,
    *_ROTATION_MODELS_INPUTS,
]


def algorithm_summary_components():
    return [
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
    ]


def _MAGIC_FACTORIES_components():
    return [
        html.P("Enter number of magic state factories:"),
        dcc.RadioItems(
            id='_MAGIC_FACTORIES',
            options=sorted(_MAGIC_FACTORIES),
            value=next(iter(_MAGIC_FACTORIES.keys())),
        ),
        dcc.Input(debounce=True, id='_MAGIC_FACTORIES_number', type='number', min=1, value=1),
        html.Details(
            [
                html.Summary("Magic Factories' information"),
                html.Table(
                    [
                        html.Tr([html.Td(k), html.Td(str(v), style={'padding': '10px'})])
                        for k, v in _MAGIC_FACTORIES.items()
                    ]
                ),
            ]
        ),
    ]


def qec_summary_components():
    return [
        html.P("Select a QEC algorithm:"),
        dcc.RadioItems(
            id='QEC', options=sorted(_QEC_SCHEMES), value=next(iter(_QEC_SCHEMES.keys()))
        ),
        html.Details(
            [
                html.Summary("QECs' information"),
                html.Table(
                    [
                        html.Tr([html.Td(k), html.Td(str(v), style={'padding': '10px'})])
                        for k, v in _QEC_SCHEMES.items()
                    ]
                ),
            ]
        ),
    ]


def rotation_cost_model_components():
    return [
        html.P("Select Rotation Cost Model:"),
        dcc.RadioItems(
            id='rotation-cost-model',
            options=sorted(_ROTATION_MODELS),
            value=next(iter(_ROTATION_MODELS.keys())),
        ),
        html.Details(
            [
                html.Summary("Models' information"),
                html.Table(
                    [
                        html.Tr([html.Td(r), html.Td(str(v), style={'padding': '10px'})])
                        for r, v in _ROTATION_MODELS.items()
                    ]
                ),
            ]
        ),
    ]


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
            id='estimation_model', options=_SUPPORTED_ESTIMATION_MODELS, value=_GIDNEY_FOLWER_MODEL
        ),
        *algorithm_summary_components(),
        *qec_summary_components(),
        *_MAGIC_FACTORIES_components(),
        *rotation_cost_model_components(),
    ]


def create_ouputs():
    return [
        html.Table(
            [
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td('Total Number of Toffoli gates', id='magic-name'),
                                html.Td(
                                    dcc.Input(
                                        id='total_magic', type="text", value='', readOnly=True
                                    )
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td('#magic factories for min runtime'),
                                html.Td(dcc.Input(id='min_factory_count', value='', readOnly=True)),
                            ],
                            id='magic-factory-container',
                        ),
                        html.Tr(
                            [
                                html.Td('Duration'),
                                html.Td(
                                    dcc.Input(id='duration', type='text', value='', readOnly=True)
                                ),
                            ],
                            id='duration-container',
                        ),
                    ]
                )
            ]
        )
    ]


app = Dash(__name__)
app.layout = html.Div(
    [
        html.H4('Interactive QEC overhead estimation'),
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
    ]
)


def create_qubit_pie_chart(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: AlgorithmSummary,
    magic_factory: magic_state_factory.MagicStateFactory,
    magic_count: int,
    needed_magic: MagicCount,
) -> go.Figure:
    """Create a pie chart of the physical qubit utilization."""
    if estimation_model == _GIDNEY_FOLWER_MODEL:
        res, factory, _ = get_ccz2t_costs_from_grid_search(
            n_magic=needed_magic,
            n_algo_qubits=algorithm.algorithm_qubits,
            phys_err=physical_error_rate,
            error_budget=error_budget,
            factory_iter=[MultiFactory(f, int(magic_count)) for f in iter_ccz2t_factories()],
        )
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [res.footprint - factory.footprint(), factory.footprint()]
        fig = px.pie(
            memory_footprint, values='qubits', names='source', title='Physical Qubit Utilization'
        )
        fig.update_traces(textinfo='value')
        return fig
    else:
        factory = MultiFactory(magic_factory, int(magic_count))
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [logical_qubits(algorithm), factory.footprint()]
        fig = px.pie(
            memory_footprint, values='qubits', names='source', title='Physical Qubit Utilization'
        )
        fig.update_traces(textinfo='value')
        return fig


def format_duration(duration: Sequence[float]) -> Tuple[str, Sequence[float]]:
    """Returns a tuple of the format (unit, duration)

    Finds the best unit to report `duration` and assumes that `duration` is initially in us.
    """
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


def create_runtime_plot(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: AlgorithmSummary,
    qec: qecs.QuantumErrorCorrectionSchemeSummary,
    magic_factory: magic_state_factory.MagicStateFactory,
    magic_count: int,
    rotation_model: rotation_cost_model.RotationCostModel,
    needed_magic: MagicCount,
) -> Tuple[Dict[str, Any], go.Figure]:
    """Creates the runtime figure and decides whether to display it or not.

    Currently displays the runtime plot for the Beverland model only.
    """
    if estimation_model == _GIDNEY_FOLWER_MODEL:
        return {'display': 'none'}, go.Figure()
    factory = MultiFactory(magic_factory, int(magic_count))
    c_min = minimum_time_steps(
        error_budget=error_budget, alg=algorithm, rotation_model=rotation_model
    )
    factory_cycles = factory.n_cycles(needed_magic, physical_error_rate)
    min_num_factories = int(np.ceil(factory_cycles / c_min))
    magic_counts = list(
        1 + np.random.choice(min_num_factories, replace=False, size=min(min_num_factories, 5))
    )
    magic_counts.sort(reverse=True)
    magic_counts = np.array(magic_counts)
    time_steps = np.ceil(factory_cycles / magic_counts)
    magic_counts[0] = min_num_factories
    time_steps[0] = c_min
    cds = [
        code_distance(
            error_budget=error_budget,
            time_steps=t,
            alg=algorithm,
            qec=qec,
            physical_error_rate=physical_error_rate,
        )
        for t in time_steps
    ]
    duration = [qec.error_detection_circuit_time_us(d) * t for t, d in zip(time_steps, cds)]
    unit, duration = format_duration(duration)
    duration_name = f'Duration ({unit})'
    num_qubits = logical_qubits(algorithm) + factory.footprint() * magic_counts
    df = pd.DataFrame(
        {
            'label': [
                f'code distance: {c}<br>time steps: {t:g}<br>num factories: {f}'
                for t, c, f in zip(time_steps, cds, magic_count * magic_counts)
            ],
            duration_name: duration,
            'num qubits': num_qubits,
        }
    )
    fig = px.line(df, x=duration_name, y='num qubits', text='label')
    return {'display': 'block'}, fig


@app.callback(
    Output("qubit-pie-chart", "figure"),
    Output('runtime-container', 'style'),
    Output("runtime-plot", "figure"),
    Output('magic-name', 'children'),
    Output("total_magic", "value"),
    Output('magic-factory-container', 'style'),
    Output('min_factory_count', 'value'),
    Output('duration-container', 'style'),
    Output("duration", "value"),
    *_ALL_INPUTS,
)
def update(
    physical_error_rate,
    error_budget,
    estimation_model,
    algorithm_data,
    qec_name,
    magic_name,
    magic_count,
    rotaion_model_name,
):
    """Updates the visualization."""
    if any(x is None for x in [physical_error_rate, error_budget, *algorithm_data, magic_count]):
        raise PreventUpdate
    algorithm = AlgorithmSummary(*algorithm_data)
    qec = _QEC_SCHEMES[qec_name]
    magic_factory = _MAGIC_FACTORIES[magic_name]
    rotation_model = _ROTATION_MODELS[rotaion_model_name]
    needed_magic = algorithm.to_magic_count(rotation_model, error_budget / 3)
    magic_count = int(magic_count)
    return (
        create_qubit_pie_chart(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            magic_factory,
            magic_count,
            needed_magic,
        ),
        *create_runtime_plot(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            qec,
            magic_factory,
            magic_count,
            rotation_model,
            needed_magic,
        ),
        *total_magic(estimation_model, needed_magic),
        *min_num_factories(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_factory,
            needed_magic,
        ),
        *compute_duration(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_count,
            needed_magic,
        ),
    )


def total_magic(estimation_model: str, needed_magic: MagicCount) -> Tuple[Tuple[str, ...], str]:
    """Compute the number of magic states needed for the algorithm and their type."""
    total_t = needed_magic.n_t + 4 * needed_magic.n_ccz
    total_ccz = total_t / 4
    if estimation_model == _GIDNEY_FOLWER_MODEL:
        return ['Total Number of Toffoli gates'], f'{total_ccz:g}'
    else:
        return ['Total Number of T gates'], f'{total_t:g}'


def min_num_factories(
    physical_error_rate,
    error_budget: float,
    estimation_model: str,
    algorithm: AlgorithmSummary,
    rotation_model: rotation_cost_model.RotationCostModel,
    magic_factory: magic_state_factory.MagicStateFactory,
    needed_magic: MagicCount,
) -> Tuple[Dict[str, Any], int]:
    if estimation_model == _GIDNEY_FOLWER_MODEL:
        return {'display': 'none'}, 1
    c_min = minimum_time_steps(
        error_budget=error_budget, alg=algorithm, rotation_model=rotation_model
    )
    return {'display': 'block'}, int(
        np.ceil(magic_factory.n_cycles(needed_magic, physical_error_rate) / c_min)
    )


def compute_duration(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: AlgorithmSummary,
    rotation_model: rotation_cost_model.RotationCostModel,
    magic_count: int,
    needed_magic: MagicCount,
) -> Tuple[Dict[str, Any], str]:
    """Compute the duration of running the algorithm and whether to display the result or not.

    Currently displays the result only for GidneyFolwer (arxiv:1812.01238).
    """
    if estimation_model == _GIDNEY_FOLWER_MODEL:
        res, _, _ = get_ccz2t_costs_from_grid_search(
            n_magic=needed_magic,
            n_algo_qubits=algorithm.algorithm_qubits,
            phys_err=physical_error_rate,
            error_budget=error_budget,
            factory_iter=[MultiFactory(f, magic_count) for f in iter_ccz2t_factories()],
        )
        unit, duration = format_duration([res.duration_hr * 60 * 60 * 10**6])
        return {'display': 'block'}, f'{duration[0]:g} {unit}'
    else:
        return {'display': 'none'}, ''


if __name__ == '__main__':
    app.run_server(debug=True)
