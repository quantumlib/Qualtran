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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import (
    AlgorithmSummary,
    beverland_et_al_model,
    CCZ2TFactory,
    FastDataBlock,
    FifteenToOne,
    gidney_fowler_model,
    LogicalErrorModel,
    MagicStateFactory,
    MultiFactory,
    PhysicalParameters,
    QECScheme,
    rotation_cost_model,
)


def get_objects(modules, object_type):
    """Get all objects of a given type from a list of modules."""
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj_name, obj


_QEC_SCHEMES = {
    'BeverlandEtAl': QECScheme.make_beverland_et_al(),
    'GidneyFowler': QECScheme.make_gidney_fowler(),
}
_MAGIC_FACTORIES = {
    'FifteenToOne733': FifteenToOne(7, 3, 3),
    'FifteenToOne933': FifteenToOne(9, 3, 3),
    'GidneyFowlerCCZ': CCZ2TFactory(),
}
_ROTATION_MODELS = dict(get_objects([rotation_cost_model], rotation_cost_model.RotationCostModel))

_GIDNEY_FOWLER_MODEL = 'GidneyFowler (arxiv:1812.01238)'
_BEVERLAND_MODEL = 'Beverland et al (arxiv:2211.07629)'
_SUPPORTED_ESTIMATION_MODELS = [_GIDNEY_FOWLER_MODEL, _BEVERLAND_MODEL]

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
            id='estimation_model', options=_SUPPORTED_ESTIMATION_MODELS, value=_GIDNEY_FOWLER_MODEL
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
app.title = 'Qualtran Resource Estimation'
app.layout = html.Div(
    [
        html.H4('Interactive QEC overhead estimation'),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td(input_components(), style={'width': '30vw'}),
                        html.Td(create_ouputs(), style={'width': '20vw'}),
                        html.Td(
                            dcc.Loading(
                                dcc.Graph(id="qubit-pie-chart"),
                                type="cube",
                                style={'max-width': '80vw', 'float': 'right'},
                            )
                        ),
                    ]
                )
            ],
            style={'width': '100vw'},
        ),
        html.Div(
            id='runtime-container',
            children=[
                dcc.Loading(dcc.Graph(id="runtime-plot"), type="cube", style={'max-width': '80vw'})
            ],
        ),
    ]
)


def create_qubit_pie_chart(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: 'AlgorithmSummary',
    magic_factory: MagicStateFactory,
    magic_count: int,
    n_logical_gates: 'GateCounts',
) -> go.Figure:
    """Create a pie chart of the physical qubit utilization."""
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, factory, _ = gidney_fowler_model.get_ccz2t_costs_from_grid_search(
            n_logical_gates=n_logical_gates,
            n_algo_qubits=int(algorithm.n_algo_qubits),
            phys_err=physical_error_rate,
            error_budget=error_budget,
            factory_iter=[
                MultiFactory(f, int(magic_count))
                for f in gidney_fowler_model.iter_ccz2t_factories()
            ],
        )
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [
            res.footprint - factory.n_physical_qubits(),
            factory.n_physical_qubits(),
        ]
        fig = px.pie(
            memory_footprint, values='qubits', names='source', title='Physical Qubit Utilization'
        )
        fig.update_traces(textinfo='value')
        return fig
    else:
        multi_factory = MultiFactory(magic_factory, int(magic_count))
        memory_footprint = pd.DataFrame(columns=['source', 'qubits'])
        memory_footprint['source'] = [
            'logical qubits + routing overhead',
            'Magic State Distillation',
        ]
        memory_footprint['qubits'] = [
            FastDataBlock.get_n_tiles(int(algorithm.n_algo_qubits)),
            multi_factory.n_physical_qubits(),
        ]
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
    if duration[0] > 86400_000_000:
        duration = [d / 86400_000_000 for d in duration]
        unit = 'days'
    elif duration[0] > 3600_000_000:
        duration = [d / 3600_000_000 for d in duration]
        unit = 'hours'
    elif duration[0] > 60_000_000:
        duration = [d / 60_000_000 for d in duration]
        unit = 'min'
    elif duration[0] > 1000_000:
        duration = [d / 1000_000 for d in duration]
        unit = 's'
    elif duration[0] > 1000:
        duration = [d / 1000 for d in duration]
        unit = 'ms'
    return unit, duration


def create_runtime_plot(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: 'AlgorithmSummary',
    qec: QECScheme,
    magic_factory: MagicStateFactory,
    magic_count: int,
    rotation_model: rotation_cost_model.RotationCostModel,
    n_logical_gates: 'GateCounts',
) -> Tuple[Dict[str, Any], go.Figure]:
    """Creates the runtime figure and decides whether to display it or not.

    Currently displays the runtime plot for the Beverland model only.
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {'display': 'none'}, go.Figure()
    factory = MultiFactory(magic_factory, int(magic_count))
    c_min = beverland_et_al_model.minimum_time_steps(
        error_budget=error_budget, alg=algorithm, rotation_model=rotation_model
    )
    err_model = LogicalErrorModel(qec_scheme=qec, physical_error=physical_error_rate)
    factory_cycles = factory.n_cycles(
        n_logical_gates=n_logical_gates, logical_error_model=err_model
    )
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
        beverland_et_al_model.code_distance(
            error_budget=error_budget,
            time_steps=t,
            alg=algorithm,
            qec_scheme=qec,
            physical_error=physical_error_rate,
        )
        for t in time_steps
    ]
    # TODO: plumb through PhysicalParameters
    cycle_time_us = PhysicalParameters.make_gidney_fowler().cycle_time_us
    duration = [cycle_time_us * t for t, d in zip(time_steps, cds)]
    unit, duration = format_duration(duration)
    duration_name = f'Duration ({unit})'
    num_qubits = (
        FastDataBlock.get_n_tiles(int(algorithm.n_algo_qubits))
        + factory.n_physical_qubits() * magic_counts
    )
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
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm_data: Sequence[Any],
    qec_name: str,
    magic_name: str,
    magic_count: int,
    rotaion_model_name: str,
):
    """Updates the visualization."""
    if any(x is None for x in [physical_error_rate, error_budget, *algorithm_data, magic_count]):
        raise PreventUpdate

    # TODO: We implicitly rely on the order of the input components
    qubits, measurements, ts, toffolis, rotations, n_rotation_layers = algorithm_data
    algorithm = AlgorithmSummary(
        n_algo_qubits=qubits,
        n_logical_gates=GateCounts(
            measurement=measurements, t=ts, toffoli=toffolis, rotation=rotations
        ),
        n_rotation_layers=n_rotation_layers,
    )
    qec = _QEC_SCHEMES[qec_name]
    magic_factory = _MAGIC_FACTORIES[magic_name]
    rotation_model = _ROTATION_MODELS[rotaion_model_name]
    n_logical_gates = beverland_et_al_model.n_discrete_logical_gates(
        eps_syn=error_budget / 3, alg=algorithm, rotation_model=rotation_model
    )
    # n_logical_gates = GateCounts(t=int(n_logical_gates.t), toffoli=int(n_logical_gates.toffoli))
    magic_count = int(magic_count)
    logical_err_model = LogicalErrorModel(qec_scheme=qec, physical_error=physical_error_rate)
    return (
        create_qubit_pie_chart(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            magic_factory,
            magic_count,
            n_logical_gates,
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
            n_logical_gates,
        ),
        *total_magic(estimation_model, n_logical_gates),
        *min_num_factories(
            logical_err_model,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_factory,
            n_logical_gates,
        ),
        *compute_duration(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_count,
            n_logical_gates,
        ),
    )


def total_magic(estimation_model: str, n_logical_gates: 'GateCounts') -> Tuple[List[str], str]:
    """Compute the number of magic states needed for the algorithm and their type."""
    total_t = n_logical_gates.total_t_count()
    total_ccz = total_t / 4
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return ['Total Number of Toffoli gates'], f'{total_ccz:g}'
    else:
        return ['Total Number of T gates'], f'{total_t:g}'


def min_num_factories(
    logical_error_model: 'LogicalErrorModel',
    error_budget: float,
    estimation_model: str,
    algorithm: 'AlgorithmSummary',
    rotation_model: rotation_cost_model.RotationCostModel,
    magic_factory: MagicStateFactory,
    n_logical_gates: 'GateCounts',
) -> Tuple[Dict[str, Any], int]:
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {'display': 'none'}, 1
    c_min = beverland_et_al_model.minimum_time_steps(
        error_budget=error_budget, alg=algorithm, rotation_model=rotation_model
    )
    return {'display': 'block'}, int(
        np.ceil(
            magic_factory.n_cycles(
                n_logical_gates=n_logical_gates, logical_error_model=logical_error_model
            )
            / c_min
        )
    )


def compute_duration(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: 'AlgorithmSummary',
    rotation_model: rotation_cost_model.RotationCostModel,
    magic_count: int,
    n_logical_gates: 'GateCounts',
) -> Tuple[Dict[str, Any], str]:
    """Compute the duration of running the algorithm and whether to display the result or not.

    Currently displays the result only for GidneyFowler (arxiv:1812.01238).
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, _, _ = gidney_fowler_model.get_ccz2t_costs_from_grid_search(
            n_logical_gates=n_logical_gates,
            n_algo_qubits=int(algorithm.n_algo_qubits),
            phys_err=physical_error_rate,
            error_budget=error_budget,
            factory_iter=[
                MultiFactory(f, magic_count) for f in gidney_fowler_model.iter_ccz2t_factories()
            ],
        )
        unit, duration = format_duration([res.duration_hr * 60 * 60 * 10**6])
        return {'display': 'block'}, f'{duration[0]:g} {unit}'
    else:
        return {'display': 'none'}, ''


if __name__ == '__main__':
    app.run_server(debug=True)
