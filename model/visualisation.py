from mesa.experimental.jupyter_viz import *
from gym_model import Gym, GymLayout
from gym_agent import Muscle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_layout(layout: GymLayout, ax: plt.Axes, title="Gym layout", letters=True, cmap_name="inferno") -> None:
    """Draw a gym layout, where each machine is a colored square.
    Muscle enum is roughly ordered by body part, so prefer sequential colormap eg. "viridis", "plasma", "inferno", "magma", "cividis"
    (see https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential)."""

    cmap = plt.get_cmap(cmap_name)
    image = np.zeros(layout.shape + (4,))

    for i, j in np.ndindex(layout.shape):
        if (machine := layout[i, j]) is not None:
            image[i, j] = cmap(bg_spectrum := machine.muscle.value / len(Muscle))
            if letters:
                words = machine.name.split("_")
                lab = words[0].lower()[:3] if len(words) == 1 else "".join([w[0] for w in words])
                ax.text(i, j, lab, c=cmap((bg_spectrum + .5) % 1), ha='center', va='center')


    ax.imshow(np.transpose(image, (1, 0, 2)), origin='lower')
    ax.set_axis_off()
    ax.set_title(title)
    # TODO: optionally add annotated legend (the whole spectrum)
    


# Avoid interactive backend
plt.switch_backend("agg") # TODO?: delete this


@solara.component
def JupyterVizGym( # NOTE: copy-pasted mesa/experimental/jupyter_viz.py (one line edited)
    model_class,
    model_params,
    measures=None,
    name="Mesa Model",
    agent_portrayal=None,
    space_drawer="default",
    play_interval=150,
):
    """Initialize a component to visualize a model.
    Args:
        model_class: class of the model to instantiate
        model_params: parameters for initializing the model
        measures: list of callables or data attributes to plot
        name: name for display
        agent_portrayal: options for rendering agents (dictionary)
        space_drawer: method to render the agent space for
            the model; default implementation is the `SpaceMatplotlib` component;
            simulations with no space to visualize should
            specify `space_drawer=False`
        play_interval: play interval (default: 150)
    """
    current_step = solara.use_reactive(0)

    # 1. Set up model parameters
    user_params, fixed_params = split_model_params(model_params)
    model_parameters, set_model_parameters = solara.use_state(
        {**fixed_params, **{k: v["value"] for k, v in user_params.items()}}
    )

    # 2. Set up Model
    def make_model():
        model = model_class(**model_parameters)
        current_step.value = 0
        return model

    reset_counter = solara.use_reactive(0)
    model = solara.use_memo(
        make_model, dependencies=[*list(model_parameters.values()), reset_counter.value]
    )

    def handle_change_model_params(name: str, value: any):
        set_model_parameters({**model_parameters, name: value})

    def ColorCard(color, layout_type):
        # TODO: turn this into a Solara component, but must pass in current
        # step as a dependency for the plots, so that there is no flickering
        # due to rerender.
        with rv.Card(
            style_=f"background-color: {color}; width: 100%; height: 100%"
        ) as main:
            if "Space" in layout_type:
                rv.CardTitle(children=["Space"])
                if space_drawer == "default":
                    # draw with the default implementation
                    SpaceMatplotlib(
                        model, agent_portrayal, dependencies=[current_step.value]
                    )
                elif space_drawer:
                    # if specified, draw agent space with an alternate renderer
                    space_drawer(model, agent_portrayal)
            elif "Measure" in layout_type:
                rv.CardTitle(children=["Measure"])
                measure = measures[layout_type["Measure"]]
                if callable(measure):
                    # Is a custom object
                    measure(model)
                else:
                    make_plot(model, measure)
        return main

    # 3. Set up UI

    with solara.AppBar():
        solara.AppBarTitle(name)

    # render layout and plot

    # jupyter
    def render_in_jupyter():
        with solara.GridFixed(columns=2):
            UserInputs(user_params, on_change=handle_change_model_params)
            ModelController(model, play_interval, current_step, reset_counter)
            solara.Markdown(md_text=f"###Step - {current_step}")

        with solara.GridFixed(columns=2):
            # 4. Space
            if space_drawer == "default":
                # draw with the default implementation
                SpaceMatplotlib(
                    model, agent_portrayal, dependencies=[current_step.value]
                )
            elif space_drawer:
                # if specified, draw agent space with an alternate renderer
                space_drawer(model, agent_portrayal, dependencies=[current_step.value]) # NOTE: this line edited
            # otherwise, do nothing (do not draw space)

            # 5. Plots

            for measure in measures:
                if callable(measure):
                    # Is a custom object
                    measure(model)
                else:
                    make_plot(model, measure)

    def render_in_browser():
        if measures:
            layout_types = [{"Space": "default"}] + [
                {"Measure": elem} for elem in range(len(measures))
            ]
        else:
            layout_types = [{"Space": "default"}]
        grid_layout_initial = get_initial_grid_layout(layout_types=layout_types)
        grid_layout, set_grid_layout = solara.use_state(grid_layout_initial)

        with solara.Sidebar():
            with solara.Card("Controls", margin=1, elevation=2):
                UserInputs(user_params, on_change=handle_change_model_params)
                ModelController(model, play_interval, current_step, reset_counter)
            with solara.Card("Progress", margin=1, elevation=2):
                solara.Markdown(md_text=f"####Step - {current_step}")

        items = [
            ColorCard(color="white", layout_type=layout_types[i])
            for i in range(len(layout_types))
        ]
        solara.GridDraggable(
            items=items,
            grid_layout=grid_layout,
            resizable=True,
            draggable=True,
            on_grid_layout=set_grid_layout,
        )

    if ("ipykernel" in sys.argv[0]) or ("colab_kernel_launcher.py" in sys.argv[0]):
        # When in Jupyter or Google Colab
        render_in_jupyter()
    else:
        render_in_browser()


# modified functions from mesa/experimental/jupyter_viz.py

@solara.component
def GymDrawer(model: Gym, agent_portrayal, dependencies: Optional[List[any]] = None):
    """adapated function SpaceMatplotlib"""
    space_fig = Figure()
    space_ax = space_fig.subplots()
    space = getattr(model, "grid", None)
    if space is None:
        # Sometimes the space is defined as model.space instead of model.grid
        space = model.space

    assert not isinstance(space, (mesa.space.NetworkGrid, mesa.space.ContinuousSpace))
    draw_gym(space, space_ax, agent_portrayal)
    space_ax.set_axis_off()
    solara.FigureMatplotlib(space_fig, format="png", dependencies=dependencies) 


def draw_gym(space: mesa.space._Grid, space_ax, agent_portrayal):
    attrs = ["x", "y", "s", "color", "marker", "alpha", "linewidths", "edgecolors"]

    def portray(g):
        out = {attr: [] for attr in attrs}

        for i in range(g.width):
            for j in range(g.height):
                content = g._grid[i][j]
                if not content:
                    continue
                if not hasattr(content, "__iter__"):
                    # Is a single grid
                    content = [content]
                for agent in content:
                    data = agent_portrayal(agent)
                    out["x"].append(i)
                    out["y"].append(j)
                    
                    for attr in attrs[2:]:
                        if attr in data:
                            out[attr].append(data[attr])

                space_ax.scatter(**{attr: out[attr][-1] for attr in out if out[attr]})
        
        return {k: v for k, v in out.items() if len(v) > 0}

    portray(space)
    # space_ax.scatter(**portray(space))
