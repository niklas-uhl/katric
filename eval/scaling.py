import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import katric_eval

    return alt, katric_eval, mo, pl


@app.cell
def _(mo):
    # kaval experiment directories (<suite>_<date>); `just fetch-from-*` copies cluster runs into eval/data/<cluster>
    repo = mo.notebook_dir().parent
    dirs = [repo / "experiment_data" / "smoke_26_11_11"]
    return (dirs,)


@app.cell
def _(dirs, katric_eval):
    # parses every profile with Thicket; runs again only when `dirs` changes
    r = katric_eval.load(*dirs)
    return (r,)


@app.cell
def _(katric_eval, pl, r):
    # configuration label from the options that differ between runs: the value, or the option name for a set flag
    def _label(o):
        if set(r.runs[o].unique()) <= {"true", "false"}:
            return pl.when(pl.col(o) == "true").then(pl.lit(o))
        return pl.col(o)

    runs = r.runs.with_columns(
        config=pl.concat_str([_label(o) for o in katric_eval.config_options(r.runs)], separator=" ", ignore_nulls=True)
    )
    return (runs,)


@app.cell
def _(mo, r):
    regions = sorted(r.timers.filter(iteration=0)["region"].unique())
    region = mo.ui.dropdown(regions, value="total", label="region")
    region
    return (region,)


@app.cell
def _(pl, r, region, runs):
    # one row per graph, configuration and core count: the run time of an iteration is the maximum across ranks,
    # averaged over the iterations
    times = (
        r.timers.filter(region=region.value)
        .join(runs, on="run")
        .group_by("graph", "config", "cores", "threads")
        .agg(time=pl.col("max").mean())
        .with_columns(line=pl.format("{} {}", "config", "threads"))  # one line per configuration and threads/rank
        .sort("graph", "config", "cores")
    )
    return (times,)


@app.cell
def _(alt, mo, region, times):
    # hover for details; click points (shift-click for more) to list their runs below
    _select = alt.selection_point(fields=["graph", "config", "cores", "threads"])
    _legend = alt.selection_point(fields=["config"], bind="legend")  # click a legend entry to highlight it
    chart = mo.ui.altair_chart(
        alt.Chart(times)
        .mark_line(point=True)
        .add_params(_select, _legend)
        .encode(
            x=alt.X("cores:Q", scale=alt.Scale(type="log", base=2),
                    axis=alt.Axis(values=sorted(times["cores"].unique()))),  # ticks at the measured core counts
            y=alt.Y("time:Q", scale=alt.Scale(type="log"), title=f"time of '{region.value}' [s]"),
            color=alt.Color("config:N", legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=0)),
            shape=alt.Shape("threads:N", title="threads/rank"),
            detail="line:N",
            opacity=alt.condition(_select & _legend, alt.value(1.0), alt.value(0.15)),
            tooltip=["config", "threads", "cores", alt.Tooltip("time:Q", format=".4g")],
        )
        .properties(width=250, height=200)
        .facet(alt.Facet("graph:N", title=None), columns=3)
        .resolve_scale(y="independent")
    )
    chart
    return (chart,)


@app.cell
def _(chart, mo, pl, r, region, runs):
    # the selected points' runs with the time of every region (maximum across ranks, mean over the iterations)
    _selected = runs.join(chart.value.select("graph", "config", "cores", "threads"),
                          on=["graph", "config", "cores", "threads"])
    _regions = (
        r.timers.filter(pl.col("iteration").is_not_null())
        .join(_selected.select("run", "graph", "config", "cores", "threads"), on="run")
        .group_by("run", "graph", "config", "cores", "threads", "region")
        .agg(time=pl.col("max").mean())
        .sort("run", "region")
    )
    _regions if len(_selected) else mo.md(f"Select points in the plot to see their runs (region '{region.value}').")
    return


if __name__ == "__main__":
    app.run()
