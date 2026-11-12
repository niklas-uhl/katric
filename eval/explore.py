import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import katric_eval

    return alt, katric_eval, mo, pl


@app.cell(hide_code=True)
def _(mo):
    # kaval experiment directories (<suite>_<date>), or a directory holding several (e.g. eval/data/supermuc);
    # `just fetch-from-*` copies cluster runs into eval/data/<cluster>
    repo = mo.notebook_dir().parent
    browser = mo.ui.file_browser(
        initial_path=repo / "eval" / "data",
        selection_mode="directory",
        multiple=True,
        restrict_navigation=False,
        label="Experiment directories (select several to compare machines or suites)",
    )
    browser
    return (browser,)


@app.cell(hide_code=True)
def _(mo, r):
    regions = sorted(r.timers.filter(iteration=0)["region"].unique())
    region = mo.ui.dropdown(regions, value="total", label="region for the scaling plot")
    merge = mo.ui.checkbox(
        value=True,
        label="merge configurations across directories (ignore options that only differ between them, e.g. iterations)",
    )
    mo.hstack([region, merge], justify="start", gap=2)
    return merge, region


@app.cell(hide_code=True)
def _(browser, katric_eval, mo):
    # a selected directory without profiles of its own stands for the experiment directories below it
    def _expand(d):
        if (d / "output").is_dir() or any(d.glob("*.cali")):
            return [d]
        return sorted(c for c in d.iterdir() if c.is_dir() and ((c / "output").is_dir() or any(c.glob("*.cali"))))

    dirs = [e for f in browser.value for e in _expand(f.path)]
    mo.stop(not dirs, mo.md("Select one or more experiment directories above."))
    r = katric_eval.load(*dirs)
    return (r,)


@app.cell(hide_code=True)
def _(katric_eval, merge, pl, r):
    # configuration label from the options that differ between runs (with `merge`: within a directory): the value, or
    # the option name for a set flag; `series` adds the machine when runs of several clusters are loaded
    def _label(o):
        if set(r.runs[o].unique()) <= {"true", "false"}:
            return pl.when(pl.col(o) == "true").then(pl.lit(o))
        return pl.col(o)

    _machines = {"hkn": "HoreKa", "ircs": "SuperMUC-NG"}
    runs = r.runs.with_columns(
        config=pl.concat_str([_label(o) for o in katric_eval.config_options(r.runs, within_dirs=merge.value)],
                             separator=" ", ignore_nulls=True),
        machine=pl.col("cluster").replace(_machines),
    )
    runs = runs.with_columns(
        series=pl.format("{} @ {}", "config", "machine") if runs["machine"].n_unique() > 1 else pl.col("config")
    )
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Scaling

    Run time of an iteration: maximum across ranks of the selected region, mean over the iterations. Hover for
    details, click points (shift-click for more) to list their runs under *Details*, click a legend entry to
    highlight a configuration.
    """)
    return


@app.cell(hide_code=True)
def _(pl, r, region, runs):
    times = (
        r.timers.filter(region=region.value)
        .join(runs, on="run")
        .group_by("graph", "series", "machine", "config", "cores", "threads")
        .agg(time=pl.col("max").mean())
        .with_columns(line=pl.format("{} {}", "series", "threads"))  # one line per series and threads/rank
        .sort("graph", "series", "cores")
    )
    return (times,)


@app.cell(hide_code=True)
def _(alt, mo, region, times):
    _select = alt.selection_point(fields=["graph", "series", "cores", "threads"])
    _legend = alt.selection_point(fields=["series"], bind="legend")
    chart = mo.ui.altair_chart(
        alt.Chart(times)
        .mark_line(point=True)
        .add_params(_select, _legend)
        .encode(
            x=alt.X("cores:Q", scale=alt.Scale(type="log", base=2),
                    axis=alt.Axis(values=sorted(times["cores"].unique()))),  # ticks at the measured core counts
            y=alt.Y("time:Q", scale=alt.Scale(type="log"), title=f"time of '{region.value}' [s]"),
            color=alt.Color("series:N", legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=0)),
            shape=alt.Shape("threads:N", title="threads/rank"),
            detail="line:N",
            opacity=alt.condition(_select & _legend, alt.value(1.0), alt.value(0.15)),
            tooltip=["series", "threads", "cores", alt.Tooltip("time:Q", format=".4g")],
        )
        .properties(width=250, height=200)
        .facet(alt.Facet("graph:N", title=None), columns=3)
        .resolve_scale(y="independent")
    )
    chart
    return (chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Where the time goes

    Exclusive time per region, averaged across ranks and over the iterations. The phases end in barriers, so the
    parts add up to the iteration's run time; all `*/barrier` regions together are the load imbalance, and the global
    phase's `termination` is its wait for the other ranks. Click legend entries to highlight parts or configurations.
    """)
    return


@app.cell(hide_code=True)
def _(pl, r, runs):
    def _part(region):
        leaf = region.rsplit("/", 1)[-1]
        return "barrier (imbalance)" if leaf == "barrier" else region.removeprefix("total/")

    breakdown = (
        r.timers.filter(pl.col("iteration").is_not_null(), pl.col("region").str.starts_with("total/"))
        .with_columns(part=pl.col("region").map_elements(_part, return_dtype=pl.String))
        .group_by("run", "iteration", "part")
        .agg(time=pl.col("avg_exc").sum())
        .group_by("run", "part")
        .agg(time=pl.col("time").mean())
        .join(runs.select("run", "graph", "machine", "config", "cores"), on="run")
        .group_by("machine", "graph", "config", "cores", "part")
        .agg(time=pl.col("time").mean())
        .filter(pl.col("time") > 0)
    )
    return (breakdown,)


@app.cell(hide_code=True)
def _(alt, breakdown, mo):
    # fading instead of filtering on legend clicks, as filtered entries would drop out of the legend
    _part = alt.selection_point(fields=["part"], bind="legend")
    _config = alt.selection_point(fields=["config"], bind="legend")
    _facet = {"column": alt.Column("graph:N", title=None)}
    if breakdown["machine"].n_unique() > 1:
        _facet["row"] = alt.Row("machine:N", title=None)
    mo.ui.altair_chart(
        alt.Chart(breakdown)
        .mark_bar()
        .add_params(_part, _config)
        .encode(
            x=alt.X("cores:O"),
            xOffset="config:N",
            y=alt.Y("sum(time):Q", title="time per iteration [s]"),
            color=alt.Color("part:N", legend=alt.Legend(orient="bottom", columns=4, labelLimit=0)),
            # shape only to get a legend for the configurations (bars have no shape)
            shape=alt.Shape("config:N", legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=0)),
            opacity=alt.condition(_part & _config, alt.value(1.0), alt.value(0.15)),
            tooltip=["machine", "config", "cores", "part", alt.Tooltip("sum(time):Q", format=".4g")],
        )
        .properties(width=220, height=200)
        .facet(**_facet)
        .resolve_scale(y="independent")
    )
    return


@app.cell(hide_code=True)
def _(chart, pl, r, runs):
    # the selected points' runs with the time of every region (maximum across ranks, mean over the iterations)
    _selected = runs.join(chart.value.select("graph", "series", "cores", "threads"),
                          on=["graph", "series", "cores", "threads"])
    selected_regions = (
        r.timers.filter(pl.col("iteration").is_not_null())
        .join(_selected.select("run", "graph", "series", "cores", "threads"), on="run")
        .group_by("run", "graph", "series", "cores", "threads", "region")
        .agg(time=pl.col("max").mean())
        .sort("run", "region")
    )
    return (selected_regions,)


@app.cell(hide_code=True)
def _(pl, r, runs):
    # maximum over average of the ranks' exclusive work per phase (1 = balanced)
    imbalance = (
        r.timers.filter(pl.col("iteration").is_not_null(), pl.col("region").str.count_matches("/") == 1,
                        pl.col("avg_exc") > 1e-4)
        .join(runs.select("run", "machine", "graph", "config", "cores"), on="run")
        .group_by("machine", "graph", "config", "cores", "region")
        .agg(max_over_avg=(pl.col("max_exc") / pl.col("avg_exc")).mean())
        .pivot(on="region", index=["machine", "graph", "config", "cores"], values="max_over_avg")
        .sort("machine", "graph", "config", "cores")
    )
    return (imbalance,)


@app.cell(hide_code=True)
def _(pl, r, runs):
    # Input shape (counters in `ingest`) and how the global phase spends its time: building messages (send_ns) and
    # handling received ones (handle_ns, also inside `termination`, so the shares can add up to more than 1), summed
    # over the ranks. cut = cut edges / edges; *_share = share of the global phase's time summed over the ranks.
    def _sum(region, counter):
        return r.counters.filter(region=region, counter=counter).group_by("run").agg(pl.col("sum").mean().alias(counter))

    def _time(region, name):
        return r.timers.filter(region=region).group_by("run").agg(pl.col("total").mean().alias(name))

    shape = (
        runs.select("run", "machine", "graph", "config", "cores")
        .join(_sum("ingest", "local_edges"), on="run")
        .join(_sum("ingest", "cut_edges"), on="run")
        .join(_sum("ingest", "max_degree"), on="run")
        .join(_sum("total/global_phase", "send_volume"), on="run", how="left")
        .join(_sum("total/global_phase", "send_ns"), on="run", how="left")
        .join(_sum("total/global_phase", "handle_ns"), on="run", how="left")
        .join(_time("total/global_phase", "global_s"), on="run", how="left")
        .join(_time("total/global_phase/termination", "termination_s"), on="run", how="left")
        .group_by("machine", "graph", "config", "cores")
        .agg(pl.all().exclude("run").mean())
        .with_columns(
            cut=pl.col("cut_edges") / pl.col("local_edges"),
            volume_per_cut_edge=pl.col("send_volume") / pl.col("cut_edges"),
            send_share=pl.col("send_ns") * 1e-9 / pl.col("global_s"),
            handle_share=pl.col("handle_ns") * 1e-9 / pl.col("global_s"),
            termination_share=pl.col("termination_s") / pl.col("global_s"),
        )
        .select("machine", "graph", "config", "cores", "cut", "max_degree", "volume_per_cut_edge", "send_share",
                "handle_share", "termination_share")
        .sort("machine", "graph", "config", "cores")
    )
    return (shape,)


@app.cell(hide_code=True)
def _(imbalance, mo, selected_regions, shape):
    mo.accordion(
        {
            "Details: selected runs (click points in the scaling plot)": (
                selected_regions if len(selected_regions) else mo.md("No points selected.")
            ),
            "Details: imbalance per phase (max/avg of the ranks' exclusive time)": imbalance,
            "Details: input shape and the global phase's split": shape,
        }
    )
    return


if __name__ == "__main__":
    app.run()
