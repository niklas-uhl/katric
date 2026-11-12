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
    # a kaval directory of the `phases` suite (experiments/local/phases.suite.yaml)
    repo = mo.notebook_dir().parent
    dirs = sorted((repo / "experiment_data").glob("phases_*"))[-1:]
    return (dirs,)


@app.cell
def _(dirs, katric_eval):
    r = katric_eval.load(*dirs)
    return (r,)


@app.cell
def _(katric_eval, pl, r):
    # configuration label from the options that differ between runs (as in scaling.py)
    def _label(o):
        if set(r.runs[o].unique()) <= {"true", "false"}:
            return pl.when(pl.col(o) == "true").then(pl.lit(o))
        return pl.col(o)

    runs = r.runs.with_columns(
        config=pl.concat_str([_label(o) for o in katric_eval.config_options(r.runs)], separator=" ", ignore_nulls=True)
    )
    return (runs,)


@app.cell
def _(pl, r, runs):
    # Where an iteration's time goes. Each region's exclusive time, averaged across ranks and over the iterations.
    # The phases end in barriers, so these add up to the iteration's run time; all `*/barrier` regions together are
    # the load imbalance, and the global phase's `termination` is its wait for the other ranks.
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
        .join(runs.select("run", "graph", "config", "cores"), on="run")
        .group_by("graph", "config", "cores", "part")
        .agg(time=pl.col("time").mean())
        .filter(pl.col("time") > 0)
    )
    return (breakdown,)


@app.cell
def _(alt, breakdown, mo):
    # click a legend entry to highlight that part / configuration (shift-click for more, double-click to reset);
    # fading instead of filtering, as filtered entries would drop out of the legend
    _part = alt.selection_point(fields=["part"], bind="legend")
    _config = alt.selection_point(fields=["config"], bind="legend")
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
            tooltip=["config", "cores", "part", alt.Tooltip("sum(time):Q", format=".4g")],
        )
        .properties(width=250, height=200)
        .facet(alt.Facet("graph:N", title=None), columns=3)
        .resolve_scale(y="independent")
    )
    return


@app.cell
def _(pl, r, runs):
    # Imbalance per phase: maximum over average of the ranks' exclusive work (1 = balanced)
    imbalance = (
        r.timers.filter(pl.col("iteration").is_not_null(), pl.col("region").str.count_matches("/") == 1,
                        pl.col("avg_exc") > 1e-4)
        .join(runs.select("run", "graph", "config", "cores"), on="run")
        .group_by("graph", "config", "cores", "region")
        .agg(max_over_avg=(pl.col("max_exc") / pl.col("avg_exc")).mean())
        .pivot(on="region", index=["graph", "config", "cores"], values="max_over_avg")
        .sort("graph", "config", "cores")
    )
    imbalance
    return


@app.cell
def _(pl, r, runs):
    # Input shape per rank (counters in `ingest`, the same for all configurations) and how the global phase spends its
    # time: building messages (send_ns) and handling received ones (handle_ns) against the phase's time, all summed
    # over the ranks. cut = cut edges / edges; *_share = share of the global phase's time summed over the ranks.
    # handle_ns includes the messages handled inside `termination`, so the shares can add up to more than 1.
    def _sum(region, counter):
        return (
            r.counters.filter(region=region, counter=counter)
            .group_by("run")
            .agg(pl.col("sum").mean().alias(counter))
        )

    _global = (
        r.timers.filter(region="total/global_phase")
        .group_by("run")
        .agg(global_s=pl.col("total").mean())
    )
    _termination = (
        r.timers.filter(region="total/global_phase/termination")
        .group_by("run")
        .agg(termination_s=pl.col("total").mean())
    )
    shape = (
        runs.select("run", "graph", "config", "cores")
        .join(_sum("ingest", "local_edges"), on="run")
        .join(_sum("ingest", "cut_edges"), on="run")
        .join(_sum("ingest", "max_degree"), on="run")
        .join(_sum("total/global_phase", "send_volume"), on="run", how="left")
        .join(_sum("total/global_phase", "send_ns"), on="run", how="left")
        .join(_sum("total/global_phase", "handle_ns"), on="run", how="left")
        .join(_global, on="run", how="left")
        .join(_termination, on="run", how="left")
        .group_by("graph", "config", "cores")
        .agg(pl.all().exclude("run").mean())
        .with_columns(
            cut=pl.col("cut_edges") / pl.col("local_edges"),
            volume_per_cut_edge=pl.col("send_volume") / pl.col("cut_edges"),
            send_share=pl.col("send_ns") * 1e-9 / pl.col("global_s"),
            handle_share=pl.col("handle_ns") * 1e-9 / pl.col("global_s"),
            termination_share=pl.col("termination_s") / pl.col("global_s"),
        )
        .select("graph", "config", "cores", "cut", "max_degree", "volume_per_cut_edge", "send_share", "handle_share",
                "termination_share")
        .sort("graph", "config", "cores")
    )
    shape
    return


if __name__ == "__main__":
    app.run()
