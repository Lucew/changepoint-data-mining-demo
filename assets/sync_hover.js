window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside3: {
        sync_raw_signal_hover: function(clickDataList, ids, relayoutDataList, figures, selectedX) {
        const Patch = window.dash_clientside.Patch;
        const no_update = window.dash_clientside.no_update;
        const MARKER_NAME = "__linked_cursor__";

        function toComparable(v) {
            if (v === null || v === undefined) return null;
            if (typeof v === "number") return v;

            const t = Date.parse(v);
            if (!Number.isNaN(t)) return t;

            return String(v);
        }

        function baseRangeFromFigure(fig) {
            const layout = (fig && fig.layout) || {};
            const xaxis = layout.xaxis || {};

            if (Array.isArray(xaxis.range) && xaxis.range.length === 2) {
                return xaxis.range;
            }

            let xmin = null;
            let xmax = null;

            for (const trace of (fig && fig.data) || []) {
                const xs = trace && trace.x;
                if (!Array.isArray(xs)) continue;

                for (const x of xs) {
                    if (x === null || x === undefined) continue;
                    const cx = toComparable(x);
                    if (cx === null) continue;

                    if (xmin === null || cx < xmin) xmin = cx;
                    if (xmax === null || cx > xmax) xmax = cx;
                }
            }

            return xmin === null ? null : [xmin, xmax];
        }

        function currentVisibleRange(relayoutData, fig) {
            if (
                relayoutData &&
                relayoutData["xaxis.range[0]"] !== undefined &&
                relayoutData["xaxis.range[1]"] !== undefined
            ) {
                return [
                    relayoutData["xaxis.range[0]"],
                    relayoutData["xaxis.range[1]"]
                ];
            }

            if (
                relayoutData &&
                relayoutData.xaxis &&
                Array.isArray(relayoutData.xaxis.range) &&
                relayoutData.xaxis.range.length === 2
            ) {
                return relayoutData.xaxis.range;
            }

            if (relayoutData && relayoutData["xaxis.autorange"]) {
                return baseRangeFromFigure(fig);
            }

            return baseRangeFromFigure(fig);
        }

        function sameX(a, b, clickedRange) {
            const aa = toComparable(a);
            const bb = toComparable(b);

            if (aa === null || bb === null) return false;

            // numeric/date axis -> tolerance relative to clicked plot range
            if (typeof aa === "number" && typeof bb === "number") {
                if (!clickedRange) {
                    return aa === bb;
                }

                const r0 = toComparable(clickedRange[0]);
                const r1 = toComparable(clickedRange[1]);
                if (typeof r0 !== "number" || typeof r1 !== "number") {
                    return aa === bb;
                }

                const span = Math.abs(r1 - r0);

                // choose one:
                // 0.5% of visible range
                const tol = span * 0.005;

                // optional minimum tolerance so very deep zoom still feels clickable
                const minTol = 0;
                const effectiveTol = Math.max(tol, minTol);

                return Math.abs(aa - bb) <= effectiveTol;
            }

            // categorical/string axis
            return aa === bb;
        }

        function findMarkerIndex(fig) {
            const shapes = (((fig || {}).layout || {}).shapes) || [];
            return shapes.findIndex(
                s => s && s.type === "line" && s.name === MARKER_NAME
            );
        }

        function makeVerticalLine(x) {
            return {
                type: "line",
                name: MARKER_NAME,
                xref: "x",
                yref: "paper",
                x0: x,
                x1: x,
                y0: 0,
                y1: 1,
                line: {
                    color: "black",
                    width: 1,
                    dash: "dot"
                }
            };
        }

        const triggeredId = dash_clientside.callback_context.triggered_id;
        if (!triggeredId) {
            return [Array(ids.length).fill(no_update), no_update];
        }

        const triggerIndex = ids.findIndex(
            id => id && id.type === triggeredId.type && id.index === triggeredId.index
        );
        if (triggerIndex === -1) {
            return [Array(ids.length).fill(no_update), no_update];
        }

        const sourceClick = clickDataList[triggerIndex];
        const point = sourceClick && sourceClick.points && sourceClick.points[0];

        if (!point) {
            return [
                figures.map((fig) => {
                    const patch = new Patch();
                    const markerIndex = findMarkerIndex(fig);

                    if (markerIndex !== -1) {
                            patch.delete(["layout", "shapes", markerIndex]);
                    }
                    return patch.build();
                }),
                null
            ];
        }

        const clickX = point.x;
        const clickedRange = currentVisibleRange(
            relayoutDataList[triggerIndex],
            figures[triggerIndex]
        );

        // toggle off if same x within clicked plot's visible range tolerance
        const newSelectedX = sameX(clickX, selectedX, clickedRange) ? null : clickX;

        const patchedFigures = figures.map((fig, i) => {
            const patch = new Patch();
            const markerIndex = findMarkerIndex(fig);

            if (markerIndex !== -1) {
                    patch.delete(["layout", "shapes", markerIndex]);
            }

            if (newSelectedX === null ) {
                return patch.build();
            }

            const range = currentVisibleRange(relayoutDataList[i], fig);
            if (!range) {
                return patch.build();
            }

            const xCmp = toComparable(newSelectedX);
            const r0 = toComparable(range[0]);
            const r1 = toComparable(range[1]);
            const minR = r0 <= r1 ? r0 : r1;
            const maxR = r0 <= r1 ? r1 : r0;

            if (xCmp >= minR && xCmp <= maxR) {
                patch.append(["layout", "shapes"], makeVerticalLine(newSelectedX));
            }

            return patch.build();
        });

        return [patchedFigures, newSelectedX];
    }
    }
});