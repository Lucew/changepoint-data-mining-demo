window.mouseState = { isDown: false };

document.addEventListener("mousedown", () => {
    window.mouseState.isDown = true;
});
document.addEventListener("mouseup", () => {
    window.mouseState.isDown = false;
});
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    move_shapes_namespace: {
        move_shapes:
            function emit_event_move_shapes(relayout_data, plot_div_id) {

                // do nothing if the user has pressed down the mouse
                if (window.mouseState.isDown) {
                    return window.dash_clientside.no_update;
                }

                function allKeysStartWithShape(obj) {
                    if (obj === null || typeof obj !== "object" || Array.isArray(obj)) {
                        return false;
                    }

                    const symbolKeys = Object.getOwnPropertySymbols(obj);
                    if (symbolKeys.length > 0) {
                        return false;
                    }

                    return Object.keys(obj).every(key => key.toString().startsWith("shapes["));
                }

                // check whether the relayout data has the form that we expect for a shape move
                const is_valid_move = allKeysStartWithShape(relayout_data)
                if (!is_valid_move){
                    return window.dash_clientside.no_update;
                }

                // get the shapes from the plotly plot contained in the div
                // get the plotly plot
                const target = document.getElementById(plot_div_id);
                const gd = target.querySelector('.js-plotly-plot');
                const plot_shapes = (gd.layout && gd.layout.shapes) || [];

                const shape_move = new CustomEvent('shapeMove', {
                    bubbles: true,
                    detail: {
                        children: plot_shapes,
                        relayout_data: relayout_data,
                    }
                });

                console.log(shape_move)
                target.dispatchEvent(shape_move)
                return window.dash_clientside.no_update
            }
    }
});