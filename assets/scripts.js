if (!window.dash_clientside) {
    window.dash_clientside = {};
}

function getPlotDiv(graphId) {
    const graphRoot = document.getElementById(graphId);
    if (!graphRoot) {
        return null;
    }

    if (graphRoot.classList.contains("js-plotly-plot")) {
        return graphRoot;
    }

    return graphRoot.querySelector(".js-plotly-plot");
}

function getShapes(graphId, storeId) {
    const plotDiv = getPlotDiv(graphId)
    const shapes = plotDiv.layout && Array.isArray(plotDiv.layout.shapes)  ? plotDiv.layout.shapes : [];

    dash_clientside.set_props(storeId, {
        data: {
            graphId: graphId,
            shapes: shapes
        }
    });
}

window.dash_shared = window.dash_shared || {};

window.dash_shared.get_figure_shapes = function (wrapper_id) {
    const target = document.getElementById(wrapper_id);
    if (!target) {
        return [];
    }

    const gd = target.querySelector(".js-plotly-plot");
    if (!gd) {
        return [];
    }

    const plot_shapes = (gd.layout && gd.layout.shapes) || [];
    return plot_shapes;
};

window.dash_shared.get_figure_active_shape_idx = function(wrapper_id){

    // create a function that finds the active shape
    const find_active = function (element){
        return element.childNodes[0].hasOwnProperty('_ontouchstart');
    };

    // create a function to find the index of a shape
    const get_index = function(element){
        return element.childNodes[1].getAttribute("data-index");
    };

    // get the graph by the id
    const target = document.getElementById(wrapper_id)

    // get the active shape from the figure
    const shape_list = Array.from(target.getElementsByClassName('shape-group'));
    const active_shapes = shape_list.filter(find_active).map(get_index);
    return active_shapes;
};

window.dash_shared.build_delete_event = function(children, shapes){

    // create an event with the shape information in it
    const drop_complete = new CustomEvent('shapeDeletion', {
        bubbles: true,
        detail: {
            children: children,
            shapes: shapes,
        }
    });
    return drop_complete;
};

window.dash_clientside.clientside = {
    make_draggable: function (id, id2) {
        setTimeout(function () {
            var ele = document.getElementById(id)
            var ele2 = document.getElementById(id2)
            var drake = dragula([ele, ele2]);
            drake.on("drop", function (_el, target, source, sibling) {
                // a component has been dragged & dropped

                // get the order of the ids from the DOM
                var order_ids = Array.from(ele.children).map(function (child) {
                    return child.id;
                });
                const drop_complete = new CustomEvent('dropcomplete', {
                    bubbles: true,
                    detail: {
                        name: "Additional event infos",
                        children: order_ids
                    }
                });
                target.dispatchEvent(drop_complete)
                // How can I trigger an update on the children property
                // ???
            })
        }, 1)
        return window.dash_clientside.no_update
    },
    select_signals: function (textboxContent, id1, id2, target_id) {

        // check whether the textbox is empty
        if (textboxContent == null || textboxContent === ''){
            return window.dash_clientside.no_update;
        }

        // Clear any previous selections
        let selectedSignals = [];
        let remainingSignals = [];

        // Convert the textbox content to a regular expression
        let regexString = textboxContent.replace(/\*/g, '.*'); // Replace * with .* for wildcard
        let regex;
        try {
            regex = new RegExp(regexString);
        } catch (e) {
            console.error('Invalid regular expression:', e);
            return window.dash_clientside.no_update;
        }

        // replace the children into the containers
        const stay_container = document.getElementById(id1)
        let ignore_container = document.getElementById(id2)

        // Iterate through the signal list and match against the regex
        for (const child of stay_container.children) {
            var signalText = child.textContent.trim();
            if (regex.test(signalText)) {
                selectedSignals.push(child);
            } else {
                remainingSignals.push(child);
            }
        }

        // iterate through the selected signals that we need to delete
        selectedSignals.forEach(function(signalDiv) {
            // Move the selected signal to the other div
            ignore_container.appendChild(signalDiv);
            // Remove the selected signal from the current div
            // stay_container.removeChild(signalDiv_node);
        })

        // get the order of the ids from the DOM
        let target_container = document.getElementById(target_id)
        const order_ids = Array.from(target_container.children).map(function (child) {
            return child.id;
        });
        const drop_complete = new CustomEvent('dropcomplete', {
            bubbles: true,
            detail: {
                name: "Additional event infos",
                children: order_ids
            }
        });
        target_container.dispatchEvent(drop_complete)
        return window.dash_clientside.no_update
    },
    delete_stuff: function(id) {
        document.addEventListener("keydown", function(event) {
            if (event.key === 'Delete') {

                // get the graph by the id
                const target = document.getElementById(id)

                // create a function that finds the active shape
                const find_active = function (element){
                    return element.childNodes[0].hasOwnProperty('_ontouchstart')
                }

                // create a function to find the index of a shape
                const get_index = function(element){
                    return element.childNodes[1].getAttribute("data-index")
                }

                const shapes = Array.from(target.getElementsByClassName('shape-group')).filter(find_active).map(get_index)
                // create an event with the shape information in it
                const drop_complete = new CustomEvent('shapeDeletion', {
                    bubbles: true,
                    detail: {
                        children: shapes
                    }
                });
                console.log(drop_complete)
                target.dispatchEvent(drop_complete)
            }
            if (event.keyCode >= 48 && event.keyCode <= 57) {
                var element = document.getElementById('column-container')
                if (event.keyCode != 48) {
                    element = document.getElementById('signal-selection' + event.key);
                }

                if (element) {
                    element.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" });
                }
            }
        });
        return window.dash_clientside.no_update
    },
    delete_stuff_scatter: function(id, scatter_id) {
        document.addEventListener("keydown", function(event) {
            if (event.key === 'Delete') {

                // get the graph by the id
                const target = document.getElementById(id)

                // get the plot shapes
                const plot_shapes = window.dash_shared.get_figure_shapes(id)
                const active_shapes = window.dash_shared.get_figure_active_shape_idx(id)

                // create an event with the shape information in it
                const drop_complete = window.dash_shared.build_delete_event(active_shapes, plot_shapes)
                console.log(drop_complete)
                target.dispatchEvent(drop_complete)
            }
            if (/^[0-9]$/.test(event.key)) {
                const number = Number(event.key);
                let element = document.getElementById(scatter_id).children[number-1]
                if (number === 0) {
                    element = document.getElementById(id);
                }
                if (element) {
                    element.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" });
                }
            }
        });
        return window.dash_clientside.no_update
    },
    delete_all_stuff: function(button_id, id) {
        document.getElementById(button_id).addEventListener("click",
            function(n_clicks) {

                // get the graph by the id
                const target = document.getElementById(id)

                // get the plot shapes
                const plot_shapes = window.dash_shared.get_figure_shapes(id);

                // create an event with the shape information in it
                const drop_complete = window.dash_shared.build_delete_event('all', plot_shapes)
                console.log(drop_complete)
                target.dispatchEvent(drop_complete)
            });
        return window.dash_clientside.no_update;
    },
    delete_active_shape: function(button_id, id) {
        document.getElementById(button_id).addEventListener("click",
            function(n_clicks) {

                // get the graph by the id
                const target = document.getElementById(id)

                // get the plot shapes
                const plot_shapes = window.dash_shared.get_figure_shapes(id)
                const active_shapes = window.dash_shared.get_figure_active_shape_idx(id)

                // create an event with the shape information in it
                const drop_complete = window.dash_shared.build_delete_event(active_shapes, plot_shapes)
                console.log(drop_complete)
                target.dispatchEvent(drop_complete)
            });
        return window.dash_clientside.no_update;
    }
}