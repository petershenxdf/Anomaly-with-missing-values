let scatterplotDiv;  // Declare scatterplotDiv as a global variable

let all_points;
document.getElementById('dataset-select').addEventListener('change', function() {
    const dataset = this.value;
    if (dataset) {
        fetch('/scatterplot-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ dataset }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                mdsData = data.points;  // Store MDS data including imputed points
                //imputationData = data.imputation_data;  // Store imputation data (average + imputation points)
                all_points=data.all_points
                //console.log("all Data:", all_points);  // Log imputationData to inspect its structure
                drawPlotlyScatterplot(mdsData);  // Draw the scatterplot
            }
        });
    }
});

function drawPlotlyScatterplot(points) {
    console.log(points);
    scatterplotDiv = document.getElementById('scatterplot');

    console.log("Number of points received: ", points.length);

    const validPoints = points.filter(d => Array.isArray(d) && d.length > 1);  
    console.log("Valid points after filtering: ", validPoints.length);

    const xData = validPoints.map(d => d[0]);  // Extract x coordinates
    const yData = validPoints.map(d => d[1]);  // Extract y coordinates
    const dotSize = parseInt(getComputedStyle(scatterplotDiv).getPropertyValue('--scatterplot-dot-size'));

    // Set default opacity for points: Imputed points hidden at the start
    const markerOpacity = validPoints.map((_, index) => {
        if (all_points[index]?.isAverage) {
            return 1; // Show average imputed points by default
        } else if (all_points[index]?.isImputated) {
            //console.log(`Imputed Point ${index}: Initially Hidden`);
            return 0; // Hide imputed points by default (fully transparent)
        }
        return 1; // Show non-imputed points by default
    });

    // Set default colors for points:
    const markerColors = validPoints.map((_, index) => {
        if (all_points[index]?.isAverage) {
            return 'orange'; // Color for average imputed points
        }
        return 'steelblue'; // Default color for non-imputed points
    });

    const markerSymbols = validPoints.map((_, index) => {
        // Log the imputation data for each point to ensure we are accessing it properly
        //console.log(`Index ${index}:`, imputationData[index]);
    
        if (all_points[index]?.isAverage) {
            //console.log(`Point ${index} (Average): Shape = diamond`);
            return 'diamond'; // Shape for average imputed points
        } else if (all_points[index]?.isImputated) {
            return 'triangle-up';
        }
    
        console.log(`Point ${index} (Non-Imputed): Shape = circle`);
        return 'circle'; // Shape for non-imputed points
    });
    

    const trace = {
        x: xData,
        y: yData,
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: dotSize,
            color: markerColors,
            opacity: markerOpacity,
            symbol: markerSymbols
        }
    };

    const layout = {
        title: 'Scatterplot with Hover-based Imputation Visibility',
        xaxis: { title: 'X Axis' },
        yaxis: { title: 'Y Axis' },
        margin: { t: 40, r: 30, l: 50, b: 50 },
        height: 600,
        width: scatterplotDiv.offsetWidth,
        hovermode: 'closest',
        hoverdistance: 10,
        autosize: true,
    };

    Plotly.newPlot(scatterplotDiv, [trace], layout).then(() => {
        addClickListeners(scatterplotDiv, markerOpacity);  // Add hover listeners after plot is rendered
    });
}

function addClickListeners(scatterplotDiv, markerOpacity) {
    let activeClickIndex = null;

    scatterplotDiv.on('plotly_click', function(data) {
        console.log(data);
        const pointIndex = data.points[0].pointIndex;
        const isMainPoint = data.points[0].curveNumber === 0;

        //console.log("Hovered Point Index: ", pointIndex, "Is Main Point: ", isMainPoint);

        // Ensure hover effects are applied only to average imputed points
        if (isMainPoint && pointIndex !== activeClickIndex && all_points[pointIndex]?.isAverage) {
            activeClickIndex = pointIndex;

            const imputedIndices = all_points[pointIndex]?.imputedIndices || [];
            console.log("Imputed Indices to Show: ", imputedIndices);

            // Adjust opacity on hover: show hovered point and its imputed points, and make all others semi-transparent
            const newOpacity = markerOpacity.map((_, index) => {
                if (index === pointIndex || imputedIndices.includes(index)) {
                    return 1; // Show the hovered average point and its imputed points
                }else if(!all_points[index]?.isImputated || all_points[index]?.isAverage){
                    return 0.3;
                }
                 return 0
            });

            Plotly.restyle(scatterplotDiv, { 'marker.opacity': [newOpacity] });
        }
    });

    scatterplotDiv.on('plotly_doubleclick', function() {
        if (activeClickIndex !== null) {
            const imputedIndices = all_points[activeClickIndex]?.imputedIndices || [];
            //console.log("Unhovered Point Index: ", activeHoverIndex);dasdadadadad
            //console.log("Imputed Indices to Hide: ", imputedIndices);

            // Reset the opacity for imputed points to 0 and keep average points fully visible
            const resetOpacity = markerOpacity.map((_, index) => {
                if (!all_points[index].isAverage && all_points[index].isImputated) {
                    return 0;  // Hide imputed points
                }
                return 1; // Keep average imputed and non-imputed points visible
            });

            Plotly.restyle(scatterplotDiv, { 'marker.opacity': [resetOpacity] });
        }

        activeClickIndex = null;
    });
}

