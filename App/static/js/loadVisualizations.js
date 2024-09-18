console.log('loaded');

function loadVisualization(url, containerId, scriptUrl) {
    fetch(url)
        .then(response => response.text())
        .then(data => {
            document.getElementById(containerId).innerHTML = data;

            // Manually load the corresponding JS file for the visualization
            if (scriptUrl) {
                var script = document.createElement('script');
                script.src = scriptUrl;
                document.body.appendChild(script);
            }
        })
        .catch(error => console.error('Error loading visualization:', error));
}

// Load the visualizations into their respective containers
window.onload = function() {
    loadVisualization('/scatterplot', 'scatterplot-container', '/static/js/scatterplot.js');
    loadVisualization('/bar-chart', 'bar-chart-container', '/static/js/bar-chart.js');
    loadVisualization('/line-chart', 'line-chart-container', '/static/js/line-chart.js');
};
