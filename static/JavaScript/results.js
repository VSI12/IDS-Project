document.addEventListener("DOMContentLoaded", function() {
    // Display loading spinner while waiting for results
    document.getElementById("loader").style.display = "block";

    // Fetch results from server
    fetch('/result')
        .then(response => response.json())
        .then(data => {
            // Update the results container with fetched data
            document.getElementById("confusionMatrix").innerHTML = `
                <h2>${data.results}</h2>
                <img src="data:image/png;base64,${data.confusion_matrix}" alt="Confusion Matrix">
            `;
            document.getElementById("metrics").innerHTML = `
                <h1>${data.Accuracy}</h1>
                <h1>${data.Precision}</h1>
                <h1>${data.Recall}</h1>
                <h1>${data.Fm}</h1>
                <h1>${data.Train}</h1>
                <h1>${data.Test}</h1>
            `;
            // Hide loading spinner
            document.getElementById("loader").style.display = "none";
        })
        .catch(error => console.error('Error fetching results:', error));
});