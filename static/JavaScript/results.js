// document.getElementById("image-link").addEventListener("click", function(event) {
//     event.preventDefault(); // Prevent the default behavior of the link

//     // Navigate back in the browser's history
//     window.history.back();
// });

// // script.js
// window.addEventListener('load', function() {
//     const confusionMatrixDiv = document.getElementById('confusionMatrix');
//     const loadingGif = document.createElement('img');
//     loadingGif.src = 'loading.gif'; // Replace 'loading.gif' with the path to your loading GIF
//     loadingGif.alt = 'Loading...';
//     confusionMatrixDiv.appendChild(loadingGif);

//     fetch('/get_confusion_matrix_data')
//     .then(response => response.json())
//     .then(data => {
//         // Remove the loading GIF
//         confusionMatrixDiv.removeChild(loadingGif);

//         // Display image
//         const imgElement = document.createElement('img');
//         imgElement.src = `data:image/png;base64, ${data.confusion_matrix}`;
//         confusionMatrixDiv.appendChild(imgElement);

//         // Display text
//         const resultsElement = document.getElementById('results');
//         resultsElement.textContent = data.results;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         // Remove the loading GIF and display an error message
//         confusionMatrixDiv.removeChild(loadingGif);
//         confusionMatrixDiv.textContent = 'Error loading image.';
//     });
// });


// // script.js
// fetch('/get_confusion_matrix_data')
// .then(response => response.json())
// .then(data => {
//     // Remove the loading message or animation
//     document.getElementById('loading').style.display = 'none';

//     // Display the confusion matrix image
//     const confusionMatrixDiv = document.getElementById('confusionMatrix');
//     const imgElement = document.createElement('img');
//     imgElement.src = `data:image/png;base64, ${data.confusion_matrix}`;
//     confusionMatrixDiv.appendChild(imgElement);
//     confusionMatrixDiv.style.display = 'block';

//     // Display the results
//     const resultsElement = document.getElementById('results');
//     resultsElement.textContent = data.results;
//     resultsElement.style.display = 'block';
// })
// .catch(error => {
//     console.error('Error:', error);
//     // Display an error message if there's an error fetching the data
//     document.getElementById('loading').textContent = 'Error loading data.';
// });


// window.addEventListener('load', function() {
//     document.getElementById('loader').style.display = 'none';
    
//     fetch('/result')
//     .then(response => response.json())
//     .then(data => {
//         document.getElementById('loader').style.display = 'none';
//         document.getElementById('results').style.display = 'block';
//         document.getElementById('results').innerText = data.results;
//          // Display image
//          const imgElement = document.createElement('img');
//          imgElement.src = `data:image/png;base64, ${data.confusion_matrix}`;
//          document.getElementById('confusionMatrix').appendChild(imgElement);
 
//          // Display text
//          const resultsElement = document.getElementById('results');
//          resultsElement.textContent = data.results;
//     })

    
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });


// results.js
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
            // Hide loading spinner
            document.getElementById("loader").style.display = "none";
        })
        .catch(error => console.error('Error fetching results:', error));
});