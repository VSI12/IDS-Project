document.getElementById("image-link").addEventListener("click", function(event) {
    event.preventDefault(); // Prevent the default behavior of the link

    // Navigate back in the browser's history
    window.history.back();
});