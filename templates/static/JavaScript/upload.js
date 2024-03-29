function upload() {
    var fileInput = document.getElementById('file');
    var file = fileInput.files[0];
    var formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById('result').innerHTML = data;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}