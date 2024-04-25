
document.getElementById("commence").addEventListener("click", function() {
    window.location.href = "/model";
});

function adjustZIndex() {
    var otherContent = document.getElementById('reveal');
    otherContent.style.zIndex = 50; // Adjust z-index value as needed
  }

  // Delay the execution of adjustZIndex function
  setTimeout(adjustZIndex, 5000); // Adjust delay time in milliseconds 
  
