document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById('myCanvas');
  const ctx = canvas.getContext('2d');

  ctx.strokeStyle = 'black';
  ctx.lineWidth = 3;          

  let isDrawing = false;
  let paths = [];
  let currentPath = { x: [], y: [] };
  let intervalId = null;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);

    currentPath = { x: [], y: [] };

    currentPath.x.push(e.offsetX);
    currentPath.y.push(e.offsetY);

    lastX = e.offsetX;
    lastY = e.offsetY;

    intervalId = setInterval(() => {
      if (isDrawing) {
        currentPath.x.push(lastX); 
        currentPath.y.push(lastY);
      }
    }, 30);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();

      lastX = e.offsetX;
      lastY = e.offsetY;
    }
  });

  canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.closePath();

    clearInterval(intervalId);

    paths.push([currentPath.x, currentPath.y]);

    currentPath = { x: [], y: [] };
  });

  canvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    clearInterval(intervalId);
  });

  function saveDrawingAndClassify(event) {
    event.preventDefault();  // Prevent any default action

    const data = JSON.stringify(paths);

    fetch('http://127.0.0.1:5000/classify-drawing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data
    })
    .then(response => response.json())
    .then(result => {
      console.log(result);  // Log the response to check if it's received correctly

      // Display the predicted class
      document.getElementById('result').textContent = `Predicted Class: ${result.predicted_class}`;

      // Display class probabilities
      if (result.class_probabilities) {
        const probabilitiesText = Object.entries(result.class_probabilities)
          .map(([className, probability]) => `${className}: ${(probability * 100).toFixed(2)}%`)
          .join('<br>');
        document.getElementById('probabilities').innerHTML = `<strong>Class Probabilities:</strong><br>${probabilitiesText}`;
      }
    })
    .catch(error => console.error('Error:', error));
  }

  document.querySelector('.button2').addEventListener('click', saveDrawingAndClassify);
});
