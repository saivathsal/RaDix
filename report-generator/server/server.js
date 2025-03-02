const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const axios = require("axios");
const fs = require("fs");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("uploads"));

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});
const upload = multer({ storage });

// Endpoint to upload the image and send it to the Python API
app.post("/upload", upload.single("file"), async (req, res) => {
  const filePath = path.join(__dirname, "uploads", req.file.filename);

  try {
    // Send the uploaded image to the Python Flask API
    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath));

    const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
      headers: formData.getHeaders(),
    });

    // Delete the file after prediction
    fs.unlinkSync(filePath);

    // Respond with the prediction result
    res.json({
      prediction: response.data.prediction,
      file: req.file.filename,
    });
  } catch (error) {
    console.error("Error during prediction:", error.message || error.response?.data || error);
    res.status(500).json({
      error: "Prediction failed.",
      details: error.response?.data || error.message,
    });
  }
});

// Start the server
app.listen(5001, () => {
  console.log("Node.js server running on port 5001");
  console.log("Ensure that Flask server is running on port 5000");
});

