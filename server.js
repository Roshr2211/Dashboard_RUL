const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname)));
app.use(express.json());

app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error from Flask server:', error.response?.data || error.message);
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            res.status(error.response.status).json({
                error: error.response.data?.error || 'An error occurred while making the prediction'
            });
        } else if (error.request) {
            // The request was made but no response was received
            res.status(503).json({ error: 'Unable to reach the prediction service' });
        } else {
            // Something happened in setting up the request that triggered an Error
            res.status(500).json({ error: 'An unexpected error occurred' });
        }
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});
