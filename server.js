const express = require('express');
const path = require('path');
const axios = require('axios');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname)));
app.use(express.json());

const flaskServerUrl = 'http://127.0.0.1:5000';

app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post(`${flaskServerUrl}/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        handleAxiosError(error, res);
    }
});

app.get('/data', (req, res) => {
    // Spawn a child process to run the Python script and get the data
    const pythonProcess = spawn('python', ['generate_data_and_plot.py']);

    let dataString = '';

    pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        try {
            const jsonData = JSON.parse(dataString);
            res.json(jsonData);
        } catch (error) {
            console.error('Error parsing JSON:', error);
            console.log('Received data:', dataString);
            res.status(500).json({ error: 'Error parsing data from Python script' });
        }
    });
});

function handleAxiosError(error, res) {
    console.error('Error from Flask server:', error.response?.data || error.message);
    if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        res.status(error.response.status).json({
            error: error.response.data?.error || 'An error occurred while processing the request'
        });
    } else if (error.request) {
        // The request was made but no response was received
        res.status(503).json({ error: 'Unable to reach the Flask service' });
    } else {
        // Something happened in setting up the request that triggered an Error
        res.status(500).json({ error: 'An unexpected error occurred' });
    }
}

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});