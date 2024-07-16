const express = require('express');
const path = require('path');
const axios = require('axios');
const { spawn } = require('child_process');
const app = express();
const port = 3000;
const cors = require('cors');
app.use(cors());

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

app.post('/data', (req, res) => {
    const experiment = req.body.experiment;
    if (!experiment) {
        console.error("Missing 'experiment' in request data");
        return res.status(400).json({ error: "Missing 'experiment' in request data" });
    }

    console.log(`Processing experiment: ${experiment}`);

    const pythonProcess = spawn('python', ['generate_data_and_plot.py', experiment]);

    let dataString = '';
    let errorString = '';

    pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errorString += data.toString();
        console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        if (code !== 0) {
            console.error(`Python script error: ${errorString}`);
            return res.status(500).json({ error: 'Error in Python script', details: errorString });
        }
        try {
            const jsonData = JSON.parse(dataString);
            if (jsonData.error) {
                console.error(`Error in Python script: ${jsonData.error}`);
                return res.status(400).json(jsonData);
            }
            res.json(jsonData.result);
        } catch (error) {
            console.error('Error parsing JSON:', error);
            console.log('Received data:', dataString);
            res.status(500).json({ error: 'Error parsing data from Python script', details: dataString });
        }
    });
});

app.post('/getCycleCount', async (req, res) => {
    console.log('Received request for /getCycleCount:', req.body);
    try {
        console.log('Sending request to Flask server...');
        const response = await axios.post(`${flaskServerUrl}/getCycleCount`, req.body);
        // console.log('Received response from Flask server:', response.data);
        res.json(response.data);
    } catch (error) {
        console.error('Error in /getCycleCount route:', error.response?.data || error.message);
        handleAxiosError(error, res);
    }
});

app.post('/getDischargeCount', async (req, res) => {
    console.log('Received request for /getDischargeCount:', req.body);
    try {
        console.log('Sending request to Flask server...');
        const response = await axios.post(`${flaskServerUrl}/getDischargeCount`, req.body);
        // console.log('Received response from Flask server:', response.data);
        res.json(response.data);
    } catch (error) {
        console.error('Error in /getDischargeCount route:', error.response?.data || error.message);
        res.status(500).json({ error: 'Failed to fetch data from Flask server' });
    }
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