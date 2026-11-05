const https = require('https');
const apiKey = process.env.GEMINI_API_KEY || '';

const prompt = 'You are a disaster dispatch AI. User asks: "Find fastest safe ambulance route from VESIT to Zen Multi Speciality Hospital avoiding flooded streets". Nodes available: [N_CAMPUS_VESIT, N_HOSP_ZEN, N_CHEMBUR_NAKA, N_HOSP_SURANA]. Extract origin and destination node IDs. Reply in json: {"origin": "...", "destination": "..."}';

const postData = JSON.stringify({
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: { responseMimeType: 'application/json' }
});

const req = https.request('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=' + apiKey, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
}, (res) => {
    let data = '';
    res.on('data', c => data += c);
    res.on('end', () => {
        try {
            const j = JSON.parse(data);
            if (j.candidates && j.candidates[0]) {
                console.log('SUCCESS! Gemini response:');
                console.log(j.candidates[0].content.parts[0].text);
            } else {
                console.log('Error/Response:', data);
            }
        } catch(e) {
            console.error(e, data);
        }
    });
});
req.on('error', console.error);
req.write(postData);
req.end();
