const https = require('https');
const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, '..', 'data', 'damage_images');
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

// Bounding boxes centered on each Chembur location (delta ~ 0.0028 deg for high resolution zoom)
const targets = [
    {
        filename: 'sat_vesit_campus.png',
        lat: 19.0463,
        lon: 72.8872,
        name: "VESIT Campus"
    },
    {
        filename: 'sat_zen_hospital.png',
        lat: 19.0582,
        lon: 72.9035,
        name: "Zen Multi Speciality Hospital"
    },
    {
        filename: 'sat_surana_hospital.png',
        lat: 19.0542,
        lon: 72.8912,
        name: "Surana Sethia Hospital"
    },
    {
        filename: 'sat_chembur_naka.png',
        lat: 19.0596,
        lon: 72.8988,
        name: "Chembur Naka Transit Hub"
    },
    {
        filename: 'sat_amar_mahal.png',
        lat: 19.0688,
        lon: 72.8945,
        name: "Amar Mahal Flyover"
    },
    {
        filename: 'sat_rcf_shelter.png',
        lat: 19.0465,
        lon: 72.9095,
        name: "RCF Sports Complex Shelter"
    },
    {
        filename: 'sat_inlaks_hospital.png',
        lat: 19.0515,
        lon: 72.8998,
        name: "Inlaks General Hospital"
    },
    {
        filename: 'sat_shell_colony.png',
        lat: 19.0652,
        lon: 72.8875,
        name: "Shell Colony Sector"
    },
    {
        filename: 'sat_tilak_nagar.png',
        lat: 19.0672,
        lon: 72.8912,
        name: "Tilak Nagar Station Cross"
    },
    {
        filename: 'sat_mankhurd_depot.png',
        lat: 19.0565,
        lon: 72.9315,
        name: "Mankhurd Supply Depot"
    },
    {
        filename: 'sat_shell_underpass.png',
        lat: 19.0665,
        lon: 72.8942,
        name: "Shell Colony Railway Subway (Flooded)"
    },
    {
        filename: 'sat_diamond_garden.png',
        lat: 19.0558,
        lon: 72.9015,
        name: "Diamond Garden Central Circle"
    },
    {
        filename: 'sat_mithi_culvert.png',
        lat: 19.0520,
        lon: 72.8850,
        name: "Mithi River Stormwater Culvert"
    }
];

function downloadTarget(t) {
    return new Promise((resolve, reject) => {
        const delta = 0.0030;
        const min_lon = (t.lon - delta).toFixed(5);
        const min_lat = (t.lat - delta * 0.75).toFixed(5);
        const max_lon = (t.lon + delta).toFixed(5);
        const max_lat = (t.lat + delta * 0.75).toFixed(5);
        const url = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=${min_lon},${min_lat},${max_lon},${max_lat}&bboxSR=4326&size=800,600&imageSR=4326&format=png&f=image`;

        https.get(url, (res) => {
            if (res.statusCode !== 200) {
                return reject(new Error(`Failed ${t.name}: status ${res.statusCode}`));
            }
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                const buffer = Buffer.concat(chunks);
                const dest = path.join(outDir, t.filename);
                fs.writeFileSync(dest, buffer);
                console.log(`[SUCCESS] Downloaded real satellite image for ${t.name} -> ${t.filename} (${buffer.length} bytes)`);
                resolve();
            });
        }).on('error', reject);
    });
}

async function run() {
    for (const t of targets) {
        try {
            await downloadTarget(t);
        } catch (err) {
            console.error(err.message);
        }
    }
    console.log("All satellite downloads finished!");
}

run();

