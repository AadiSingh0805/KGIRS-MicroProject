const https = require('https');
const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, '..', 'data', 'damage_images');

function fetchJson(url) {
    return new Promise((resolve, reject) => {
        https.get(url, {
            headers: { 'User-Agent': 'CrisisNavBot/1.0 (academic microproject; contact@example.com)' }
        }, res => {
            let data = '';
            res.on('data', c => data += c);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch(e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

function downloadBinary(url, dest) {
    return new Promise((resolve, reject) => {
        https.get(url, {
            headers: { 'User-Agent': 'CrisisNavBot/1.0 (academic microproject; contact@example.com)' }
        }, res => {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                return downloadBinary(res.headers.location, dest).then(resolve).catch(reject);
            }
            if (res.statusCode !== 200) {
                return reject(new Error(`HTTP ${res.statusCode}`));
            }
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                const buf = Buffer.concat(chunks);
                fs.writeFileSync(dest, buf);
                resolve(buf.length);
            });
        }).on('error', reject);
    });
}

async function run() {
    const apiUrl = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:KatrinaNewOrleansFlooded_edit2.jpg|File:Collapsed_I35W_bridge_and_10th_Ave_bridge_from_air.jpg&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json';
    console.log('Fetching media info from Wikimedia Commons API...');
    const res = await fetchJson(apiUrl);
    const pages = res.query.pages;
    
    for (const pid in pages) {
        const page = pages[pid];
        const title = page.title;
        const imgInfo = page.imageinfo && page.imageinfo[0];
        if (!imgInfo) continue;
        
        const thumbUrl = imgInfo.thumburl || imgInfo.url;
        let outName = '';
        if (title.includes('Katrina')) {
            outName = 'sat_disaster_shell_colony_flood.png';
        } else {
            outName = 'sat_disaster_amar_mahal_collapse.png';
        }
        
        console.log(`Downloading ${title} from ${thumbUrl}...`);
        const dest = path.join(outDir, outName);
        const size = await downloadBinary(thumbUrl, dest);
        console.log(`[SUCCESS] Saved ${outName} (${size} bytes)`);
    }
}

run().catch(console.error);
