const { app, BrowserWindow, session, Menu, globalShortcut } = require('electron');
const path = require('path');
const http = require('http');
const https = require('https');
const fs = require('fs');

let mainWindow = null;
let server = null;
let serverPort = 0;

// Production Cloud URL (Synced to GitHub main branch via Vercel)
const CLOUD_PRODUCTION_URL = 'https://ai-based-emotion-recognition-system.vercel.app';

// MIME types for embedded local server
const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg'
};

/**
 * Embedded Local Offline Server
 * Guarantees 100% offline functionality whenever no internet is present
 */
function startLocalServer() {
  return new Promise((resolve, reject) => {
    const staticDir = path.join(__dirname, 'ui', 'static');

    server = http.createServer((req, res) => {
      let reqPath = decodeURI(req.url.split('?')[0]);
      if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

      const filePath = path.join(staticDir, reqPath);

      if (!filePath.startsWith(staticDir)) {
        res.writeHead(403);
        res.end('Access Denied');
        return;
      }

      fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
          const indexPath = path.join(staticDir, 'index.html');
          fs.readFile(indexPath, (readErr, data) => {
            if (readErr) {
              res.writeHead(404);
              res.end('Not Found');
            } else {
              res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
              res.end(data);
            }
          });
          return;
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/octet-stream';

        const range = req.headers.range;
        if (range && (contentType.startsWith('video/') || contentType.startsWith('audio/'))) {
          const totalSize = stats.size;
          const parts = range.replace(/bytes=/, '').split('-');
          const start = parseInt(parts[0], 10);
          const end = parts[1] ? parseInt(parts[1], 10) : totalSize - 1;

          if (start >= totalSize || end >= totalSize) {
            res.writeHead(416, { 'Content-Range': `bytes */${totalSize}` });
            res.end();
            return;
          }

          res.writeHead(206, {
            'Content-Range': `bytes ${start}-${end}/${totalSize}`,
            'Accept-Ranges': 'bytes',
            'Content-Length': (end - start) + 1,
            'Content-Type': contentType
          });

          fs.createReadStream(filePath, { start, end }).pipe(res);
        } else {
          res.writeHead(200, {
            'Content-Length': stats.size,
            'Content-Type': contentType,
            'Cache-Control': 'no-cache'
          });
          fs.createReadStream(filePath).pipe(res);
        }
      });
    });

    server.listen(0, '127.0.0.1', () => {
      serverPort = server.address().port;
      console.log(`[AIPS Offline Engine] Local fallback server active on http://127.0.0.1:${serverPort}`);
      resolve(serverPort);
    });

    server.on('error', (err) => reject(err));
  });
}

/**
 * Fast Cloud Connectivity Probe
 */
function checkCloudAvailability(url, timeoutMs = 2500) {
  return new Promise((resolve) => {
    try {
      const req = https.get(url, { timeout: timeoutMs }, (res) => {
        resolve(res.statusCode >= 200 && res.statusCode < 400);
      });
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
      req.on('error', () => resolve(false));
    } catch {
      resolve(false);
    }
  });
}

function createMainWindow(port) {
  const iconPath = path.join(__dirname, 'ui', 'static', 'demo_samples', 'app_icon.png');

  mainWindow = new BrowserWindow({
    width: 1366,
    height: 860,
    minWidth: 1024,
    minHeight: 680,
    title: 'Human Emotion Recognition AI',
    backgroundColor: '#0B0F19',
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      backgroundThrottling: false
    }
  });

  Menu.setApplicationMenu(null);

  // Permanent Media Stream Approvals (Camera & Microphone)
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'mediaKeySystem', 'notifications', 'accessibilityEvents'];
    callback(allowed.includes(permission) || true);
  });

  session.defaultSession.setPermissionCheckHandler(() => true);

  // Determine Target URL (Cloud Live-Sync vs Local Offline)
  checkCloudAvailability(CLOUD_PRODUCTION_URL).then((isOnline) => {
    const offlineUrl = `http://127.0.0.1:${port}/index.html`;

    if (isOnline) {
      console.log('[AIPS Software] Cloud Live-Sync Active: Connected to latest GitHub release');
      mainWindow.loadURL(CLOUD_PRODUCTION_URL).catch(() => {
        console.warn('[AIPS Software] Cloud load failed, falling back to local offline bundle');
        mainWindow.loadURL(offlineUrl);
      });
    } else {
      console.log('[AIPS Software] Offline Mode: Running bundled offline local engine');
      mainWindow.loadURL(offlineUrl);
    }
  });

  // Fail-safe fallback: If any page load fails, auto-route to local offline bundle
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    const offlineUrl = `http://127.0.0.1:${port}/index.html`;
    if (validatedURL !== offlineUrl) {
      console.warn(`[AIPS Software] Navigation error (${errorCode}: ${errorDescription}). Re-routing to offline bundle...`);
      mainWindow.loadURL(offlineUrl);
    }
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App Lifecycle
app.whenReady().then(async () => {
  try {
    const port = await startLocalServer();
    createMainWindow(port);

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createMainWindow(port);
      }
    });
  } catch (err) {
    console.error('[AIPS Software] Initialization error:', err);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (server) server.close();
  if (process.platform !== 'darwin') app.quit();
});
