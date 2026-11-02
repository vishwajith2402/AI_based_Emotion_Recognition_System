const { app, BrowserWindow, session, Menu, globalShortcut, nativeTheme } = require('electron');
const path = require('path');
const http = require('http');
const https = require('https');
const fs = require('fs');

// Apply system theme matching EmotiSync Frosted Pearl & Electric Indigo
if (nativeTheme) {
  nativeTheme.themeSource = 'light';
}

let mainWindow = null;
let server = null;
let serverPort = 0;

// Production Cloud URL (Live synced with GitHub repository via Vercel)
const CLOUD_PRODUCTION_URL = 'https://ai-based-emotion-recognition-system.vercel.app';

// Comprehensive MIME type mapping for embedded offline HTTP engine
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
  '.ogg': 'audio/ogg',
  '.wasm': 'application/wasm'
};

/**
 * Embedded Local Offline Server
 * Guarantees 100% offline functionality if internet is disconnected
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

        // Support HTTP Range requests for video and audio scrubbing
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
      console.log(`[EmotiSync] Local offline server listening on http://127.0.0.1:${serverPort}`);
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
  const assetIco = path.join(__dirname, 'ui', 'static', 'assets', 'app_icon.ico');
  const demoIco = path.join(__dirname, 'ui', 'static', 'demo_samples', 'app_icon.ico');
  const assetPng = path.join(__dirname, 'ui', 'static', 'assets', 'app_icon.png');
  const demoPng = path.join(__dirname, 'ui', 'static', 'demo_samples', 'app_icon.png');
  const icoPath = fs.existsSync(assetIco) ? assetIco : demoIco;
  const pngPath = fs.existsSync(assetPng) ? assetPng : demoPng;
  const iconPath = (process.platform === 'win32' && fs.existsSync(icoPath)) ? icoPath : pngPath;

  mainWindow = new BrowserWindow({
    width: 1366,
    height: 860,
    minWidth: 1024,
    minHeight: 680,
    title: 'EmotiSync — Emotion Recognition System (Facial • Voice)',
    backgroundColor: '#EEF2F6',
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    autoHideMenuBar: true,
    show: false,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#FFFFFF',
      symbolColor: '#0F172A',
      height: 38
    },
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      backgroundThrottling: false
    }
  });

  Menu.setApplicationMenu(null);

  // Auto-grant persistent camera & microphone hardware access without browser prompts
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'mediaKeySystem', 'notifications', 'accessibilityEvents'];
    callback(allowed.includes(permission) || true);
  });

  session.defaultSession.setPermissionCheckHandler(() => true);

  // Live Cloud-Sync vs Local Offline Strategy
  checkCloudAvailability(CLOUD_PRODUCTION_URL).then((isOnline) => {
    const offlineUrl = `http://127.0.0.1:${port}/index.html`;

    if (isOnline) {
      console.log('[EmotiSync] Cloud Live-Sync Active: Connected to live deployment.');
      mainWindow.loadURL(CLOUD_PRODUCTION_URL).catch(() => {
        console.warn('[EmotiSync] Cloud load failed, falling back to bundled offline engine.');
        mainWindow.loadURL(offlineUrl);
      });
    } else {
      console.log('[EmotiSync] Offline Mode Active: Running bundled local engine.');
      mainWindow.loadURL(offlineUrl);
    }
  });

  // Fail-safe fallback: If remote navigation fails, instantly revert to local offline server
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    const offlineUrl = `http://127.0.0.1:${port}/index.html`;
    if (validatedURL !== offlineUrl) {
      console.warn(`[EmotiSync] Network fail (${errorCode}: ${errorDescription}). Reverting to offline engine.`);
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

    // Development / presentation shortcuts: F5 reload, F11 fullscreen
    globalShortcut.register('F5', () => {
      if (mainWindow) mainWindow.reload();
    });
    globalShortcut.register('CommandOrControl+R', () => {
      if (mainWindow) mainWindow.reload();
    });
    globalShortcut.register('F11', () => {
      if (mainWindow) mainWindow.setFullScreen(!mainWindow.isFullScreen());
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createMainWindow(port);
      }
    });
  } catch (err) {
    console.error('[EmotiSync] Initialization error:', err);
    app.quit();
  }
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  if (server) server.close();
  if (process.platform !== 'darwin') app.quit();
});
