const { app, BrowserWindow, session, Menu } = require('electron');
const path = require('path');
const http = require('http');
const fs = require('fs');

let mainWindow = null;
let server = null;
let serverPort = 0;

// MIME types for static asset serving
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
 * Embedded Lightweight Local Host Server
 * Guarantees Secure Context for Camera & Web Audio API
 */
function startLocalServer() {
  return new Promise((resolve, reject) => {
    const staticDir = path.join(__dirname, 'ui', 'static');

    server = http.createServer((req, res) => {
      let reqPath = decodeURI(req.url.split('?')[0]);
      if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

      const filePath = path.join(staticDir, reqPath);

      // Security check: ensure path is inside staticDir
      if (!filePath.startsWith(staticDir)) {
        res.writeHead(403);
        res.end('Access Denied');
        return;
      }

      fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
          // Fallback to index.html for SPA routing
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

        // Support HTTP Range requests for video/audio seeking
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
      console.log(`[AIPS Desktop Software] Local server running on http://127.0.0.1:${serverPort}`);
      resolve(serverPort);
    });

    server.on('error', (err) => {
      reject(err);
    });
  });
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1366,
    height: 860,
    minWidth: 1024,
    minHeight: 680,
    title: 'Human Emotion Recognition AI - Desktop Software',
    backgroundColor: '#0B0F19',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      backgroundThrottling: false
    }
  });

  // Remove native default menu for modern software aesthetic
  Menu.setApplicationMenu(null);

  // Automatically approve camera & microphone hardware access
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'mediaKeySystem', 'notifications', 'accessibilityEvents'];
    if (allowed.includes(permission)) {
      callback(true);
    } else {
      callback(true);
    }
  });

  session.defaultSession.setPermissionCheckHandler(() => true);

  const targetUrl = `http://127.0.0.1:${port}/index.html`;
  mainWindow.loadURL(targetUrl);

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
    console.error('[AIPS Desktop Software] Initialization error:', err);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (server) {
    server.close();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
