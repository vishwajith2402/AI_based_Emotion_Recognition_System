/**
 * VoiceAnalyzer - Real-Time, High-Performance Audio & Voice Visualizer Module
 * for Multimodal Affective Computing Applications.
 *
 * Implements:
 * 1. Web Audio API Capture (FFT 2048, float time-domain & byte frequency domain)
 * 2. Autocorrelation Pitch Detection (F0 in Hz, 65-550 Hz, parabolic interpolation)
 * 3. RMS Energy & Loudness in dB (relative to full scale)
 * 4. Pitch Inflection & Prosody Variance (rolling buffer std-dev)
 * 5. Dual-Layer Canvas Visualizer (64-bar cyber-neon spectrum + glowing cyan oscilloscope + HUD overlay)
 * 6. Prosodic Emotion Classifier with Softmax & EMA temporal smoothing
 * 7. Web Speech API STT with lexical sentiment reinforcement
 * 8. Fallback / Simulation Mode for zero-permission or silent environments
 */

class VoiceAnalyzer {
  constructor(options = {}) {
    this.options = {
      fftSize: 2048,
      smoothingTimeConstant: 0.8,
      minPitch: 65,
      maxPitch: 550,
      rmsSilenceThreshold: 0.015,
      pitchBufferSize: 30,
      emaAlpha: 0.20,
      numBars: 64,
      simulationMode: false,
      sttLanguage: 'en-US',
      ...options
    };

    // Web Audio API instances
    this.audioCtx = null;
    this.analyser = null;
    this.source = null;
    this.stream = null;
    this.isRunning = !!this.options.simulationMode;
    this.isSimulating = !!this.options.simulationMode;

    // Signal processing buffers
    this.timeDomainBuffer = new Float32Array(this.options.fftSize);
    this.frequencyBuffer = new Uint8Array(this.options.fftSize / 2);
    this.pitchHistory = [];

    // Real-time extracted acoustic features
    this.features = {
      pitch: 0,            // Fundamental frequency F0 (Hz)
      loudness: -80,       // Decibels relative to full scale (-80 to 0 dB)
      rms: 0.0,            // Root Mean Square energy (0.0 to 1.0)
      pitchVariance: 0.0,  // Standard deviation of rolling pitch history
      isVoiced: false      // True if harmonic vocal signal detected
    };

    // Emotion probability distribution
    this.emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral'];
    this.smoothedProbs = {
      Happy: 0.10,
      Sad: 0.10,
      Angry: 0.10,
      Surprise: 0.10,
      Neutral: 0.60
    };
    this.dominantEmotion = 'Neutral';
    this.emotionConfidence = 60.0;

    // Speech-to-Text & Lexical Intent
    this.transcript = '';
    this.interimTranscript = '';
    this.lexicalSentiment = 'Neutral';
    this.speechRecognizer = null;
    this.isListeningSTT = false;

    // Visualizer animation & simulation loop state
    this.animationFrameId = null;
    this.simPhase = 0;
    this.listeners = {
      update: [],
      transcript: [],
      emotion: []
    };

    // Initialize STT if supported
    this._initSpeechRecognition();
  }

  /**
   * Subscribe to analyzer events
   * @param {'update'|'transcript'|'emotion'} event
   * @param {Function} callback
   */
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  _emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(fn => {
        try { fn(data); } catch (e) { console.error('[VoiceAnalyzer] Callback error:', e); }
      });
    }
  }

  /**
   * Connect to user's microphone stream or create a new AudioContext
   * @param {MediaStream} [existingStream] Optional existing audio stream
   */
  async start(existingStream = null) {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        throw new Error('Web Audio API not supported in this browser environment');
      }

      if (!this.audioCtx) {
        this.audioCtx = new AudioContextClass();
      }

      if (this.audioCtx.state === 'suspended') {
        await this.audioCtx.resume();
      }

      // Configure AnalyserNode
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = this.options.fftSize;
      this.analyser.smoothingTimeConstant = this.options.smoothingTimeConstant;

      if (existingStream) {
        this.stream = existingStream;
      } else {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: false,
            autoGainControl: true
          },
          video: false
        });
      }

      this.source = this.audioCtx.createMediaStreamSource(this.stream);
      this.source.connect(this.analyser);

      this.isRunning = true;
      this.isSimulating = false;

      // Start STT transcription
      this.startSTT();

      console.log('[VoiceAnalyzer] Audio capture initialized. Sample Rate:', this.audioCtx.sampleRate);
      return true;
    } catch (err) {
      console.warn('[VoiceAnalyzer] Microphone initialization failed. Falling back to simulation mode:', err.message);
      this.isRunning = true;
      this.isSimulating = true;
      return false;
    }
  }

  /**
   * Stop audio capture and release resources
   */
  stop() {
    this.isRunning = false;
    if (this.source) {
      try { this.source.disconnect(); } catch (e) {}
      this.source = null;
    }
    if (this.stream && !this.options.keepStreamAlive) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      try { this.audioCtx.suspend(); } catch (e) {}
    }
    this.stopSTT();
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Primary frame update: extracts features, classifies prosody, emits data
   */
  update() {
    if (!this.isRunning) return this.features;

    if (this.isSimulating || !this.analyser) {
      this._updateSimulation();
    } else {
      // Capture Time-Domain PCM and Frequency-Domain spectrum
      this.analyser.getFloatTimeDomainData(this.timeDomainBuffer);
      this.analyser.getByteFrequencyData(this.frequencyBuffer);

      // Acoustic Feature Extraction
      this._extractFeatures(this.timeDomainBuffer, this.audioCtx.sampleRate);
    }

    // Emotion Classification from acoustic features & lexical cues
    const emotionResult = this.classifyEmotion();

    const metrics = {
      ...this.features,
      emotion: emotionResult.dominant,
      confidence: emotionResult.confidence,
      probabilities: emotionResult.probabilities,
      transcript: this.transcript,
      interim: this.interimTranscript
    };

    this._emit('update', metrics);
    return metrics;
  }

  /**
   * Real-time acoustic feature extraction
   * @param {Float32Array} buffer 
   * @param {number} sampleRate 
   */
  _extractFeatures(buffer, sampleRate) {
    const len = buffer.length;

    // 1. Calculate Root Mean Square (RMS) energy
    let sumSquares = 0;
    for (let i = 0; i < len; i++) {
      sumSquares += buffer[i] * buffer[i];
    }
    const rms = Math.sqrt(sumSquares / len);
    this.features.rms = rms;

    // 2. Sound Level / Loudness (dB relative to full scale)
    // Formula: Loudness (dB) = max(-80, round(20 * log10(RMS + 1e-4)))
    const loudness = Math.max(-80, Math.round(20 * Math.log10(rms + 1e-4)));
    this.features.loudness = loudness;

    // 3. Trim silence/noise below RMS threshold
    if (rms < this.options.rmsSilenceThreshold) {
      this.features.isVoiced = false;
      this.features.pitch = 0;
      return;
    }

    // 4. Autocorrelation Pitch Detection with Parabolic Interpolation
    const minPeriod = Math.floor(sampleRate / this.options.maxPitch);
    const maxPeriod = Math.ceil(sampleRate / this.options.minPitch);

    const r = new Float32Array(maxPeriod + 2);
    let bestR = -1;
    let bestLag = -1;

    for (let lag = minPeriod; lag <= maxPeriod; lag++) {
      let sum = 0;
      for (let i = 0; i < len - lag; i++) {
        sum += buffer[i] * buffer[i + lag];
      }
      r[lag] = sum;
      if (sum > bestR) {
        bestR = sum;
        bestLag = lag;
      }
    }

    if (bestLag <= minPeriod || bestLag >= maxPeriod || bestR <= 0) {
      this.features.isVoiced = false;
      this.features.pitch = 0;
      return;
    }

    // Parabolic sub-sample interpolation around best lag
    const alpha = r[bestLag - 1];
    const beta = r[bestLag];
    const gamma = r[bestLag + 1];
    const denom = alpha - 2 * beta + gamma;
    let delta = 0;
    if (Math.abs(denom) > 1e-6) {
      delta = 0.5 * (alpha - gamma) / denom;
    }
    const truePeriod = bestLag + delta;
    const f0 = sampleRate / truePeriod;

    if (f0 >= this.options.minPitch && f0 <= this.options.maxPitch) {
      this.features.isVoiced = true;
      this.features.pitch = Math.round(f0);

      // Pitch Inflection & Prosody Variance (rolling buffer of last 30 samples)
      this.pitchHistory.push(f0);
      if (this.pitchHistory.length > this.options.pitchBufferSize) {
        this.pitchHistory.shift();
      }

      if (this.pitchHistory.length >= 3) {
        const mean = this.pitchHistory.reduce((a, b) => a + b, 0) / this.pitchHistory.length;
        const varianceSum = this.pitchHistory.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
        this.features.pitchVariance = Math.round(Math.sqrt(varianceSum / this.pitchHistory.length) * 10) / 10;
      }
    } else {
      this.features.isVoiced = false;
      this.features.pitch = 0;
    }
  }

  /**
   * Prosodic Emotion Classification with Softmax & EMA Temporal Smoothing
   */
  classifyEmotion() {
    const { pitch, loudness, pitchVariance, isVoiced, rms } = this.features;

    // Base logits
    const logits = {
      Happy: 0.15,
      Sad: 0.15,
      Angry: 0.15,
      Surprise: 0.15,
      Neutral: 0.40
    };

    if (isVoiced && rms >= this.options.rmsSilenceThreshold) {
      // 1. High Pitch (>210 Hz) + High Loudness (>-24 dB) + High Variance (>28) => Happy / Surprise
      if (pitch > 210 && loudness > -24 && pitchVariance > 28) {
        logits.Happy += 3.4;
        logits.Surprise += 2.9;
      } else if (pitch > 195 && loudness > -28) {
        logits.Happy += 1.8;
      }

      // 2. High Loudness (>-20 dB) + Mid Pitch (110-220 Hz) + Low Variance (<=28) => Angry
      if (loudness > -20 && pitch >= 110 && pitch <= 220 && pitchVariance <= 28) {
        logits.Angry += 3.6;
      } else if (loudness > -22 && pitchVariance <= 20) {
        logits.Angry += 1.9;
      }

      // 3. Low Pitch (<135 Hz) + Low Loudness (<-32 dB) + Monotone (variance <= 18) => Sad
      if (pitch < 135 && loudness < -32 && pitchVariance <= 18) {
        logits.Sad += 3.8;
      } else if (pitch < 145 && loudness < -30) {
        logits.Sad += 1.7;
      }

      // 4. Moderate Pitch + Moderate Loudness => Neutral
      if (pitch >= 135 && pitch <= 205 && loudness >= -34 && loudness <= -22 && pitchVariance <= 25) {
        logits.Neutral += 2.6;
      }
    } else {
      // Silence / unvoiced defaults strongly to Neutral
      logits.Neutral += 3.2;
    }

    // Lexical reinforcement from spoken transcription
    if (this.lexicalSentiment && this.lexicalSentiment !== 'Neutral') {
      if (logits[this.lexicalSentiment] !== undefined) {
        logits[this.lexicalSentiment] += 2.2;
      }
    }

    // Softmax normalization
    const maxLogit = Math.max(...Object.values(logits));
    let sumExp = 0;
    const expScores = {};
    for (const em of this.emotions) {
      expScores[em] = Math.exp(logits[em] - maxLogit);
      sumExp += expScores[em];
    }

    const currentProbs = {};
    for (const em of this.emotions) {
      currentProbs[em] = expScores[em] / sumExp;
    }

    // Exponential Moving Average (EMA) temporal smoothing (alpha = 0.20)
    const alpha = this.options.emaAlpha;
    let dominant = 'Neutral';
    let maxP = 0;

    for (const em of this.emotions) {
      this.smoothedProbs[em] = alpha * currentProbs[em] + (1 - alpha) * this.smoothedProbs[em];
      if (this.smoothedProbs[em] > maxP) {
        maxP = this.smoothedProbs[em];
        dominant = em;
      }
    }

    this.dominantEmotion = dominant;
    this.emotionConfidence = Math.round(maxP * 100);

    const result = {
      dominant: this.dominantEmotion,
      confidence: this.emotionConfidence,
      probabilities: { ...this.smoothedProbs }
    };

    this._emit('emotion', result);
    return result;
  }

  /**
   * Dual-Layer Visualizer Render Function
   * Renders:
   * 1. Technical Cyber-Grid
   * 2. 64-Bar Equalizer with Violet (#8b5cf6) to Cyan (#00f2fe) vertical gradient
   * 3. Glowing Cyan Oscilloscope Waveform (2.5px, 8px drop-shadow blur)
   * 4. Floating HUD Pills (Pitch F0, Loudness dB, RMS Energy)
   * @param {HTMLCanvasElement} canvas
   */
  drawVisualizer(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI crisp rendering
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(rect.width * dpr) || canvas.width;
    const targetH = Math.round(rect.height * dpr) || canvas.height;

    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    const w = canvas.width;
    const h = canvas.height;

    // 1. Clear background
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#080c16';
    ctx.fillRect(0, 0, w, h);

    // 2. Subtle Technical Cyber-Grid
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1 * dpr;

    const gridStepX = 38 * dpr;
    const gridStepY = 28 * dpr;

    ctx.beginPath();
    for (let x = gridStepX; x < w; x += gridStepX) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = gridStepY; y < h; y += gridStepY) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
    ctx.restore();

    // 3. Layer 1 (Background): 64-Bar Frequency Spectrum Equalizer
    const numBars = this.options.numBars;
    const barGap = 2.5 * dpr;
    const totalGap = (numBars - 1) * barGap;
    const barWidth = Math.max(1, (w - totalGap - 16 * dpr) / numBars);
    const startX = 8 * dpr;
    const maxBarH = h * 0.42;

    const barGrad = ctx.createLinearGradient(0, h, 0, h - maxBarH);
    barGrad.addColorStop(0, '#8b5cf6');   // Aurora Violet bottom
    barGrad.addColorStop(0.5, '#6366f1'); // Indigo middle
    barGrad.addColorStop(1, '#00f2fe');   // Cyber Cyan top

    ctx.fillStyle = barGrad;

    const binCount = this.frequencyBuffer.length;
    for (let i = 0; i < numBars; i++) {
      // Map to logarithmic / voice perceptual bin distribution
      const binIdx = Math.min(binCount - 1, Math.floor(Math.pow(i / numBars, 1.4) * (binCount * 0.45)));
      const rawVal = this.frequencyBuffer[binIdx] || 0;
      const normalized = rawVal / 255.0;
      const barH = Math.max(2 * dpr, normalized * maxBarH);

      const bx = startX + i * (barWidth + barGap);
      const by = h - barH;

      ctx.fillRect(bx, by, barWidth, barH);
    }

    // 4. Layer 2 (Foreground): Real-time Glowing Oscilloscope Waveform Line
    const waveY = h * 0.42; // Vertically centered
    const bufferLen = this.timeDomainBuffer.length;
    const sliceWidth = w / bufferLen;

    ctx.save();
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2.5 * dpr;
    ctx.shadowColor = '#00f2fe';
    ctx.shadowBlur = 8 * dpr;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    let x = 0;
    for (let i = 0; i < bufferLen; i++) {
      const v = this.timeDomainBuffer[i]; // -1.0 to 1.0
      // Scale amplitude for clean, expressive visualization
      const y = waveY + (v * (h * 0.35));

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }
    ctx.lineTo(w, waveY);
    ctx.stroke();
    ctx.restore();

    // 5. Layer 3: Technical HUD Overlay Badges (Pills)
    this._drawHudPills(ctx, w, h, dpr);
  }

  /**
   * Render real-time HUD badge pills matching exact design specifications
   */
  _drawHudPills(ctx, w, h, dpr) {
    const pillW = 86 * dpr;
    const pillH = 38 * dpr;
    const pillY = h - pillH - 14 * dpr;
    const radius = 6 * dpr;

    const pills = [
      {
        x: 16 * dpr,
        label: 'PITCH (F₀)',
        val: `${this.features.pitch} Hz`
      },
      {
        x: Math.round(w / 2 - pillW / 2),
        label: 'LOUDNESS',
        val: `${this.features.loudness} dB`
      },
      {
        x: w - pillW - 16 * dpr,
        label: 'ENERGY',
        val: `${this.features.rms.toFixed(2)}`
      }
    ];

    ctx.save();
    for (const p of pills) {
      // Rounded pill backdrop
      ctx.fillStyle = 'rgba(11, 17, 32, 0.82)';
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
      ctx.lineWidth = 1 * dpr;

      ctx.beginPath();
      ctx.roundRect(p.x, pillY, pillW, pillH, radius);
      ctx.fill();
      ctx.stroke();

      // Top label (small gray caps)
      ctx.font = `600 ${8.5 * dpr}px "SF Mono", "Fira Code", monospace, sans-serif`;
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(p.label, p.x + pillW / 2, pillY + 6 * dpr);

      // Bottom value (violet/cyan accent font)
      ctx.font = `700 ${12 * dpr}px "SF Mono", "Fira Code", monospace, sans-serif`;
      ctx.fillStyle = '#a78bfa';
      ctx.textBaseline = 'bottom';
      ctx.fillText(p.val, p.x + pillW / 2, pillY + pillH - 5 * dpr);
    }
    ctx.restore();
  }

  /**
   * Realistic audio simulation when hardware microphone is inaccessible or silent
   */
  _updateSimulation() {
    this.simPhase += 0.04;
    const time = this.simPhase;

    // Simulate realistic speech activity cycles (talking for 3s, pause for 1s)
    const voiceEnvelope = Math.max(0, Math.sin(time * 0.7) * 0.85 + Math.sin(time * 1.3) * 0.25);
    const isSimVoiced = voiceEnvelope > 0.18;

    const len = this.timeDomainBuffer.length;
    const simFreq = 160 + Math.sin(time * 0.9) * 45; // Vary pitch between 115-205 Hz

    for (let i = 0; i < len; i++) {
      if (isSimVoiced) {
        const fundamental = Math.sin((i / len) * simFreq * 0.2 + time * 3) * voiceEnvelope * 0.45;
        const harmonic = Math.sin((i / len) * simFreq * 0.4 + time * 5) * voiceEnvelope * 0.18;
        const noise = (Math.random() - 0.5) * 0.03;
        this.timeDomainBuffer[i] = fundamental + harmonic + noise;
      } else {
        // Resting baseline whisper
        this.timeDomainBuffer[i] = Math.sin((i / len) * 2 + time) * 0.008;
      }
    }

    // Simulate frequency bars
    const freqLen = this.frequencyBuffer.length;
    for (let i = 0; i < freqLen; i++) {
      if (isSimVoiced) {
        const falloff = Math.exp(-i / 80);
        const flutter = Math.sin(i * 0.2 + time * 4) * 0.3 + 0.7;
        this.frequencyBuffer[i] = Math.floor(voiceEnvelope * falloff * flutter * 210);
      } else {
        this.frequencyBuffer[i] = Math.floor(Math.random() * 8);
      }
    }

    const rms = isSimVoiced ? 0.04 + voiceEnvelope * 0.12 : 0.004;
    this.features.rms = rms;
    this.features.loudness = Math.max(-80, Math.round(20 * Math.log10(rms + 1e-4)));
    this.features.pitch = isSimVoiced ? Math.round(simFreq) : 0;
    this.features.isVoiced = isSimVoiced;

    if (isSimVoiced) {
      this.pitchHistory.push(this.features.pitch);
      if (this.pitchHistory.length > this.options.pitchBufferSize) {
        this.pitchHistory.shift();
      }
      if (this.pitchHistory.length >= 3) {
        const mean = this.pitchHistory.reduce((a, b) => a + b, 0) / this.pitchHistory.length;
        const varianceSum = this.pitchHistory.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
        this.features.pitchVariance = Math.round(Math.sqrt(varianceSum / this.pitchHistory.length) * 10) / 10;
      }
    }
  }

  /**
   * Web Speech API continuous transcription setup
   */
  _initSpeechRecognition() {
    const SpeechRecognition = typeof window !== 'undefined' ? (window.SpeechRecognition || window.webkitSpeechRecognition || null) : null;
    if (!SpeechRecognition) return;

    try {
      this.speechRecognizer = new SpeechRecognition();
      this.speechRecognizer.continuous = true;
      this.speechRecognizer.interimResults = true;
      this.speechRecognizer.lang = this.options.sttLanguage || 'en-US';

      this.speechRecognizer.onresult = (event) => {
        let finalStr = '';
        let interimStr = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const res = event.results[i];
          if (res && res[0]) {
            const txt = res[0].transcript;
            if (res.isFinal) finalStr += txt + ' ';
            else interimStr += txt;
          }
        }

        if (finalStr.trim()) {
          this.transcript = (this.transcript + ' ' + finalStr).trim();
        }
        this.interimTranscript = interimStr;

        const currentFull = (this.transcript + ' ' + interimStr).trim();
        this.lexicalSentiment = this._classifyLexicalSentiment(currentFull);

        this._emit('transcript', {
          transcript: this.transcript,
          interim: this.interimTranscript,
          combined: currentFull,
          sentiment: this.lexicalSentiment
        });
      };

      this.speechRecognizer.onerror = () => {};
      this.speechRecognizer.onend = () => {
        if (this.isRunning && this.isListeningSTT) {
          try { this.speechRecognizer.start(); } catch (e) {}
        }
      };
    } catch (err) {
      console.warn('[VoiceAnalyzer] SpeechRecognition initialization error:', err);
    }
  }

  startSTT() {
    if (!this.speechRecognizer) return;
    this.isListeningSTT = true;
    try {
      this.speechRecognizer.lang = this.options.sttLanguage || 'en-US';
      this.speechRecognizer.start();
    } catch (e) {}
  }

  stopSTT() {
    this.isListeningSTT = false;
    if (this.speechRecognizer) {
      try { this.speechRecognizer.stop(); } catch (e) {}
    }
  }

  setLanguage(bcp47Lang) {
    this.options.sttLanguage = bcp47Lang;
    if (this.speechRecognizer) {
      try {
        this.speechRecognizer.lang = bcp47Lang;
        if (this.isListeningSTT) {
          this.speechRecognizer.abort();
        }
      } catch (e) {}
    }
  }

  /**
   * Multilingual lexical sentiment analysis
   */
  _classifyLexicalSentiment(text) {
    if (!text || typeof text !== 'string') return 'Neutral';
    const lower = text.toLowerCase();

    const angry = ['angry', 'mad', 'furious', 'hate', 'rage', 'shut up', 'terrible', 'annoying', 'irritated', 'stop it', 'worst', 'stupid', 'horrible', 'gussa', 'krodh', 'naraz', 'bakwas', 'kovam', 'erichal', 'kopam', 'enojado', 'odio', 'wütend', 'colère'];
    const sad = ['sad', 'unhappy', 'crying', 'depressed', 'sorrow', 'grief', 'lonely', 'miserable', 'heartbroken', 'pain', 'gloomy', 'tears', 'upset', 'hurt', 'udaas', 'dukhi', 'dard', 'kavalai', 'varutham', 'badha', 'triste', 'traurig', 'chagrin'];
    const happy = ['happy', 'glad', 'joy', 'smile', 'great', 'awesome', 'wonderful', 'excellent', 'love', 'fantastic', 'amazing', 'good', 'pleased', 'cheerful', 'excited', 'yay', 'khush', 'badhiya', 'mast', 'santhosham', 'super', 'arumai', 'feliz', 'content', 'glücklich'];
    const surprise = ['wow', 'omg', 'surprise', 'surprised', 'unbelievable', 'astonished', 'shocked', 'whoa', 'really', 'no way', 'unexpected', 'arey', 'waah', 'hairan', 'aacharyam', 'guau', 'ouah'];

    let scores = { Angry: 0, Sad: 0, Happy: 0, Surprise: 0 };
    angry.forEach(w => { if (lower.includes(w)) scores.Angry += 2; });
    sad.forEach(w => { if (lower.includes(w)) scores.Sad += 2; });
    happy.forEach(w => { if (lower.includes(w)) scores.Happy += 2; });
    surprise.forEach(w => { if (lower.includes(w)) scores.Surprise += 2; });

    let best = 'Neutral';
    let max = 0;
    for (const [em, cnt] of Object.entries(scores)) {
      if (cnt > max) {
        max = cnt;
        best = em;
      }
    }
    return best;
  }
}

// Global & CommonJS / ES module export compatibility
if (typeof window !== 'undefined') {
  window.VoiceAnalyzer = VoiceAnalyzer;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { VoiceAnalyzer };
}
