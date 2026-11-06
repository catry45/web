/**
 * Traductor LSM – Interfaz Web
 * Conecta con el backend FastAPI (WebSocket) que usa los modelos reales.
 */

class TraductorLSM {
  constructor() {
    this.video = document.getElementById('webcam');
    this.canvas = document.getElementById('overlay');
    this.ctx = this.canvas.getContext('2d');
    this.guideOverlay = document.getElementById('guideOverlay');
    this.btnStart = document.getElementById('btnStartCamera');
    this.btnToggle = document.getElementById('btnToggleCamera');
    this.btnSwitch = document.getElementById('btnSwitchCamera');
    this.statusDot = document.getElementById('statusDot');
    this.statusText = document.getElementById('statusText');
    this.fpsBadge = document.getElementById('fpsBadge');
    this.currentPrediction = document.getElementById('currentPrediction');
    this.confidenceBadge = document.getElementById('confidenceBadge');
    this.lastUpdate = document.getElementById('lastUpdate');
    this.translatedText = document.getElementById('translatedText');
    this.historyList = document.getElementById('historyList');
    this.statDetections = document.getElementById('statDetections');
    this.statLetters = document.getElementById('statLetters');
    this.statSession = document.getElementById('statSession');
    this.motorBadge = document.getElementById('motorBadge');

    this.stream = null;
    this.isRunning = false;
    this.facingMode = 'user';
    this.ws = null;
    this.wsReady = false;
    this.sendInterval = null;
    this.sessionStart = null;
    this.sessionInterval = null;
    this.detections = 0;
    this.lettersCount = 0;
    this.history = [];
    this.currentText = '';
    this.lastAccepted = null;
    this.frameCount = 0;
    this.lastFpsUpdate = performance.now();
    this.fps = 0;

    this.captureCanvas = document.createElement('canvas');
    this.captureCtx = this.captureCanvas.getContext('2d');

    this.bindEvents();
    this.checkBackend();
  }

  bindEvents() {
    this.btnStart.addEventListener('click', () => this.startCamera());
    this.btnToggle.addEventListener('click', () => this.stopCamera());
    this.btnSwitch.addEventListener('click', () => this.switchCamera());
    document.getElementById('btnClearText').addEventListener('click', () => this.clearText());
    document.getElementById('btnCopyText').addEventListener('click', () => this.copyText());
    document.getElementById('btnClearHistory').addEventListener('click', () => this.clearHistory());

    document.addEventListener('keydown', (e) => {
      if (e.key === ' ' && e.target === document.body) {
        e.preventDefault();
        this.isRunning ? this.stopCamera() : this.startCamera();
      }
    });
  }

  async checkBackend() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        this.showToast('Backend conectado ✓', 'success');
        console.log('Clases disponibles:', data.modelos);
      }
    } catch {
      this.showToast('Backend no disponible. Ejecuta: python backend/server.py', 'error');
    }
  }

  async startCamera() {
    try {
      this.updateStatus('Solicitando cámara...', 'yellow');
      const constraints = {
        video: { facingMode: this.facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      };
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();

      this.resizeCanvas();
      window.addEventListener('resize', () => this.resizeCanvas());

      this.isRunning = true;
      document.body.classList.add('camera-active');
      this.guideOverlay.style.opacity = '0';
      this.guideOverlay.style.pointerEvents = 'none';

      this.updateStatus('Conectando IA...', 'yellow');
      await this.connectWebSocket();

      this.sessionStart = Date.now();
      this.sessionInterval = setInterval(() => this.updateSessionTime(), 1000);
      this.sendInterval = setInterval(() => this.sendFrame(), 90);
      this.loopFps();

      this.showToast('Cámara + IA activadas', 'success');
    } catch (err) {
      console.error(err);
      this.updateStatus('Error de cámara', 'red');
      let msg = 'No se pudo acceder a la cámara.';
      if (err.name === 'NotAllowedError') msg = 'Permiso de cámara denegado.';
      if (err.name === 'NotFoundError') msg = 'No se encontró cámara.';
      this.showToast(msg, 'error');
    }
  }

  stopCamera() {
    this.isRunning = false;
    if (this.sendInterval) { clearInterval(this.sendInterval); this.sendInterval = null; }
    if (this.sessionInterval) { clearInterval(this.sessionInterval); this.sessionInterval = null; }
    if (this.ws) { try { this.ws.close(); } catch {} this.ws = null; this.wsReady = false; }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    document.body.classList.remove('camera-active');
    this.guideOverlay.style.opacity = '1';
    this.guideOverlay.style.pointerEvents = 'auto';
    this.updateStatus('Cámara apagada', 'gray');
    this.fpsBadge.textContent = '-- FPS';
    this.showToast('Sesión detenida', 'info');
  }

  async switchCamera() {
    this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
    this.stopCamera();
    await new Promise(r => setTimeout(r, 350));
    this.startCamera();
  }

  resizeCanvas() {
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
  }

  connectWebSocket() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const url = `${proto}://${location.host}/ws/traducir`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.wsReady = true;
        this.updateStatus('IA activa', 'green');
        resolve();
      };

      this.ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.tipo === 'prediccion') this.handlePrediction(msg);
          else if (msg.tipo === 'error') this.showToast(msg.mensaje, 'error');
        } catch (e) {
          console.warn('Mensaje WS inválido', e);
        }
      };

      this.ws.onerror = () => {
        this.updateStatus('Error de conexión', 'red');
        this.showToast('No se pudo conectar al backend WebSocket', 'error');
        reject(new Error('WS error'));
      };

      this.ws.onclose = () => {
        this.wsReady = false;
        if (this.isRunning) this.updateStatus('IA desconectada', 'yellow');
      };

      setTimeout(() => {
        if (!this.wsReady) reject(new Error('Timeout WS'));
      }, 5000);
    });
  }

  sendFrame() {
    if (!this.wsReady || !this.isRunning || this.video.readyState < 2) return;
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) return;

    this.captureCanvas.width = w;
    this.captureCanvas.height = h;
    this.captureCtx.drawImage(this.video, 0, 0, w, h);
    const dataUrl = this.captureCanvas.toDataURL('image/jpeg', 0.55);

    try {
      this.ws.send(JSON.stringify({
        frame: dataUrl,
        landmarks: true,
        devolver_frame: false
      }));
    } catch (e) {
      console.warn('Error enviando frame', e);
    }
  }

  handlePrediction(msg) {
    if (this.motorBadge) {
      this.motorBadge.textContent = msg.motor || '—';
      this.motorBadge.className = 'text-xs font-mono px-2 py-0.5 rounded-md ' +
        (msg.motor === 'Dinámico'
          ? 'bg-emerald-500/20 text-emerald-400'
          : 'bg-sky-500/20 text-sky-400');
    }

    const valor = msg.oracion || msg.prediccion;
    if (!valor) return;

    if (valor === this.lastAccepted) {
      this.updateConfidence(msg.confianza);
      return;
    }

    if (msg.oracion && msg.oracion !== this.lastAccepted) {
      this.lastAccepted = msg.oracion;
      this.onNewDetection(msg.oracion, msg.confianza);
    } else if (msg.prediccion && msg.confianza >= 70) {
      this.showRawPrediction(msg.prediccion, msg.confianza);
    }
  }

  showRawPrediction(letter, confidence) {
    this.currentPrediction.textContent = letter;
    this.currentPrediction.classList.remove('prediction-pop', 'prediction-glow');
    void this.currentPrediction.offsetWidth;
    this.currentPrediction.classList.add('prediction-pop');
    this.updateConfidence(confidence);
    this.lastUpdate.textContent = new Date().toLocaleTimeString('es-MX', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  }

  onNewDetection(text, confidence) {
    this.currentPrediction.textContent = text;
    this.currentPrediction.classList.remove('prediction-pop', 'prediction-glow');
    void this.currentPrediction.offsetWidth;
    this.currentPrediction.classList.add('prediction-pop', 'prediction-glow');
    this.updateConfidence(confidence);

    this.lastUpdate.textContent = new Date().toLocaleTimeString('es-MX', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    if (this.currentText === '') this.translatedText.innerHTML = '';

    const esLetra = text.length === 1;
    if (esLetra) {
      this.currentText += text;
    } else {
      this.currentText += (this.currentText ? ' ' : '') + text;
    }
    this.translatedText.textContent = this.currentText;

    this.lettersCount++;
    this.statLetters.textContent = this.lettersCount;
    this.detections++;
    this.statDetections.textContent = this.detections;
    this.addToHistory(text);
  }

  updateConfidence(conf) {
    const c = Math.round(conf || 0);
    this.confidenceBadge.textContent = `${c}%`;
    this.confidenceBadge.className = 'text-xs font-mono px-2 py-0.5 rounded-md ' +
      (c >= 85 ? 'bg-emerald-500/20 text-emerald-400' :
       c >= 65 ? 'bg-amber-500/20 text-amber-400' :
       'bg-slate-800 text-slate-400');
  }

  addToHistory(item) {
    this.history.unshift(item);
    if (this.history.length > 20) this.history.pop();
    this.renderHistory();
  }

  renderHistory() {
    if (!this.history.length) {
      this.historyList.innerHTML = '<span class="text-sm text-slate-600 italic">Aún no hay detecciones</span>';
      return;
    }
    this.historyList.innerHTML = this.history.map((l, i) =>
      `<span class="history-chip ${i === 0 ? 'new' : ''}">${l}</span>`
    ).join('');
  }

  clearText() {
    this.currentText = '';
    this.lettersCount = 0;
    this.lastAccepted = null;
    this.statLetters.textContent = '0';
    this.translatedText.innerHTML = '<span class="text-slate-600 italic text-base">Las señas detectadas aparecerán aquí...</span>';
    if (this.wsReady) this.ws.send(JSON.stringify({ cmd: 'reset' }));
    this.showToast('Texto limpiado', 'info');
  }

  clearHistory() {
    this.history = [];
    this.renderHistory();
  }

  async copyText() {
    if (!this.currentText) {
      this.showToast('No hay texto para copiar', 'error');
      return;
    }
    try {
      await navigator.clipboard.writeText(this.currentText);
      this.showToast('Texto copiado', 'success');
    } catch {
      this.showToast('No se pudo copiar', 'error');
    }
  }

  updateSessionTime() {
    if (!this.sessionStart) return;
    const s = Math.floor((Date.now() - this.sessionStart) / 1000);
    this.statSession.textContent = `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  }

  loopFps() {
    if (!this.isRunning) return;
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
      this.fpsBadge.textContent = `${this.fps} FPS`;
    }
    requestAnimationFrame(() => this.loopFps());
  }

  updateStatus(text, color = 'gray') {
    this.statusText.textContent = text;
    const map = { green: 'bg-emerald-400', yellow: 'bg-amber-400', red: 'bg-red-400', gray: 'bg-slate-500' };
    this.statusDot.className = `w-2 h-2 rounded-full ${map[color] || map.gray}`;
  }

  showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    toast.innerHTML = `<span class="text-lg">${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.traductor = new TraductorLSM();
  console.log('%c🤟 Traductor LSM Web listo', 'color:#38bdf8;font-weight:bold');
});
