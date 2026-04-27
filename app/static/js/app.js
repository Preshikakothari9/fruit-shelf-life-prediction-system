/* ═══════════════════════════════════════════════════════════════
   FruitFresh Antigravity — Dashboard Logic
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const previewStrip= document.getElementById('preview-strip');
  const previewImg  = document.getElementById('preview-img');
  const fileName    = document.getElementById('file-name');
  const btnAnalyze  = document.getElementById('btn-analyze');
  const analyzing   = document.getElementById('analyzing');
  const resultCard  = document.getElementById('result-card');

  if (!dropZone) return; // not on dashboard

  let selectedFile = null;

  // ── Drag & Drop ─────────────────────────────────────────────
  ['dragenter', 'dragover'].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
  );
  ['dragleave', 'drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
  );
  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  });

  // ── File input change ───────────────────────────────────────
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    fileName.textContent = file.name;
    previewStrip.style.display = 'flex';
    resultCard.classList.remove('visible');
  }

  // ── Analyze button ──────────────────────────────────────────
  btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzing.classList.add('active');
    resultCard.classList.remove('visible');
    btnAnalyze.disabled = true;

    const form = new FormData();
    form.append('image', selectedFile);

    try {
      const res = await fetch('/predict', { method: 'POST', body: form });
      const data = await res.json();

      analyzing.classList.remove('active');
      btnAnalyze.disabled = false;

      if (data.success) {
        showResult(data.prediction);
        prependHistory(data.prediction);
      } else {
        showToast(data.error || 'Prediction failed', 'error');
      }
    } catch (err) {
      analyzing.classList.remove('active');
      btnAnalyze.disabled = false;
      showToast('Network error — is the server running?', 'error');
    }
  });

  // ── Render result card ──────────────────────────────────────
  function showResult(p) {
    document.getElementById('res-emoji').textContent  = p.emoji || '🍎';
    document.getElementById('res-fruit').textContent   = p.fruit_type;
    document.getElementById('res-status').textContent  = p.freshness_status;
    document.getElementById('res-status').className    = 'status-badge status-' + p.freshness_status.replace(/\s+/g, '-');
    document.getElementById('res-days').textContent    = p.predicted_days;
    const resTemp = document.getElementById('res-temp');
    if (resTemp) resTemp.textContent = p.temperature || '—';
    document.getElementById('res-confidence').textContent = p.confidence + '%';
    document.getElementById('res-tip').textContent     = p.storage_tip;

    resultCard.classList.add('visible');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // ── Prepend to history grid (no page reload) ────────────────
  function prependHistory(p) {
    const grid  = document.getElementById('history-grid');
    const empty = document.getElementById('empty-state');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'history-item glass';
    item.id = `hist-${p.id}`;
    item.innerHTML = `
      <div class="hi-top">
        <img class="hi-thumb" src="${p.image_url}" alt="${p.fruit_type}">
        <div class="hi-info">
          <h4>${p.emoji || ''} ${p.fruit_type}</h4>
          <span>Just now</span>
        </div>
        <button class="btn-delete" onclick="deleteRecord(${p.id})" title="Delete">🗑️</button>
      </div>
      <div class="hi-stats">
        <span class="hi-chip">📅 ${p.predicted_days} days</span>
        <span class="hi-chip">🎯 ${p.confidence}%</span>
        <span class="hi-chip status-badge status-${p.freshness_status.replace(/\s+/g, '-')}">${p.freshness_status}</span>
      </div>`;
    grid.prepend(item);
  }

  // ── Toast notification ──────────────────────────────────────
  function showToast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `flash flash-${type}`;
    el.textContent = msg;
    el.style.cssText = 'position:fixed;top:1rem;right:1rem;z-index:9999;min-width:260px;';
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 400); }, 3500);
  }
});

// ── Global delete helper ──────────────────────────────────────
async function deleteRecord(id) {
  if (!confirm('Delete this prediction?')) return;
  try {
    const res = await fetch(`/history/${id}`, { method: 'DELETE' });
    if (res.ok) {
      const el = document.getElementById(`hist-${id}`);
      if (el) { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }
    }
  } catch { /* silent */ }
}
