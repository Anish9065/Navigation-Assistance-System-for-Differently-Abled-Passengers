/* app.js — global utilities for Navigation Assistance System */

// ── Toast notifications ───────────────────────────────────────
function showToast(msg, type = 'info', duration = 3000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText =
      'position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
    document.body.appendChild(container);
  }

  const colors = { info: '#1a56db', success: '#05b47e', error: '#e53935', warn: '#ff8c00' };
  const icons  = { info: 'ti-info-circle', success: 'ti-circle-check', error: 'ti-alert-circle', warn: 'ti-alert-triangle' };

  const toast = document.createElement('div');
  toast.style.cssText =
    `background:#1a1d27;border:1px solid ${colors[type]}55;color:#e8eaf0;` +
    `padding:10px 14px;border-radius:10px;font-size:13px;display:flex;` +
    `align-items:center;gap:8px;box-shadow:0 4px 20px rgba(0,0,0,0.4);` +
    `min-width:220px;border-left:3px solid ${colors[type]};` +
    `animation:slideIn 0.2s ease;`;
  toast.innerHTML = `<i class="ti ${icons[type]}" style="color:${colors[type]}"></i><span>${msg}</span>`;
  container.appendChild(toast);

  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300); }, duration);
}

// ── Drag & drop on drop zones ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.drop-zone').forEach(zone => {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = '#1a56db'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.style.borderColor = '';
      const files = e.dataTransfer.files;
      // Find associated file input
      const inp = zone.parentElement.querySelector('input[type=file]');
      if (inp && files.length) {
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        inp.files = dt.files;
        inp.dispatchEvent(new Event('change'));
      }
    });
  });
});
