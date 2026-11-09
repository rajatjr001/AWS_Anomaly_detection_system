(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NUMERIC_FIELDS = ['temperature', 'humidity', 'pressure', 'rainfall', 'wind_speed', 'wind_direction'];
  const REFRESH_MS = 5000;
  const MOTION_TARGETS = '.section > .label, .section > h2, .section > .sub, .section > .card, .grid > .card, .grid3 > .card, .two > .card, .band, pre';
  let chart = null;

  const fmt = v => (v == null ? '—' : Number(v).toFixed(1));
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const statusClass = s => (s === 'Normal/Good' ? 'good' : s === 'Suspicious' ? 'suspicious' : 'bad');

  // ---------- Motion ----------
  function setupMotion() {
    const nodes = [...document.querySelectorAll(MOTION_TARGETS)];
    nodes.forEach(n => {
      const i = [...n.parentElement.children].indexOf(n);
      n.style.setProperty('--d', Math.min(i, 6) * 90 + 'ms');
    });
    if (reduceMotion) return;

    document.querySelectorAll('.card').forEach(card => {
      card.addEventListener('pointermove', e => {
        const r = card.getBoundingClientRect();
        card.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        card.style.setProperty('--my', (e.clientY - r.top) + 'px');
      });
    });

    if (!('IntersectionObserver' in window)) {
      nodes.forEach(n => n.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    nodes.forEach(n => {
      n.classList.add('rv');
      io.observe(n);
    });
  }

  // Animates a number to its new value; the final text matches the API value exactly.
  function countTo(el, value, suffix = '') {
    const end = value == null ? NaN : Number(value);
    if (reduceMotion || !Number.isFinite(end)) {
      cancelAnimationFrame(el._raf);
      el.textContent = value + suffix;
      el.dataset.n = Number.isFinite(end) ? end : '';
      return;
    }
    if (el.dataset.n !== undefined && Number(el.dataset.n) === end) {
      el.textContent = value + suffix;
      return;
    }
    cancelAnimationFrame(el._raf);
    const start = Number(el.dataset.n) || 0;
    const isInt = Number.isInteger(end);
    const t0 = performance.now();
    el.dataset.n = end;
    const step = now => {
      const p = Math.min((now - t0) / 800, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const cur = start + (end - start) * eased;
      el.textContent = p < 1 ? (isInt ? Math.round(cur) : cur.toFixed(1)) + suffix : value + suffix;
      if (p < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  // Restarts the CSS flash animation each time the message changes.
  function setMessage(text) {
    const m = $('message');
    m.textContent = text;
    m.classList.remove('flash');
    void m.offsetWidth;
    m.classList.add('flash');
  }

  // ---------- Dashboard ----------
  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} returned ${res.status}`);
    return res.json();
  }

  function renderSummary(s) {
    countTo($('total'), s.total_readings);
    countTo($('anomalies'), s.anomalies);
    countTo($('suspicious'), s.suspicious);
    countTo($('rate'), s.anomaly_percentage, '%');
  }

  function renderRows(readings) {
    $('rows').innerHTML = readings.slice().reverse().map(x => `
      <tr>
        <td>${esc((x.timestamp || '').replace('T', ' ').slice(0, 19))}</td>
        <td>${esc(x.station_id)}</td>
        <td>${esc(x.source)}</td>
        <td>${fmt(x.temperature)}</td>
        <td>${fmt(x.humidity)}</td>
        <td>${fmt(x.pressure)}</td>
        <td>${fmt(x.wind_speed)}</td>
        <td class="${statusClass(x.status)}">${esc(x.status)}</td>
      </tr>`).join('')
      || '<tr><td colspan="8">No readings stored yet. Start the collector or submit a manual reading.</td></tr>';
  }

  function renderChart(readings) {
    const labels = readings.map(x => (x.timestamp || '').slice(11, 19));
    const values = readings.map(x => x.temperature);
    if (!chart) {
      chart = new Chart($('trend'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Temperature °C',
            data: values,
            borderColor: '#ffffff',
            backgroundColor: 'rgba(255,255,255,.08)',
            pointBackgroundColor: '#000',
            pointBorderColor: '#fff',
            fill: true,
            tension: .3,
            pointRadius: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 900, easing: 'easeOutQuart' },
          plugins: { legend: { labels: { color: '#cfcfcf' } } },
          scales: {
            x: { ticks: { color: '#8a8a8a', maxTicksLimit: 8 }, grid: { color: '#1c1c1c' } },
            y: { ticks: { color: '#8a8a8a' }, grid: { color: '#1c1c1c' } }
          }
        }
      });
    } else {
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.update();
    }
  }

  async function refreshDashboard() {
    try {
      const [summary, latest, health] = await Promise.all([
        getJSON('/api/summary'),
        getJSON('/api/readings/latest?limit=100'),
        getJSON('/api/health')
      ]);
      $('health').textContent = health.trained ? 'MODEL READY · DATABASE CONNECTED' : 'MODEL NOT TRAINED';
      renderSummary(summary);
      const readings = latest.readings || [];
      renderRows(readings);
      renderChart(readings);
    } catch (e) {
      $('health').textContent = 'API ERROR';
      console.error(e);
    }
  }

  async function submitReading() {
    const btn = $('submit');
    const payload = Object.fromEntries(new FormData($('sensorForm')).entries());
    NUMERIC_FIELDS.forEach(k => (payload[k] = Number(payload[k] || 0)));
    payload.source = 'manual';

    setMessage('Processing…');
    btn.disabled = true;
    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Ingestion failed');
      setMessage(`Saved: ${result.observation.status} · anomaly score ${result.observation.anomaly_score}`);
      await refreshDashboard();
    } catch (e) {
      setMessage('Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  }

  function init() {
    setupMotion();
    if (!$('sensorForm')) return; // only the dashboard page has these elements
    $('submit').addEventListener('click', submitReading);
    $('refresh').addEventListener('click', refreshDashboard);
    refreshDashboard();
    setInterval(refreshDashboard, REFRESH_MS);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
