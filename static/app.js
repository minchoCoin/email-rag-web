const app = document.getElementById('app');
const emailList = document.getElementById('emailList');
const mailContent = document.getElementById('mailContent');
const detailTitle = document.getElementById('detailTitle');
const chat = document.getElementById('chat');
const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('search');
const resultCount = document.getElementById('resultCount');
const indexTarget = document.getElementById('indexTarget');
const chunkTokens = document.getElementById('chunkTokens');
const topK = document.getElementById('topK');
const indexStatus = document.getElementById('indexStatus');
const dbHint = document.getElementById('dbHint');
const planSummary = document.getElementById('planSummary');
const toggleMail = document.getElementById('toggleMail');
const themeToggle = document.getElementById('themeToggle');
const searchButton = searchForm.querySelector('button[type="submit"]');

let activeEmailId = null;
let emails = [];
let indexes = new Map();
let searchInProgress = false;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"]/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
  }[ch]));
}

function selectedIndexTarget() {
  return indexTarget.value;
}

function normalizedIndexTarget() {
  return selectedIndexTarget() === 'all' ? 'email' : selectedIndexTarget();
}

function selectedChunkTokens() {
  return Number(chunkTokens.value);
}

function selectedTopK() {
  const value = Number(topK.value);
  if (!Number.isFinite(value)) return 8;
  return Math.max(1, Math.min(50, Math.trunc(value)));
}

function setResultCount(count, label = ' items') {
  resultCount.textContent = `${count}${label}`;
}

function formatPlan(plan) {
  if (!plan) return '';
  const parts = [];
  if (plan.keywords?.length) parts.push(`Keywords: ${plan.keywords.join(', ')}`);
  if (plan.date_from || plan.date_to) parts.push(`Date: ${plan.date_from || '...'} ~ ${plan.date_to || '...'}`);
  if (plan.sender) parts.push(`Sender: ${plan.sender}`);
  return parts.join(' | ');
}

function indexKey(target, chunkSize) {
  return target === 'chunk' ? `chunk:${chunkSize}` : target;
}

function selectedIndexKey() {
  return indexKey(normalizedIndexTarget(), selectedChunkTokens());
}

function updateIndexStatus() {
  const target = selectedIndexTarget();
  const normalized = normalizedIndexTarget();
  const size = selectedChunkTokens();
  const index = indexes.get(selectedIndexKey());
  const label = target === 'all' ? 'All (subject + body)' : target;
  dbHint.textContent = normalized === 'chunk'
    ? `bge-m3 / ${label} / ${size} token chunks`
    : `bge-m3 / ${label}`;
  if (!index) {
    indexStatus.textContent = '';
    return;
  }
  indexStatus.textContent = index.exists
    ? `${index.db}: ${index.emails} emails · ${index.chunks} vectors`
    : `${index.db} missing`;
}

function renderEmails(items) {
  emails = items;
  setResultCount(items.length);

  if (!items.length) {
    emailList.innerHTML = '<div class="empty">No emails found.</div>';
    return;
  }

  emailList.innerHTML = items.map(item => `
    <button class="email-item ${item.id === activeEmailId ? 'active' : ''}" data-id="${item.id}" type="button">
      <div class="subject">${escapeHtml(item.subject)}</div>
      <div class="meta">${escapeHtml(item.from || item.from_addr)} · ${escapeHtml(item.date)}</div>
      ${item.score !== undefined ? `<div class="score">score ${Number(item.score).toFixed(3)}</div>` : ''}
    </button>
  `).join('');
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${text}`);
  }
  return response.json();
}

async function loadIndexInfo() {
  const data = await requestJson('/api/indexes');
  indexes = new Map((data.indexes || []).map(item => [indexKey(item.index_target, item.chunk_tokens), item]));
  if (data.default_index_target && indexTarget.querySelector(`option[value="${data.default_index_target}"]`)) {
    indexTarget.value = String(data.default_index_target);
  }
  if (data.default_chunk_tokens && chunkTokens.querySelector(`option[value="${data.default_chunk_tokens}"]`)) {
    chunkTokens.value = String(data.default_chunk_tokens);
  }
  updateIndexStatus();
}

async function loadEmails() {
  emailList.innerHTML = '<div class="empty">Loading emails...</div>';
  planSummary.textContent = '';
  const data = await requestJson(`/api/emails?limit=200&chunk_tokens=${selectedChunkTokens()}&index_target=${selectedIndexTarget()}`);
  activeEmailId = null;
  renderEmails(data.emails || []);
}

async function openEmail(id) {
  activeEmailId = id;
  renderEmails(emails);
  app.classList.add('show-mail');
  updateMailToggle();
  const item = await requestJson(`/api/emails/${id}?chunk_tokens=${selectedChunkTokens()}&index_target=${selectedIndexTarget()}`);
  detailTitle.textContent = item.subject || 'Email content';
  mailContent.innerHTML = `
    <div class="mail-head">
      <div class="mail-subject">${escapeHtml(item.subject)}</div>
      <div class="kv">
        <div>From</div><div>${escapeHtml(item.from_addr)}</div>
        <div>To</div><div>${escapeHtml(item.to_addr)}</div>
        <div>Date</div><div>${escapeHtml(item.date)}</div>
      </div>
    </div>
    <div class="body">${escapeHtml(item.body)}</div>
  `;
}

function addMessage(text, role) {
  const node = document.createElement('div');
  node.className = `msg ${role}`;
  node.textContent = text;
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
  return node;
}

function formatElapsed(startedAt) {
  return `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
}

function createRunLog() {
  const startedAt = performance.now();
  const node = document.createElement('div');
  node.className = 'msg ai log-card';
  node.innerHTML = `
    <div class="log-head">
      <span>Search log</span>
      <span class="timer">0.0s</span>
    </div>
    <div class="log-lines"></div>
  `;
  const timer = node.querySelector('.timer');
  const lines = node.querySelector('.log-lines');
  const intervalId = window.setInterval(() => {
    timer.textContent = formatElapsed(startedAt);
  }, 100);

  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;

  return {
    node,
    add(line) {
      const item = document.createElement('div');
      item.className = 'log-line';
      item.textContent = `[${formatElapsed(startedAt)}] ${line}`;
      lines.appendChild(item);
      chat.scrollTop = chat.scrollHeight;
    },
    stop(finalSeconds) {
      window.clearInterval(intervalId);
      timer.textContent = finalSeconds == null ? formatElapsed(startedAt) : `${Number(finalSeconds).toFixed(1)}s`;
    },
  };
}

emailList.addEventListener('click', event => {
  const button = event.target.closest('.email-item');
  if (button) openEmail(Number(button.dataset.id));
});

function updateMailToggle() {
  const visible = app.classList.contains('show-mail');
  toggleMail.textContent = visible ? 'Hide email' : 'Show email';
  toggleMail.title = visible ? 'Hide email content' : 'Show email content';
  toggleMail.setAttribute('aria-label', toggleMail.title);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('email-rag-theme', theme);
  themeToggle.textContent = theme === 'dark' ? 'Light' : 'Dark';
  themeToggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

function initTheme() {
  const saved = localStorage.getItem('email-rag-theme');
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  setTheme(saved || (prefersDark ? 'dark' : 'light'));
}

toggleMail.addEventListener('click', () => {
  app.classList.toggle('show-mail');
  updateMailToggle();
});

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.dataset.theme || 'light';
  setTheme(current === 'dark' ? 'light' : 'dark');
});

document.getElementById('clearChat').addEventListener('click', () => {
  chat.innerHTML = '';
});

indexTarget.addEventListener('change', () => {
  updateIndexStatus();
  activeEmailId = null;
  planSummary.textContent = '';
  detailTitle.textContent = 'Email content';
  mailContent.innerHTML = '<div class="empty">Select an email from the list.</div>';
  loadEmails().catch(error => {
    emailList.innerHTML = `<div class="empty">Error: ${escapeHtml(error.message)}</div>`;
  });
});

chunkTokens.addEventListener('change', () => {
  updateIndexStatus();
  activeEmailId = null;
  planSummary.textContent = '';
  detailTitle.textContent = 'Email content';
  mailContent.innerHTML = '<div class="empty">Select an email from the list.</div>';
  loadEmails().catch(error => {
    emailList.innerHTML = `<div class="empty">Error: ${escapeHtml(error.message)}</div>`;
  });
});

topK.addEventListener('change', () => {
  topK.value = String(selectedTopK());
  loadEmails().catch(error => {
    emailList.innerHTML = `<div class="empty">Error: ${escapeHtml(error.message)}</div>`;
  });
});

document.getElementById('loadRecent').addEventListener('click', () => {
  loadEmails().catch(error => {
    emailList.innerHTML = `<div class="empty">Error: ${escapeHtml(error.message)}</div>`;
  });
});

async function runQuestion(text) {
  if (searchInProgress) return;
  searchInProgress = true;
  chat.innerHTML = '';
  addMessage(text, 'user');
  const log = createRunLog();
  log.add('Started query.');
  log.add(`Search unit: ${selectedIndexTarget()} (${normalizedIndexTarget()}); top_k: ${selectedTopK()}; chunk tokens: ${selectedChunkTokens()}.`);
  log.add('Calling /api/chat for filter extraction, database search, and answer generation.');
  if (searchButton) searchButton.disabled = true;
  try {
    const data = await requestJson('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        limit: selectedTopK(),
        chunk_tokens: selectedChunkTokens(),
        index_target: selectedIndexTarget(),
      }),
    });
    const planText = formatPlan(data.plan);
    planSummary.textContent = planText;
    if (planText) log.add(`Search filters: ${planText}.`);
    log.add(`Retrieved ${data.results?.length || 0} emails.`);
    if (data.elapsed_seconds != null) log.add(`Backend elapsed time: ${Number(data.elapsed_seconds).toFixed(1)}s.`);
    log.add('Completed answer generation.');
    log.stop();
    addMessage(data.answer || '', 'ai');
    renderEmails(data.results || []);
  } catch (error) {
    log.add(`Error: ${error.message}`);
    log.stop();
    addMessage(`Error: ${error.message}`, 'ai');
  } finally {
    searchInProgress = false;
    if (searchButton) searchButton.disabled = false;
  }
}

searchForm.addEventListener('submit', event => {
  event.preventDefault();
  const text = searchInput.value.trim();
  if (!text || searchInProgress) return;
  searchInput.value = '';
  runQuestion(text);
});

initTheme();
updateMailToggle();

loadIndexInfo()
  .then(loadEmails)
  .catch(error => {
    emailList.innerHTML = `<div class="empty">Error: ${escapeHtml(error.message)}</div>`;
  });
