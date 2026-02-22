(function () {
  const STORAGE_API_URL = 'cortex_demo_api_url';
  const STORAGE_USER_ID = 'cortex_demo_user_id';

  function getApiBase() {
    return (typeof window !== 'undefined' && window.localStorage.getItem(STORAGE_API_URL))
      || (window.CORTEX_DEMO_CONFIG && window.CORTEX_DEMO_CONFIG.apiBaseUrl)
      || 'http://localhost:8000';
  }

  function getUserId() {
    return (typeof window !== 'undefined' && window.localStorage.getItem(STORAGE_USER_ID))
      || (window.CORTEX_DEMO_CONFIG && window.CORTEX_DEMO_CONFIG.defaultUserId)
      || '550e8400-e29b-41d4-a716-446655440000';
  }

  function setOutput(id, data, isError) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    el.classList.remove('success', 'error');
    el.classList.add(isError ? 'error' : 'success');
  }

  async function cortexFetch(path, options = {}) {
    const base = getApiBase().replace(/\/$/, '');
    const url = base + path;
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });
    const text = await res.text();
    let body;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
    if (!res.ok) {
      throw new Error(body && body.detail ? (Array.isArray(body.detail) ? body.detail.map(d => d.msg || d).join('; ') : body.detail) : `HTTP ${res.status}: ${text}`);
    }
    return body;
  }

  function bindConfig() {
    const apiUrlEl = document.getElementById('apiUrl');
    const userIdEl = document.getElementById('userId');
    const saveBtn = document.getElementById('saveConfig');
    if (apiUrlEl) apiUrlEl.value = getApiBase();
    if (userIdEl) userIdEl.value = getUserId();
    if (saveBtn) {
      saveBtn.addEventListener('click', () => {
        if (apiUrlEl) localStorage.setItem(STORAGE_API_URL, apiUrlEl.value.trim() || getApiBase());
        if (userIdEl) localStorage.setItem(STORAGE_USER_ID, userIdEl.value.trim() || getUserId());
        setOutput('statusOutput', 'Saved. API URL and User ID updated.', false);
      });
    }
  }

  function bindHealth() {
    const btn = document.getElementById('checkHealth');
    const out = 'statusOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try {
        const data = await cortexFetch('/health');
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindStatus() {
    const btn = document.getElementById('checkStatus');
    const out = 'statusOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try {
        const data = await cortexFetch('/status');
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindAddMemory() {
    const btn = document.getElementById('addMemory');
    const out = 'addOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const summary = document.getElementById('addSummary')?.value?.trim();
      const entitiesStr = document.getElementById('addEntities')?.value?.trim();
      const importance = parseFloat(document.getElementById('addImportance')?.value) || 0.6;
      const type = document.getElementById('addType')?.value || 'episodic';
      const user = getUserId();
      if (!summary) {
        setOutput(out, 'Summary is required.', true);
        return;
      }
      const entities = entitiesStr ? entitiesStr.split(',').map(s => s.trim()).filter(Boolean) : [];
      try {
        const data = await cortexFetch(`/memory/add?user=${encodeURIComponent(user)}`, {
          method: 'POST',
          body: JSON.stringify({ summary, entities, importance, type }),
        });
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindIngest() {
    const btn = document.getElementById('ingest');
    const out = 'ingestOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const content = document.getElementById('ingestContent')?.value?.trim();
      const user = getUserId();
      if (!content) {
        setOutput(out, 'Content is required.', true);
        return;
      }
      try {
        const data = await cortexFetch(`/memory/ingest?user=${encodeURIComponent(user)}`, {
          method: 'POST',
          body: JSON.stringify({ content }),
        });
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindQuery() {
    const btn = document.getElementById('query');
    const out = 'queryOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const q = document.getElementById('queryQ')?.value?.trim();
      const k = parseInt(document.getElementById('queryK')?.value, 10) || 5;
      const user = getUserId();
      if (!q) {
        setOutput(out, 'Query is required.', true);
        return;
      }
      try {
        const data = await cortexFetch(
          `/memory/query?q=${encodeURIComponent(q)}&user=${encodeURIComponent(user)}&k=${k}`,
          { method: 'GET' }
        );
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindTimeline() {
    const btn = document.getElementById('timeline');
    const out = 'timelineOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const user = getUserId();
      try {
        const data = await cortexFetch(`/memory/timeline?user=${encodeURIComponent(user)}`, { method: 'GET' });
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindGraph() {
    const btn = document.getElementById('graph');
    const out = 'graphOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const node = document.getElementById('graphNode')?.value?.trim();
      const depth = parseInt(document.getElementById('graphDepth')?.value, 10) || 2;
      if (!node) {
        setOutput(out, 'Entity (node) is required.', true);
        return;
      }
      try {
        const data = await cortexFetch(
          `/memory/graph?node=${encodeURIComponent(node)}&depth=${depth}`,
          { method: 'GET' }
        );
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindMemoryCrud() {
    const patchBtn = document.getElementById('patchMemory');
    const deleteBtn = document.getElementById('deleteMemory');
    const out = 'memoryCrudOutput';
    if (patchBtn) {
      patchBtn.addEventListener('click', async () => {
        const id = document.getElementById('memoryId')?.value?.trim();
        const summary = document.getElementById('patchSummary')?.value?.trim();
        const importanceRaw = document.getElementById('patchImportance')?.value?.trim();
        const importance = importanceRaw === '' ? undefined : parseFloat(importanceRaw);
        if (!id) {
          setOutput(out, 'Memory ID is required.', true);
          return;
        }
        if (summary === undefined && importance === undefined) {
          setOutput(out, 'Provide at least one of: summary, importance.', true);
          return;
        }
        const body = {};
        if (summary !== undefined && summary !== '') body.summary = summary;
        if (importance !== undefined && !Number.isNaN(importance)) body.importance = importance;
        if (Object.keys(body).length === 0) {
          setOutput(out, 'Provide at least one of: summary, importance.', true);
          return;
        }
        try {
          const data = await cortexFetch(`/memory/${id}`, { method: 'PATCH', body: JSON.stringify(body) });
          setOutput(out, data, false);
        } catch (e) {
          setOutput(out, e.message, true);
        }
      });
    }
    if (deleteBtn) {
      deleteBtn.addEventListener('click', async () => {
        const id = document.getElementById('memoryId')?.value?.trim();
        if (!id) {
          setOutput(out, 'Memory ID is required.', true);
          return;
        }
        try {
          const data = await cortexFetch(`/memory/${id}`, { method: 'DELETE' });
          setOutput(out, data, false);
        } catch (e) {
          setOutput(out, e.message, true);
        }
      });
    }
  }

  function bindFeedback() {
    const btn = document.getElementById('feedback');
    const out = 'feedbackOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const usedStr = document.getElementById('feedbackUsed')?.value?.trim();
      const retrievedStr = document.getElementById('feedbackRetrieved')?.value?.trim();
      const reward = parseFloat(document.getElementById('feedbackReward')?.value) ?? 0.9;
      const user = getUserId();
      const used_memory_ids = usedStr ? usedStr.split(',').map(s => s.trim()).filter(Boolean) : [];
      const retrieved_memory_ids = retrievedStr ? retrievedStr.split(',').map(s => s.trim()).filter(Boolean) : [];
      if (used_memory_ids.length === 0) {
        setOutput(out, 'At least one used memory ID is required.', true);
        return;
      }
      try {
        const data = await cortexFetch('/memory/feedback', {
          method: 'POST',
          body: JSON.stringify({
            user_id: user,
            used_memory_ids,
            retrieved_memory_ids,
            reward,
          }),
        });
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function bindConsolidate() {
    const btn = document.getElementById('consolidate');
    const out = 'consolidateOutput';
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const user = getUserId();
      try {
        const data = await cortexFetch(`/consolidate/run?user=${encodeURIComponent(user)}`, { method: 'POST' });
        setOutput(out, data, false);
      } catch (e) {
        setOutput(out, e.message, true);
      }
    });
  }

  function init() {
    bindConfig();
    bindHealth();
    bindStatus();
    bindAddMemory();
    bindIngest();
    bindQuery();
    bindTimeline();
    bindGraph();
    bindMemoryCrud();
    bindFeedback();
    bindConsolidate();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
