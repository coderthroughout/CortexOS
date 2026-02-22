(function () {
  var STORAGE_API_URL = 'cortex_demo_api_url';
  var STORAGE_USER_ID = 'cortex_demo_user_id';

  function getApiBase() {
    try {
      var s = localStorage.getItem(STORAGE_API_URL);
      if (s) return s.trim();
    } catch (e) {}
    return (window.CORTEX_DEMO_CONFIG && window.CORTEX_DEMO_CONFIG.apiBaseUrl) || 'http://3.87.235.87:8000';
  }

  function getUserId() {
    try {
      var s = localStorage.getItem(STORAGE_USER_ID);
      if (s) return s.trim();
    } catch (e) {}
    return (window.CORTEX_DEMO_CONFIG && window.CORTEX_DEMO_CONFIG.defaultUserId) || '550e8400-e29b-41d4-a716-446655440000';
  }

  function setStatus(text, isOk) {
    var el = document.getElementById('connectionStatus');
    if (!el) return;
    el.textContent = text;
    el.className = 'status' + (isOk === true ? ' ok' : isOk === false ? ' err' : '');
  }

  function cortexFetch(path, options) {
    var base = getApiBase().replace(/\/$/, '');
    var url = base + path;
    return fetch(url, {
      method: options && options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(options && options.headers || {})
      },
      body: options && options.body
    }).then(function (res) {
      return res.text().then(function (text) {
        var body;
        try { body = text ? JSON.parse(text) : null; } catch (e) { body = text; }
        if (!res.ok) {
          var msg = body && body.detail;
          if (Array.isArray(msg)) msg = msg.map(function (d) { return d.msg || d; }).join('; ');
          else if (typeof msg !== 'string' && msg) msg = msg.message || text || res.status;
          throw new Error(msg || 'HTTP ' + res.status);
        }
        return body;
      });
    });
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function addMessage(role, text, options) {
    var list = document.getElementById('messageList');
    if (!list) return null;

    var div = document.createElement('div');
    div.className = 'msg ' + role + (options && options.loading ? ' msg-loading' : '') + (options && options.error ? ' error' : '');
    var bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    div.appendChild(bubble);

    if (options && options.memories && options.memories.length > 0) {
      var card = document.createElement('div');
      card.className = 'memories';
      card.innerHTML = '<div class="memories-title">Retrieved by CortexOS</div>';
      options.memories.forEach(function (m) {
        var row = document.createElement('div');
        row.className = 'memory-item';
        var score = m.score != null ? Number(m.score).toFixed(2) : '—';
        row.innerHTML = '<span>' + escapeHtml(m.summary || m.id) + '</span><span class="memory-score">' + escapeHtml(score) + '</span>';
        card.appendChild(row);
      });
      div.appendChild(card);
    }

    if (options && options.memoryIds && options.memoryIds.length > 0) {
      var fb = document.createElement('div');
      fb.className = 'feedback-wrap';
      fb.innerHTML = 'Helpful? ';
      var yesBtn = document.createElement('button');
      yesBtn.textContent = 'Yes';
      yesBtn.type = 'button';
      var noBtn = document.createElement('button');
      noBtn.textContent = 'No';
      noBtn.type = 'button';
      function sendFeedback(reward) {
        cortexFetch('/memory/feedback', {
          method: 'POST',
          body: JSON.stringify({
            user_id: getUserId(),
            used_memory_ids: options.memoryIds,
            retrieved_memory_ids: options.memoryIds,
            reward: reward
          })
        }).then(function () {
          yesBtn.disabled = true;
          noBtn.disabled = true;
        }).catch(function () {});
      }
      yesBtn.onclick = function () { sendFeedback(0.9); };
      noBtn.onclick = function () { sendFeedback(0.1); };
      fb.appendChild(yesBtn);
      fb.appendChild(noBtn);
      div.appendChild(fb);
    }

    list.appendChild(div);
    list.scrollTop = list.scrollHeight;
    return div;
  }

  function buildReply(memories, userText) {
    if (!memories || memories.length === 0) {
      return "I don't have any relevant memories yet. Tell me more and I'll remember it.";
    }
    var parts = memories.slice(0, 3).map(function (m) { return m.summary || ''; });
    var refs = parts.map(function (p, i) { return '[' + (i + 1) + '] ' + p; }).join(' ');
    return 'Based on what I remember: ' + refs + '. What would you like to explore next?';
  }

  function sendMessage(text) {
    var user = getUserId();
    addMessage('user', text);
    var loadingEl = addMessage('assistant', 'Storing and searching memories…', { loading: true });

    var ingestFailed = false;
    cortexFetch('/memory/ingest?user=' + encodeURIComponent(user), {
      method: 'POST',
      body: JSON.stringify({ content: text })
    }).catch(function () { ingestFailed = true; })
      .then(function () {
        return cortexFetch('/memory/query?q=' + encodeURIComponent(text.slice(0, 200)) + '&user=' + encodeURIComponent(user) + '&k=5');
      })
      .then(function (memories) {
        if (loadingEl && loadingEl.parentNode) loadingEl.remove();
        var reply = buildReply(memories, text);
        if (ingestFailed && (!memories || memories.length === 0)) {
          reply = "Ingest didn't run (e.g. no OPENAI_API_KEY on server). " + reply;
        } else if (ingestFailed) {
          reply = "(New message wasn't extracted; showing existing memories.) " + reply;
        }
        var ids = (memories || []).map(function (m) { return m.id; }).filter(Boolean);
        addMessage('assistant', reply, { memories: memories || [], memoryIds: ids });
      })
      .catch(function (e) {
        if (loadingEl && loadingEl.parentNode) loadingEl.remove();
        addMessage('assistant', 'Error: ' + (e.message || e), { error: true });
      });
  }

  // Config
  document.getElementById('apiUrl').value = getApiBase();
  document.getElementById('userId').value = getUserId();

  document.getElementById('saveConfig').onclick = function () {
    var api = document.getElementById('apiUrl');
    var uid = document.getElementById('userId');
    if (api) try { localStorage.setItem(STORAGE_API_URL, (api.value || getApiBase()).trim()); } catch (e) {}
    if (uid) try { localStorage.setItem(STORAGE_USER_ID, (uid.value || getUserId()).trim()); } catch (e) {}
    setStatus('Saved', true);
    setTimeout(function () { setStatus(''); }, 2000);
  };

  document.getElementById('checkConnection').onclick = function () {
    setStatus('Checking…', null);
    cortexFetch('/health')
      .then(function () { setStatus('Connected', true); })
      .catch(function (e) { setStatus('Failed: ' + (e.message || e), false); });
  };

  // Send
  document.getElementById('sendBtn').onclick = function () {
    var input = document.getElementById('chatInput');
    var text = (input && input.value && input.value.trim()) || '';
    if (!text) return;
    if (input) input.value = '';
    sendMessage(text);
  };

  document.getElementById('chatInput').onkeydown = function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      document.getElementById('sendBtn').click();
    }
  };

  // Timeline
  document.getElementById('btnTimeline').onclick = function () {
    var el = document.getElementById('timelineResult');
    el.textContent = 'Loading…';
    el.className = 'tools-result';
    el.style.display = 'block';
    cortexFetch('/memory/timeline?user=' + encodeURIComponent(getUserId()))
      .then(function (data) {
        if (!Array.isArray(data) || data.length === 0) {
          el.textContent = 'No timeline data. Send messages first.';
          return;
        }
        var html = '';
        data.forEach(function (block) {
          html += '<div class="timeline-period">';
          html += '<div class="timeline-period-label">' + escapeHtml(block.period || '') + '</div>';
          (block.events || []).forEach(function (ev) {
            html += '<div class="timeline-event">' + escapeHtml(ev.summary || ev.id || '') + '</div>';
          });
          html += '</div>';
        });
        el.innerHTML = html;
        el.className = 'tools-result';
      })
      .catch(function (e) {
        el.textContent = 'Error: ' + (e.message || e);
        el.className = 'tools-result err';
      });
  };

  // Graph
  document.getElementById('btnGraph').onclick = function () {
    var entityInput = document.getElementById('graphEntity');
    var node = (entityInput && entityInput.value && entityInput.value.trim()) || '';
    var el = document.getElementById('graphResult');
    if (!node) {
      el.textContent = 'Enter an entity name first.';
      el.className = 'tools-result err';
      el.style.display = 'block';
      return;
    }
    el.textContent = 'Loading…';
    el.className = 'tools-result';
    el.style.display = 'block';
    cortexFetch('/memory/graph?node=' + encodeURIComponent(node) + '&depth=2')
      .then(function (data) {
        var list = Array.isArray(data) ? data : [];
        if (list.length === 0) {
          el.textContent = 'No connected memories for "' + escapeHtml(node) + '".';
        } else {
          el.textContent = 'Entity: ' + node + '\n\nConnected memories:\n' + list.map(function (item) {
            return (item && item.memory_id) ? item.memory_id : item;
          }).join('\n');
        }
        el.className = 'tools-result';
      })
      .catch(function (e) {
        el.textContent = 'Error: ' + (e.message || e);
        el.className = 'tools-result err';
      });
  };
})();
