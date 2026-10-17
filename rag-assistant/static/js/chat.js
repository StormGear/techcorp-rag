// TechCorp AI Assistant - Chat Interface

let isProcessing = false;

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatForm = document.getElementById('chat-form');
const sendButton = document.getElementById('send-button');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const clearButton = document.getElementById('clear-button');

document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    setupEventListeners();
    resizeComposer();
    chatInput.focus();
});

function setupEventListeners() {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatInput.addEventListener('input', resizeComposer);

    chatMessages.addEventListener('click', (e) => {
        const button = e.target.closest('.suggestion');
        if (!button) return;
        chatInput.value = button.dataset.query;
        resizeComposer();
        sendMessage();
    });

    clearButton.addEventListener('click', clearChat);
}

function resizeComposer() {
    chatInput.style.height = 'auto';
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 136)}px`;
}

async function sendMessage() {
    if (isProcessing) return;

    const message = chatInput.value.trim();
    if (!message) return;

    isProcessing = true;
    sendButton.disabled = true;
    chatInput.value = '';
    resizeComposer();
    hideWelcome();

    addMessage('user', message);
    await sendMessageStream(message);

    isProcessing = false;
    sendButton.disabled = false;
    chatInput.focus();
}

async function sendMessageStream(message) {
    const { contentDiv, textEl } = createAssistantShell(true);
    let fullText = '';

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let sawToken = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6);
                if (!jsonStr.trim()) continue;

                try {
                    const data = JSON.parse(jsonStr);

                    if (data.event === 'token') {
                        if (!sawToken) {
                            textEl.classList.remove('loading-dots');
                            textEl.innerHTML = '';
                            sawToken = true;
                        }
                        fullText += data.content;
                        textEl.innerHTML = formatMessage(fullText);
                        scrollToBottom();
                    } else if (data.event === 'sources' && data.sources?.length) {
                        appendSources(contentDiv, data.sources);
                    } else if (data.event === 'error') {
                        textEl.classList.remove('loading-dots');
                        textEl.innerHTML = `Sorry, I encountered an error: ${data.error}`;
                    }
                } catch (e) {
                    console.error('Error parsing SSE data:', e);
                }
            }
        }

        if (!sawToken && !textEl.textContent) {
            textEl.classList.remove('loading-dots');
            textEl.innerHTML = '<p>I did not receive a response. Please try again.</p>';
        }
    } catch (error) {
        textEl.classList.remove('loading-dots');
        textEl.innerHTML = '<p>Sorry, I couldn\'t connect to the server. Please check your connection and try again.</p>';
        console.error('Streaming error:', error);
    }
}

async function sendMessageNormal(message) {
    const typingId = showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await response.json();
        removeTypingIndicator(typingId);

        if (response.ok) {
            addMessage('assistant', data.response, data.sources);
        } else {
            addMessage('assistant', `Sorry, I encountered an error: ${data.error || 'Please try again.'}`);
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('assistant', 'Sorry, I couldn\'t connect to the server. Please check your connection and try again.');
        console.error('Chat error:', error);
    }
}

function createMessageRow(type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = type === 'user' ? 'You' : 'TC';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    return { messageDiv, contentDiv };
}

function createAssistantShell(showTyping) {
    const { contentDiv } = createMessageRow('assistant');
    const textEl = document.createElement(showTyping ? 'div' : 'div');
    if (showTyping) {
        textEl.className = 'loading-dots';
        textEl.innerHTML = '<span></span><span></span><span></span>';
    }
    contentDiv.appendChild(textEl);
    scrollToBottom();
    return { contentDiv, textEl };
}

function addMessage(type, content, sources = null) {
    const { contentDiv } = createMessageRow(type);
    contentDiv.innerHTML = formatMessage(content);
    if (sources?.length) {
        appendSources(contentDiv, sources);
    }
    scrollToBottom();
}

function uniqueSources(sources) {
    const seen = new Set();
    return sources.filter((source) => {
        const title = source.metadata?.title || 'Document';
        if (seen.has(title)) return false;
        seen.add(title);
        return true;
    });
}

function appendSources(contentDiv, sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';
    sourcesDiv.innerHTML = '<small>Sources</small>';

    uniqueSources(sources).forEach((source) => {
        const sourceSpan = document.createElement('span');
        sourceSpan.className = 'source-item';
        sourceSpan.textContent = source.metadata?.title || 'Document';
        sourceSpan.title = `Score: ${(source.score * 100).toFixed(1)}%`;
        sourcesDiv.appendChild(sourceSpan);
    });

    contentDiv.appendChild(sourcesDiv);
}

function formatMessage(content) {
    let formatted = content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^- (.*?)$/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*?)$/gm, '<li>$2</li>');

    if (!formatted.startsWith('<')) {
        formatted = `<p>${formatted}</p>`;
    }

    formatted = formatted.replace(/(<li>.*?<\/li>\s*)+/g, (match) => `<ul>${match}</ul>`);
    return formatted;
}

function showTypingIndicator() {
    const typingId = 'typing-' + Date.now();
    const { messageDiv, contentDiv } = createMessageRow('assistant');
    messageDiv.id = typingId;
    contentDiv.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    scrollToBottom();
    return typingId;
}

function removeTypingIndicator(typingId) {
    document.getElementById(typingId)?.remove();
}

function hideWelcome() {
    document.getElementById('welcome')?.remove();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (response.ok && data.status === 'operational') {
            statusDot.style.backgroundColor = '#10b981';
            statusText.textContent = 'Connected';
        } else {
            statusDot.style.backgroundColor = '#f59e0b';
            statusText.textContent = 'Limited';
        }
    } catch (error) {
        statusDot.style.backgroundColor = '#ef4444';
        statusText.textContent = 'Offline';
    }
}

function clearChat() {
    chatMessages.querySelectorAll('.message').forEach((msg) => msg.remove());
    if (!document.getElementById('welcome')) {
        chatMessages.insertAdjacentHTML('afterbegin', `
            <section class="welcome" id="welcome" aria-label="Getting started">
                <div class="welcome-badge">Internal knowledge assistant</div>
                <h2>How can I help today?</h2>
                <p>Ask about employee policies, CloudSync Pro, DataVault, meeting notes, or customer support.</p>
                <div class="suggestion-grid" id="suggestion-grid">
                    <button type="button" class="suggestion" data-query="What is our remote work policy?">Remote work policy</button>
                    <button type="button" class="suggestion" data-query="What are the key features of CloudSync Pro?">CloudSync Pro features</button>
                    <button type="button" class="suggestion" data-query="Summarize recent meeting decisions.">Meeting decisions</button>
                    <button type="button" class="suggestion" data-query="How do we handle customer support FAQs?">Customer FAQs</button>
                </div>
            </section>
        `);
    }
    chatInput.focus();
}
