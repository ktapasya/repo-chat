// Simple chat interface for repo-chat

const chatHistory = document.getElementById('chat-history');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');

// Send question to server
async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    // Disable input while processing
    questionInput.disabled = true;
    sendButton.disabled = true;

    // Add user message to chat
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user';
    const userContentDiv = document.createElement('div');
    userContentDiv.className = 'message-content';
    userContentDiv.textContent = question;
    userMessageDiv.appendChild(userContentDiv);
    chatHistory.appendChild(userMessageDiv);

    // Clear input
    questionInput.value = '';

    try {
        // Use streaming endpoint
        const response = await fetch('/chat-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Create a placeholder message for streaming response
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let sources = [];

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);

                    if (data === '[DONE]') {
                        break;
                    } else if (data.startsWith('[ERROR]')) {
                        fullAnswer += `\n\nError: ${data.slice(7)}`;
                    } else if (data.startsWith('[SOURCES]')) {
                        sources = JSON.parse(data.slice(10));
                    } else {
                        // Unescape escaped newlines (server escapes them to preserve SSE format)
                        const unescaped = data.replace(/\\n/g, '\n');
                        fullAnswer += unescaped;
                        contentDiv.innerHTML = marked.parse(fullAnswer);
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    }
                }
            }
        }

        // Add sources as a separate message bubble
        if (sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message sources';

            const sourcesLabel = document.createElement('div');
            sourcesLabel.className = 'sources-label';
            sourcesLabel.textContent = 'Sources:';
            sourcesDiv.appendChild(sourcesLabel);

            sources.forEach(source => {
                const sourceLink = document.createElement('span');
                sourceLink.className = 'source-link';
                sourceLink.textContent = source;
                sourcesDiv.appendChild(sourceLink);
            });

            chatHistory.appendChild(sourcesDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

    } catch (error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message assistant';
        errorDiv.textContent = `Error: ${error.message}`;
        chatHistory.appendChild(errorDiv);
    } finally {
        // Re-enable input
        questionInput.disabled = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
}

// Event listeners
sendButton.addEventListener('click', askQuestion);

questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

// Focus input on load
questionInput.focus();
