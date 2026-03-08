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
    addMessage('user', question);

    // Clear input
    questionInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Add assistant response with sources
        addMessage('assistant', data.answer, data.sources);
    } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
    } finally {
        // Re-enable input
        questionInput.disabled = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
}

// Add a message to the chat history
function addMessage(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    // Parse markdown and set as HTML
    contentDiv.innerHTML = marked.parse(content);
    messageDiv.appendChild(contentDiv);

    // Add sources if present
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';

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

        messageDiv.appendChild(sourcesDiv);
    }

    chatHistory.appendChild(messageDiv);

    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
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
