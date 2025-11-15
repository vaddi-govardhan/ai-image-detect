// Options page script for AI Image Detector

document.addEventListener('DOMContentLoaded', async () => {
  const apiUrlInput = document.getElementById('apiUrl');
  const saveBtn = document.getElementById('saveBtn');
  const resetBtn = document.getElementById('resetBtn');
  const testConnectionBtn = document.getElementById('testConnection');
  const saveStatus = document.getElementById('saveStatus');
  const connectionStatus = document.getElementById('connectionStatus');

  // Load saved settings
  await loadSettings();

  // Event listeners
  saveBtn.addEventListener('click', saveSettings);
  resetBtn.addEventListener('click', resetSettings);
  testConnectionBtn.addEventListener('click', testConnection);

  // Auto-hide status messages
  apiUrlInput.addEventListener('input', () => {
    saveStatus.classList.add('hidden');
  });
});

async function loadSettings() {
  try {
    const { apiUrl } = await chrome.storage.sync.get(['apiUrl']);
    const apiUrlInput = document.getElementById('apiUrl');

    if (apiUrl) {
      apiUrlInput.value = apiUrl;
    } else {
      apiUrlInput.value = 'https://fysiki-ai-image-detector.hf.space';
    }
  } catch (error) {
    console.error('Error loading settings:', error);
  }
}

async function saveSettings() {
  const apiUrlInput = document.getElementById('apiUrl');
  const saveStatus = document.getElementById('saveStatus');
  const apiUrl = apiUrlInput.value.trim();

  // Validate URL
  if (!apiUrl) {
    showStatus(saveStatus, 'error', 'Please enter an API URL');
    return;
  }

  try {
    new URL(apiUrl);
  } catch (error) {
    showStatus(saveStatus, 'error', 'Please enter a valid URL (e.g., https://fysiki-ai-image-detector.hf.space)');
    return;
  }

  try {
    // Save to storage
    await chrome.storage.sync.set({ apiUrl });

    showStatus(saveStatus, 'success', 'Settings saved successfully!');

    // Auto-hide after 3 seconds
    setTimeout(() => {
      saveStatus.classList.add('hidden');
    }, 3000);
  } catch (error) {
    console.error('Error saving settings:', error);
    showStatus(saveStatus, 'error', 'Failed to save settings');
  }
}

async function resetSettings() {
  const apiUrlInput = document.getElementById('apiUrl');
  const saveStatus = document.getElementById('saveStatus');

  apiUrlInput.value = 'https://fysiki-ai-image-detector.hf.space';

  try {
    await chrome.storage.sync.set({ apiUrl: 'https://fysiki-ai-image-detector.hf.space' });
    showStatus(saveStatus, 'success', 'Settings reset to defaults');

    setTimeout(() => {
      saveStatus.classList.add('hidden');
    }, 3000);
  } catch (error) {
    console.error('Error resetting settings:', error);
    showStatus(saveStatus, 'error', 'Failed to reset settings');
  }
}

async function testConnection() {
  const apiUrlInput = document.getElementById('apiUrl');
  const connectionStatus = document.getElementById('connectionStatus');
  const testConnectionBtn = document.getElementById('testConnection');
  const apiUrl = apiUrlInput.value.trim();

  if (!apiUrl) {
    showStatus(connectionStatus, 'error', 'Please enter an API URL');
    return;
  }

  try {
    new URL(apiUrl);
  } catch (error) {
    showStatus(connectionStatus, 'error', 'Invalid URL format');
    return;
  }

  // Show loading
  showStatus(connectionStatus, 'loading', 'Testing connection...');
  testConnectionBtn.disabled = true;

  try {
    const response = await fetch(`${apiUrl}/`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });

    if (response.ok) {
      const data = await response.json();
      showStatus(
        connectionStatus,
        'success',
        `Connected successfully! ${data.message || 'API is running'}`
      );
    } else {
      showStatus(
        connectionStatus,
        'error',
        `Connection failed with status ${response.status}`
      );
    }
  } catch (error) {
    console.error('Connection test error:', error);

    let errorMessage = 'Connection failed. ';
    if (error.name === 'TimeoutError') {
      errorMessage += 'Request timed out.';
    } else if (error.message.includes('Failed to fetch')) {
      errorMessage += 'Make sure the API server is running.';
    } else {
      errorMessage += error.message;
    }

    showStatus(connectionStatus, 'error', errorMessage);
  } finally {
    testConnectionBtn.disabled = false;

    // Auto-hide after 5 seconds
    setTimeout(() => {
      connectionStatus.classList.add('hidden');
    }, 5000);
  }
}

function showStatus(element, type, message) {
  element.classList.remove('hidden', 'success', 'error', 'loading');
  element.classList.add(type);
  element.textContent = message;
}
