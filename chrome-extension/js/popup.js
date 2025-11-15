// Popup script for AI Image Detector

document.addEventListener('DOMContentLoaded', async () => {
  // Get elements
  const welcome = document.getElementById('welcome');
  const loading = document.getElementById('loading');
  const error = document.getElementById('error');
  const results = document.getElementById('results');
  const errorMessage = document.getElementById('errorMessage');
  const settingsLink = document.getElementById('settingsLink');
  const optionsBtn = document.getElementById('optionsBtn');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');

  // Load last result
  await loadLastResult();

  // Event listeners
  settingsLink.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.runtime.openOptionsPage();
  });

  optionsBtn.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  newAnalysisBtn.addEventListener('click', () => {
    showWelcome();
  });

  document.getElementById('retryBtn')?.addEventListener('click', () => {
    showWelcome();
  });
});

async function loadLastResult() {
  try {
    const { lastResult } = await chrome.storage.local.get(['lastResult']);

    if (!lastResult) {
      showWelcome();
      return;
    }

    if (lastResult.error) {
      showError(lastResult.error);
      return;
    }

    displayResults(lastResult);
  } catch (err) {
    console.error('Error loading result:', err);
    showWelcome();
  }
}

function showWelcome() {
  hideAll();
  document.getElementById('welcome').classList.remove('hidden');
}

function showLoading() {
  hideAll();
  document.getElementById('loading').classList.remove('hidden');
}

function showError(message) {
  hideAll();
  document.getElementById('error').classList.remove('hidden');
  document.getElementById('errorMessage').textContent = message;
}

function hideAll() {
  document.getElementById('welcome').classList.add('hidden');
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('error').classList.add('hidden');
  document.getElementById('results').classList.add('hidden');
}

function displayResults(result) {
  hideAll();
  document.getElementById('results').classList.remove('hidden');

  // Display original image
  const originalImage = document.getElementById('originalImage');
  if (result.imageUrl) {
    originalImage.src = result.imageUrl;
    originalImage.style.display = 'block';
  } else {
    originalImage.style.display = 'none';
  }

  // Display classification result
  const resultCard = document.querySelector('.result-card');
  const resultLabel = document.getElementById('resultLabel');
  const resultConfidence = document.getElementById('resultConfidence');
  const progressFill = document.getElementById('progressFill');

  const probability = result.ai_probability || 0;
  const isAI = probability > 0.5;
  const confidencePercent = (probability * 100).toFixed(1);

  // Update label and styling
  resultCard.className = 'result-card ' + (isAI ? 'ai' : 'real');
  resultLabel.textContent = isAI ? 'AI Generated' : 'Real Image';
  resultConfidence.textContent = confidencePercent + '%';
  progressFill.style.width = confidencePercent + '%';

  // Display detected objects
  const objectsSection = document.getElementById('objectsSection');
  const objectsList = document.getElementById('objectsList');

  if (result.objects && result.objects.length > 0) {
    objectsSection.classList.remove('hidden');
    objectsList.innerHTML = result.objects
      .map(obj => `<span class="object-tag">${escapeHtml(obj)}</span>`)
      .join('');
  } else {
    objectsSection.classList.add('hidden');
  }

  // Display summary
  const summarySection = document.getElementById('summarySection');
  const summaryText = document.getElementById('summaryText');

  if (result.summary) {
    summarySection.classList.remove('hidden');
    summaryText.textContent = result.summary;
  } else {
    summarySection.classList.add('hidden');
  }

  // Display Grad-CAM
  const gradcamSection = document.getElementById('gradcamSection');
  const gradcamImage = document.getElementById('gradcamImage');

  if (result.gradcamUrl) {
    gradcamSection.classList.remove('hidden');
    gradcamImage.src = result.gradcamUrl;
  } else {
    gradcamSection.classList.add('hidden');
  }
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// Listen for updates from background script
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'local' && changes.lastResult) {
    const newResult = changes.lastResult.newValue;
    if (newResult) {
      if (newResult.error) {
        showError(newResult.error);
      } else {
        displayResults(newResult);
      }
    }
  }
});
