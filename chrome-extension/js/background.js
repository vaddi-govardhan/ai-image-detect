// Background service worker for AI Image Detector Chrome Extension

// Create context menu on installation
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeImage",
    title: "Analyze with AI Image Detector",
    contexts: ["image"]
  });

  // Set default API URL
  chrome.storage.sync.get(['apiUrl'], (result) => {
    if (!result.apiUrl) {
      chrome.storage.sync.set({ apiUrl: 'https://fysiki-ai-image-detector.hf.space' });
    }
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeImage") {
    analyzeImage(info.srcUrl, tab.id);
  }
});

// Analyze image function
async function analyzeImage(imageUrl, tabId) {
  try {
    // Show loading notification
    chrome.action.setBadgeText({ text: "..." });
    chrome.action.setBadgeBackgroundColor({ color: "#4688F1" });

    // Get API URL from storage
    const { apiUrl } = await chrome.storage.sync.get(['apiUrl']);
    const baseUrl = apiUrl || 'https://fysiki-ai-image-detector.hf.space';

    // Fetch the image
    const imageResponse = await fetch(imageUrl);
    const imageBlob = await imageResponse.blob();

    // Create form data
    const formData = new FormData();
    formData.append('file', imageBlob, 'image.jpg');

    // Send to API
    const response = await fetch(`${baseUrl}/analyze-image`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API returned status ${response.status}`);
    }

    const result = await response.json();

    // Store result in chrome storage
    await chrome.storage.local.set({
      lastResult: {
        ...result,
        imageUrl: imageUrl,
        timestamp: Date.now(),
        gradcamUrl: result.gradcam_image ? `${baseUrl}${result.gradcam_image}` : null
      }
    });

    // Update badge to show result
    const isAI = result.ai_probability > 0.5;
    chrome.action.setBadgeText({ text: isAI ? "AI" : "Real" });
    chrome.action.setBadgeBackgroundColor({
      color: isAI ? "#F44336" : "#4CAF50"
    });

    // Open popup
    chrome.action.openPopup();

  } catch (error) {
    console.error('Error analyzing image:', error);

    // Store error
    await chrome.storage.local.set({
      lastResult: {
        error: error.message,
        timestamp: Date.now()
      }
    });

    // Show error badge
    chrome.action.setBadgeText({ text: "ERR" });
    chrome.action.setBadgeBackgroundColor({ color: "#F44336" });

    // Open popup to show error
    chrome.action.openPopup();
  }
}

// Listen for messages from popup or content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeImage') {
    analyzeImage(request.imageUrl, sender.tab?.id)
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep message channel open for async response
  }
});
