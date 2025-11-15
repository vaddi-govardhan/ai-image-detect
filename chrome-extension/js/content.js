// Content script for AI Image Detector
// This script runs on all web pages and helps with image analysis

// Listen for messages from the background script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPageImages') {
    const images = Array.from(document.querySelectorAll('img'))
      .map(img => ({
        src: img.src,
        alt: img.alt,
        width: img.width,
        height: img.height
      }))
      .filter(img => img.src && img.width > 50 && img.height > 50);

    sendResponse({ images });
  }

  return true;
});

// Optional: Add visual feedback when analyzing an image
function showAnalyzingIndicator(element) {
  const indicator = document.createElement('div');
  indicator.className = 'ai-detector-analyzing';
  indicator.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(102, 126, 234, 0.8);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: sans-serif;
    font-size: 14px;
    z-index: 10000;
  `;
  indicator.textContent = 'Analyzing...';

  element.style.position = 'relative';
  element.appendChild(indicator);

  setTimeout(() => {
    indicator.remove();
  }, 2000);
}
