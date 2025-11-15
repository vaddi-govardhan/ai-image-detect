# AI Image Detector - Chrome Extension

A Chrome extension that allows you to detect if images are AI-generated or real using advanced machine learning. Simply right-click on any image and get instant analysis powered by Vision Transformer (ViT) models.

## Features

- **Right-click Analysis**: Analyze any image on the web with a simple right-click
- **AI Detection**: Uses state-of-the-art Vision Transformer (ViT) models to classify images
- **Object Detection**: Identifies objects in images using YOLOv8
- **Attention Heatmap**: Visualize which parts of the image the AI focused on with Grad-CAM
- **Detailed Explanations**: Get AI-generated explanations powered by Google Gemini
- **Beautiful UI**: Modern, gradient-based interface with smooth animations
- **Configurable**: Set custom API endpoints for self-hosted servers

## Screenshots

The extension provides:
- A welcome screen with usage instructions
- Real-time analysis with loading indicators
- Detailed results with confidence scores
- Object detection tags
- AI-generated explanations
- Grad-CAM visualization heatmaps

## Installation

### Prerequisites

The extension works out of the box with the hosted API at HuggingFace Spaces. No setup required!

- **Default API**: `https://fysiki-ai-image-detector.hf.space` (hosted on HuggingFace)
- **Local Development**: You can change the API URL in settings to use `http://localhost:7860`

### Installing the Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top-right corner
3. Click "Load unpacked"
4. Select the `chrome-extension` directory from this project
5. The extension should now appear in your extensions list

## Usage

### Basic Usage

1. **Navigate to any webpage** with images
2. **Right-click on an image** you want to analyze
3. **Select "Analyze with AI Image Detector"** from the context menu
4. **Wait for the analysis** (usually 2-5 seconds)
5. **View the results** by clicking the extension icon or waiting for the popup

### Understanding Results

- **Classification**: The extension shows if the image is "AI Generated" or "Real Image"
- **Confidence Score**: A percentage showing how confident the model is (0-100%)
- **Progress Bar**: Visual representation of the confidence level
  - Red gradient: AI Generated
  - Blue gradient: Real Image
- **Detected Objects**: List of objects found in the image
- **Analysis Summary**: AI-generated explanation of the classification
- **Attention Heatmap**: Visual representation of which areas the AI focused on

### Badge Indicators

The extension icon shows a badge with the result:
- **"AI"** with red background: Image detected as AI-generated
- **"Real"** with green background: Image detected as real
- **"..."** with blue background: Analysis in progress
- **"ERR"** with red background: Error occurred

## Configuration

### Changing the API URL

1. Click the extension icon
2. Click the ⚙️ Settings link in the footer (or the Settings button)
3. Enter your API server URL (e.g., `https://fysiki-ai-image-detector.hf.space` or `http://localhost:7860`)
4. Click "Test Connection" to verify the server is reachable
5. Click "Save Settings"

### Default Settings

- **API URL**: `https://fysiki-ai-image-detector.hf.space` (hosted on HuggingFace Spaces)
- The extension automatically retries failed connections
- Results are cached locally for quick access

## API Server Setup (Optional)

The extension works with the hosted API by default. However, if you want to run your own API server locally:

```bash
# Using Docker (recommended)
docker-compose up

# Or run locally
cd backend
python main.py
```

The API will be accessible at `http://localhost:7860`. Then change the API URL in extension settings.

## Project Structure

```
chrome-extension/
├── manifest.json           # Extension configuration
├── popup.html             # Main popup interface
├── options.html           # Settings page
├── icons/                 # Extension icons
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
├── css/                   # Stylesheets
│   ├── popup.css
│   └── options.css
└── js/                    # JavaScript files
    ├── background.js      # Service worker (context menu, API calls)
    ├── popup.js          # Popup UI logic
    ├── options.js        # Settings page logic
    └── content.js        # Content script (page interaction)
```

## Technical Details

### Permissions

The extension requires:
- `contextMenus`: To add right-click menu options
- `storage`: To save settings and results
- `activeTab`: To access images on the current page
- Host permissions for API communication

### API Endpoints Used

- `POST /analyze-image`: Main analysis endpoint
  - Accepts: `multipart/form-data` with image file
  - Returns: AI probability, objects, summary, Grad-CAM URL

### Browser Compatibility

- Chrome 88+
- Microsoft Edge 88+
- Any Chromium-based browser supporting Manifest V3

## Troubleshooting

### Extension Shows "ERR"

1. Check if the API server is running
2. Verify the API URL in settings
3. Click "Test Connection" in settings
4. Check browser console for detailed errors

### "Connection Failed" in Settings

1. Ensure the API server is running: `docker-compose up`
2. Verify the URL format (must include `http://` or `https://`)
3. Check firewall settings
4. For remote servers, ensure CORS is enabled

### Images Not Analyzing

1. Some websites may block extensions from accessing images
2. Try saving the image and using a different analysis tool
3. Check if the image URL is accessible

### Slow Analysis

- Analysis typically takes 2-5 seconds
- Larger images take longer to process
- Server performance affects speed
- First analysis may be slower (model loading)

## Privacy

- Images are sent to your configured API server for analysis
- No data is sent to third parties
- Results are stored locally in your browser
- The extension only accesses images when you explicitly request analysis

## Development

### Building from Source

```bash
cd chrome-extension
python3 create_simple_icons.py  # Generate icons
```

### Debugging

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Inspect views: service worker" for background script
4. Use browser DevTools for popup and options page

### Modifying the Extension

- Edit `manifest.json` for permissions and configuration
- Edit `js/background.js` for API integration
- Edit `popup.html` and `css/popup.css` for UI changes
- Edit `js/popup.js` for UI logic

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

See the main project LICENSE file.

## Credits

- Built with Chrome Extension Manifest V3
- Uses Vision Transformer (ViT) for AI detection
- YOLOv8 for object detection
- Google Gemini for explanations
- Grad-CAM for attention visualization

## Support

For issues and feature requests, please open an issue on GitHub.

## Version History

### 1.0.0 (Initial Release)
- Right-click image analysis
- AI vs Real classification
- Object detection
- Grad-CAM visualization
- AI-generated explanations
- Configurable API endpoint
- Beautiful gradient UI
