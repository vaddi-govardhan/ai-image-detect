#!/usr/bin/env python3
"""
Generate simple icons for the Chrome extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple icon with AI and eye symbol"""
    # Create image with gradient-like background
    img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
    draw = ImageDraw.Draw(img)

    # Draw a circle background
    padding = size // 8
    draw.ellipse(
        [padding, padding, size - padding, size - padding],
        fill=(118, 75, 162, 255)
    )

    # Draw an eye shape (simple representation)
    center_x = size // 2
    center_y = size // 2
    eye_width = size // 3
    eye_height = size // 6

    # Eye outline
    draw.ellipse(
        [center_x - eye_width//2, center_y - eye_height//2,
         center_x + eye_width//2, center_y + eye_height//2],
        fill=(255, 255, 255, 255)
    )

    # Pupil
    pupil_size = size // 8
    draw.ellipse(
        [center_x - pupil_size//2, center_y - pupil_size//2,
         center_x + pupil_size//2, center_y + pupil_size//2],
        fill=(102, 126, 234, 255)
    )

    # Save
    img.save(output_path, 'PNG')
    print(f'Created icon: {output_path}')

# Create icons directory if it doesn't exist
icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
os.makedirs(icons_dir, exist_ok=True)

# Create icons in different sizes
create_icon(16, os.path.join(icons_dir, 'icon16.png'))
create_icon(48, os.path.join(icons_dir, 'icon48.png'))
create_icon(128, os.path.join(icons_dir, 'icon128.png'))

print('All icons created successfully!')
