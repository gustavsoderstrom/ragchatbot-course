# Frontend Changes: Dark/Light Mode Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light modes. The toggle is positioned in the top-right corner with smooth animations and full accessibility support.

## Files Modified

### `frontend/index.html`
- Added a theme toggle button with sun/moon SVG icons
- Button is positioned before the main container for proper z-index layering
- Includes `aria-label` and `title` attributes for accessibility

### `frontend/style.css`
- Added `[data-theme="light"]` CSS variables for the light color scheme
- Added `.theme-toggle` button styles with:
  - Fixed positioning (top-right corner)
  - Circular button with hover/focus states
  - Sun/moon icon rotation and fade animations
- Added smooth transitions (0.3s) to:
  - Body background and text color
  - Sidebar background and border
  - Message content boxes
  - Chat input container and input field
- Added semantic color CSS variables:
  - `--code-bg` for consistent code block styling
  - `--link-color` and `--link-hover` for links
  - `--success-color`, `--success-bg` for success indicators
  - `--warning-color`, `--warning-bg` for warning indicators
  - `--error-color`, `--error-bg` for error indicators
- Updated all hardcoded colors to use CSS variables for theme consistency

### `frontend/script.js`
- Added `initTheme()` function that:
  - Checks for saved theme in localStorage
  - Falls back to system preference (`prefers-color-scheme`)
  - Defaults to dark theme if no preference
- Added `setTheme(theme)` function to apply theme via `data-theme` attribute on `<html>` element
- Added `toggleTheme()` function to switch between themes on button click
- Theme is initialized immediately on script load (before DOMContentLoaded) to prevent flash
- Added event listener for the toggle button

## Features

### Design
- Circular button (44x44px) with sun/moon icons
- Smooth icon transition with rotation and scale effects
- Matches existing design aesthetic with consistent colors and shadows
- Positioned in top-right corner, always visible

### Accessibility
- Keyboard navigable (focusable button)
- `aria-label="Toggle dark/light mode"` for screen readers
- `title="Toggle theme"` for tooltip
- Visible focus ring using existing `--focus-ring` color
- 44px touch target size for mobile accessibility
- WCAG AA compliant contrast ratios in both themes

### Persistence
- Theme preference saved to localStorage
- Respects system preference if no saved preference exists
- Theme applied immediately on page load to prevent flash of wrong theme

### Animations
- 0.3s smooth transitions for all color changes
- Icon swap animation with rotation (90deg) and scale (0.5 to 1)
- Button hover effect with slight scale increase
- Button active effect with scale decrease

## Color Schemes

### Dark Theme (Default)
| Property | Value | Usage |
|----------|-------|-------|
| `--background` | `#0f172a` | Page background |
| `--surface` | `#1e293b` | Cards, sidebar |
| `--surface-hover` | `#334155` | Hover states |
| `--text-primary` | `#f1f5f9` | Main text |
| `--text-secondary` | `#94a3b8` | Secondary text |
| `--border-color` | `#334155` | Borders |
| `--link-color` | `#60a5fa` | Links |
| `--link-hover` | `#93c5fd` | Link hover |
| `--success-color` | `#4ade80` | Success text |
| `--warning-color` | `#fbbf24` | Warning text |
| `--error-color` | `#f87171` | Error text |

### Light Theme
| Property | Value | Usage |
|----------|-------|-------|
| `--background` | `#f8fafc` | Page background |
| `--surface` | `#ffffff` | Cards, sidebar |
| `--surface-hover` | `#f1f5f9` | Hover states |
| `--text-primary` | `#1e293b` | Main text |
| `--text-secondary` | `#64748b` | Secondary text |
| `--border-color` | `#e2e8f0` | Borders |
| `--link-color` | `#2563eb` | Links (darker for contrast) |
| `--link-hover` | `#1d4ed8` | Link hover |
| `--success-color` | `#16a34a` | Success text (darker) |
| `--warning-color` | `#d97706` | Warning text (darker) |
| `--error-color` | `#dc2626` | Error text (darker) |

## Implementation Details

### Theme Switching Mechanism
1. Uses `data-theme` attribute on `<html>` element
2. CSS variables cascade down to all elements
3. Dark theme is the default (no attribute needed)
4. Light theme activated with `data-theme="light"`

### JavaScript Theme Management
```javascript
// Initialize on page load
initTheme();

// Toggle on button click
toggleTheme();

// Set specific theme
setTheme('light'); // or 'dark'
```

### CSS Variable Usage
All colors reference CSS variables, enabling instant theme switching:
```css
.element {
    background: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}
```
