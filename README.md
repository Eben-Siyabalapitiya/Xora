# XORA

> A privacy-first indoor sensing platform that uses Wi-Fi signals and machine learning to detect human activity, occupancy, and movement without cameras.

## Overview

XORA is an experimental indoor sensing system built around ESP32 devices and Wi-Fi Channel State Information (CSI). Instead of relying on cameras, microphones, or wearable devices, XORA analyzes changes in wireless signals to understand activity within a space.

The goal is to create an affordable, scalable, and privacy-focused alternative to traditional indoor monitoring systems while exploring the potential of Wi-Fi sensing and machine learning.

---

## Features

-  Real-time Wi-Fi CSI data collection
-  Machine learning-based activity recognition
-  Human presence and movement detection
-  Live activity heatmaps
-  Data logging and analytics
-  Privacy-focused design (no cameras required)
-  Low-cost hardware using ESP32 devices
-  Real-time signal processing pipeline

---

## How It Works

### 1. Signal Transmission

An ESP32 transmitter continuously sends Wi-Fi packets throughout the environment.

### 2. CSI Collection

Multiple ESP32 receivers capture Channel State Information (CSI), which contains detailed information about how wireless signals propagate through a room.

### 3. Signal Processing

Raw CSI data is filtered, cleaned, and transformed into usable features.

### 4. Machine Learning

Trained models analyze the processed data to identify occupancy, movement patterns, and activities.

### 5. Visualization

Results are displayed through dashboards, heatmaps, and other real-time visualizations.

---

## Architecture

```text
┌─────────────┐
│ ESP32 TX    │
└──────┬──────┘
       │ Wi-Fi Signal
       ▼
┌─────────────────────┐
│ Environment / Room  │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ ESP32 RXs   │
└──────┬──────┘
       │ CSI Data
       ▼
┌─────────────┐
│ Processing  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ML Models   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Dashboard   │
└─────────────┘
```

---

## Hardware

### Required Components

- ESP32 transmitter
- One or more ESP32 receivers
- Local Wi-Fi network
- Computer for processing and visualization

### Recommended Setup

- 1 transmitter
- 3–5 receivers
- Dedicated processing computer
- Stable indoor environment for testing

---

## Software Stack

### Firmware

- ESP-IDF
- ESP32 CSI APIs

### Backend

- Python
- NumPy
- Pandas
- Scikit-learn

### Visualization

- Matplotlib
- Plotly
- Flask
- WebSockets

---

## Project Structure

```text
xora/
├── firmware/
│   ├── transmitter/
│   └── receiver/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── training/
│
├── visualization/
│
├── dashboard/
│
├── docs/
│
└── README.md
```

---

## Current Progress

### Completed

- CSI data collection pipeline
- ESP32 communication setup
- Data storage system
- Initial preprocessing workflow

### In Progress

- Activity classification models
- Real-time visualization tools
- Multi-device synchronization
- Occupancy detection improvements

### Planned

- Multi-room support
- 3D activity mapping
- Edge AI deployment
- Advanced localization algorithms
- Mobile dashboard

---

## Potential Applications

### Smart Homes

Monitor occupancy and room usage without cameras.

### Energy Optimization

Automate lighting, heating, and cooling based on room activity.

### Research

Study human activity recognition using Wi-Fi sensing.

### Accessibility & Elder Care

Detect movement patterns and unusual inactivity while maintaining privacy.

---

## Vision

Most indoor sensing solutions rely on cameras, microphones, or wearable devices. XORA explores a different approach by turning existing Wi-Fi signals into a sensing medium capable of understanding activity within a space.

The long-term vision is to create a privacy-preserving indoor intelligence platform that is affordable, scalable, and accessible to anyone.

---

## Status

 Active Development

XORA is currently an experimental research and engineering project. Features, architecture, and performance may change significantly as development continues.

---

## Author

### Eben Siyabalapitiya

Developer, builder, and student exploring machine learning, embedded systems, computer vision, and wireless sensing technologies.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
