import type { AnalyzeResponse, HealthStatus, Detection } from '../types';

const API_BASE = 'http://127.0.0.1:8000';

export async function fetchHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) {
      return {
        status: 'offline',
        models_loaded: false,
        detector_loaded: false,
        classifier_loaded: false,
        device: 'unknown',
        review_threshold: 0.60,
        classes: [],
      };
    }
    const data = await res.json();
    return data;
  } catch {
    return {
      status: 'offline',
      models_loaded: false,
      detector_loaded: false,
      classifier_loaded: false,
      device: 'unknown',
      review_threshold: 0.60,
      classes: [],
    };
  }
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  return await res.json();
}

export async function loadDemoResults(): Promise<Detection[]> {
  const res = await fetch('/demo_results.json');
  if (!res.ok) {
    throw new Error('Failed to load demo_results.json');
  }
  return await res.json();
}
