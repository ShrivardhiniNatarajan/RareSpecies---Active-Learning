export interface Detection {
  filename: string;
  animal_index: number;
  species: string;
  classifier_confidence: number;
  detector_confidence: number;
  uncertainty: number;
  uncertainty_unit: 'nats' | 'bits';
  rarity: number;
  priority: number;
  daac: number | null;
  review_status: 'AUTO-PROCESSED' | 'NEEDS REVIEW';
  bbox: [number, number, number, number] | null; // normalized [x, y, w, h]
  frame_image: string | null; // data URL
  crop_image: string; // data URL
  gradcam_overlay: string | null; // data URL
}

export interface AnalyzeResponse {
  filename: string;
  no_animal: boolean;
  detections: Detection[];
}

export interface HealthStatus {
  status: 'ok' | 'initializing' | 'offline';
  models_loaded: boolean;
  detector_loaded: boolean;
  classifier_loaded: boolean;
  device: string;
  review_threshold: number;
  classes: string[];
}

export type PageId = 'upload' | 'batch' | 'about';
