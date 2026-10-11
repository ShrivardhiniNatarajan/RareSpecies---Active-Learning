import React, { useState, useEffect } from 'react';
import { 
  X, ChevronLeft, ChevronRight, AlertTriangle, CheckCircle2, 
  Layers, Info 
} from 'lucide-react';
import type { Detection } from '../../types';

interface DetailDrawerProps {
  detection: Detection | null;
  currentIndex: number;
  totalCount: number;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}

export const DetailDrawer: React.FC<DetailDrawerProps> = ({
  detection,
  currentIndex,
  totalCount,
  onClose,
  onPrev,
  onNext,
}) => {
  const [showGradCam, setShowGradCam] = useState<boolean>(false);

  useEffect(() => {
    // Reset CAM toggle when detection changes
    setShowGradCam(false);
  }, [detection?.filename, detection?.animal_index]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') onPrev();
      if (e.key === 'ArrowRight') onNext();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, onPrev, onNext]);

  if (!detection) return null;

  const isNeedsReview = detection.review_status === 'NEEDS REVIEW';
  const pc = detection.classifier_confidence;
  const dc = detection.detector_confidence;

  // Normalized bbox coordinates [x, y, w, h] (0.0 to 1.0)
  const bbox = detection.bbox;

  return (
    <div 
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="target-card-title"
    >
      <div className="bg-surface border border-border rounded-hud max-w-5xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        
        {/* Target Card Header */}
        <div className="px-6 py-4 bg-raised border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="hud-label text-cyan-400 font-bold">
              TARGET // {detection.filename} (ANIMAL {detection.animal_index})
            </span>
            {isNeedsReview ? (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-950/60 text-rose-400 border border-rose-500/50">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>NEEDS REVIEW</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/60 text-emerald-accent border border-emerald-500/50">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>AUTO-PROCESSED</span>
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="hud-value-mono text-xs text-muted mr-2">
              [{currentIndex + 1} / {totalCount}]
            </span>
            <button
              onClick={onPrev}
              disabled={currentIndex <= 0}
              className="p-1.5 rounded-hud bg-surface hover:bg-raised border border-border text-muted hover:text-text disabled:opacity-40 disabled:cursor-not-allowed"
              aria-label="Previous animal"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={onNext}
              disabled={currentIndex >= totalCount - 1}
              className="p-1.5 rounded-hud bg-surface hover:bg-raised border border-border text-muted hover:text-text disabled:opacity-40 disabled:cursor-not-allowed"
              aria-label="Next animal"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-hud bg-surface hover:bg-rose-950/50 hover:text-rose-400 border border-border text-muted ml-2 transition-colors"
              aria-label="Close detail view"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Card Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Main Visuals Row: Full Frame + Crop Panel */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Full Frame with Cyan Viewfinder Bounding Box */}
            <div className="lg:col-span-7 bg-bg rounded-hud border border-border p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="hud-label">SURVEY FRAME (MEGADETECTOR V6)</span>
                {bbox && (
                  <span className="hud-value-mono text-[10px] text-cyan-400 bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-800/60">
                    BOX_CONF: {(dc * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              <div className="relative w-full aspect-[4/3] bg-black/60 rounded overflow-hidden flex items-center justify-center border border-border/60">
                {detection.frame_image ? (
                  <div className="relative w-full h-full">
                    <img
                      src={detection.frame_image}
                      alt="Survey Full Frame"
                      className="w-full h-full object-contain"
                    />
                    {/* Bounding Box Overlay with Corner Brackets */}
                    {bbox && (
                      <div
                        className="absolute border-2 border-dashed border-cyan-400 pointer-events-none transition-all"
                        style={{
                          left: `${bbox[0] * 100}%`,
                          top: `${bbox[1] * 100}%`,
                          width: `${bbox[2] * 100}%`,
                          height: `${bbox[3] * 100}%`,
                        }}
                      >
                        {/* Corner Brackets */}
                        <span className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-cyan-400" />
                        <span className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-cyan-400" />
                        <span className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-cyan-400" />
                        <span className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-cyan-400" />
                        
                        {/* Box Confidence Badge on Box */}
                        <div className="absolute -top-6 left-0 bg-cyan-950/90 text-cyan-300 text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border border-cyan-500/80 whitespace-nowrap shadow-md">
                          BOX_CONF: {(dc * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <span className="text-xs text-muted">Full frame image unavailable</span>
                )}
              </div>
            </div>

            {/* Animal Crop Panel with Grad-CAM Toggle */}
            <div className="lg:col-span-5 bg-bg rounded-hud border border-border p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="hud-label">ANIMAL CROP // CLASSIFIER INPUT</span>
                
                {detection.gradcam_overlay && (
                  <button
                    onClick={() => setShowGradCam(!showGradCam)}
                    className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded transition-colors font-mono ${
                      showGradCam
                        ? 'bg-cyan-500 text-bg font-bold'
                        : 'bg-raised text-muted hover:text-text border border-border'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>{showGradCam ? 'Grad-CAM: ON' : 'Grad-CAM: OFF'}</span>
                  </button>
                )}
              </div>

              {/* Crop display with object-fit: contain at readable resolution */}
              <div className="relative w-full aspect-square bg-black/70 rounded overflow-hidden flex items-center justify-center border border-border/60">
                <img
                  src={showGradCam && detection.gradcam_overlay ? detection.gradcam_overlay : detection.crop_image}
                  alt={`Cropped ${detection.species}`}
                  className="w-full h-full object-contain transition-opacity duration-200"
                />

                {showGradCam && (
                  <div className="absolute bottom-2 left-2 bg-black/80 text-cyan-300 px-2 py-0.5 rounded text-[10px] font-mono border border-cyan-500/40">
                    GRAD-CAM ATTENTION OVERLAY
                  </div>
                )}
              </div>
            </div>

          </div>

          {/* 2x3 Stat Tiles Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {/* Species */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">SPECIES</span>
              <div className="hud-value-mono text-base font-bold text-text capitalize mt-1">
                {detection.species.replace('_', ' ')}
              </div>
            </div>

            {/* Classifier Confidence */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">CLASSIFIER CONF</span>
              <div className="hud-value-mono text-base font-bold text-emerald-accent mt-1">
                {(pc * 100).toFixed(1)}%
              </div>
            </div>

            {/* Detector Confidence */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">DETECTOR CONF</span>
              <div className="hud-value-mono text-base font-bold text-cyan-400 mt-1">
                {(dc * 100).toFixed(1)}%
              </div>
            </div>

            {/* Predictive Entropy */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">PREDICTIVE ENTROPY</span>
              <div className="hud-value-mono text-base font-bold text-text mt-1">
                {detection.uncertainty.toFixed(3)} <span className="text-xs text-muted font-normal">{detection.uncertainty_unit}</span>
              </div>
            </div>

            {/* Rarity */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">RARITY SCORE</span>
              <div className="hud-value-mono text-base font-bold text-text mt-1">
                {detection.rarity.toFixed(3)}
              </div>
            </div>

            {/* Priority */}
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">PRIORITY SCORE</span>
              <div className="hud-value-mono text-base font-bold text-amber-500 mt-1">
                {detection.priority.toFixed(3)}
              </div>
            </div>
          </div>

          {/* DAAC Experimental Stat Tile (when present) */}
          {detection.daac !== null && (
            <div className="p-3.5 bg-raised/70 border border-border rounded-hud flex items-center justify-between">
              <div>
                <span className="hud-label text-emerald-accent">DAAC SCORE (EXPERIMENTAL)</span>
                <p className="text-xs text-muted mt-0.5">
                  Detection-Aligned Attention Consistency (|R ∩ B| / |R|). Measures alignment between Grad-CAM attention and the detector bounding box.
                </p>
              </div>
              <div className="hud-value-mono text-xl font-bold text-text bg-bg px-3 py-1 rounded border border-border ml-4">
                {detection.daac.toFixed(4)}
              </div>
            </div>
          )}

          {/* Plain-Language Explanation Line */}
          <div className={`p-4 rounded-hud border flex items-start gap-3 ${
            isNeedsReview
              ? 'bg-rose-950/20 border-rose-500/40 text-rose-200'
              : 'bg-emerald-950/20 border-emerald-500/40 text-emerald-200'
          }`}>
            <Info className={`w-5 h-5 shrink-0 mt-0.5 ${isNeedsReview ? 'text-rose-400' : 'text-emerald-accent'}`} />
            <div className="text-xs space-y-1">
              <span className="font-semibold block uppercase tracking-wider font-mono">
                {isNeedsReview ? 'REVIEW REQUIRED' : 'AUTOMATIC INFERENCE VERIFIED'}
              </span>
              <p className="font-normal text-text/90">
                {isNeedsReview
                  ? 'The classifier is not confident enough in this prediction. Classifier confidence is below the established 60.0% threshold.'
                  : 'Prediction confident enough for automatic processing. Classifier confidence exceeds the 60.0% threshold.'}
              </p>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};
