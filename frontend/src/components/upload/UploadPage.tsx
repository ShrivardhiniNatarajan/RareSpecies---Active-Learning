import React, { useState, useRef } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { UploadCloud, FileImage, X, Play, ArrowRight, AlertCircle, CheckCircle, Database, RefreshCw } from 'lucide-react';
import type { Detection } from '../../types';
import { analyzeImage, loadDemoResults } from '../../api/client';

interface UploadPageProps {
  onAnalysisComplete: (detections: Detection[], isDemo: boolean) => void;
  onNavigateToBatch: () => void;
  isDemoMode: boolean;
  setIsDemoMode: (isDemo: boolean) => void;
}

export const UploadPage: React.FC<UploadPageProps> = ({
  onAnalysisComplete,
  onNavigateToBatch,
  isDemoMode,
  setIsDemoMode,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<{ current: number; total: number; currentFileName: string } | null>(null);
  const [errorMessages, setErrorMessages] = useState<string[]>([]);
  const [lastBatchSummary, setLastBatchSummary] = useState<{
    imagesProcessed: number;
    animalsDetected: number;
    needsReview: number;
    autoProcessed: number;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
  };

  const addFiles = (files: File[]) => {
    const valid = files.filter(f => /\.(jpe?g|png)$/i.test(f.name));
    setSelectedFiles(prev => {
      const existingNames = new Set(prev.map(f => f.name));
      const newFiles = valid.filter(f => !existingNames.has(f.name));
      return [...prev, ...newFiles];
    });
    setErrorMessages([]);
  };

  const removeFile = (name: string) => {
    setSelectedFiles(prev => prev.filter(f => f.name !== name));
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) return;

    setIsProcessing(true);
    setErrorMessages([]);
    setLastBatchSummary(null);
    setIsDemoMode(false);

    const accumulatedDetections: Detection[] = [];
    const errors: string[] = [];
    let imagesWithAnimalsCount = 0;

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      setCurrentProgress({
        current: i + 1,
        total: selectedFiles.length,
        currentFileName: file.name,
      });

      try {
        const response = await analyzeImage(file);
        if (response.no_animal || response.detections.length === 0) {
          // Graceful no animal found
        } else {
          imagesWithAnimalsCount++;
          accumulatedDetections.push(...response.detections);
        }
      } catch (err: any) {
        errors.push(`${file.name}: ${err?.message || 'Inference error'}`);
      }
    }

    const needsReviewCount = accumulatedDetections.filter(d => d.review_status === 'NEEDS REVIEW').length;
    const autoProcessedCount = accumulatedDetections.filter(d => d.review_status === 'AUTO-PROCESSED').length;

    setLastBatchSummary({
      imagesProcessed: selectedFiles.length,
      animalsDetected: accumulatedDetections.length,
      needsReview: needsReviewCount,
      autoProcessed: autoProcessedCount,
    });

    setErrorMessages(errors);
    setIsProcessing(false);
    setCurrentProgress(null);
    onAnalysisComplete(accumulatedDetections, false);
  };

  const handleLoadDemo = async () => {
    try {
      setIsProcessing(true);
      setErrorMessages([]);
      const demoData = await loadDemoResults();
      setIsDemoMode(true);
      const needsReviewCount = demoData.filter(d => d.review_status === 'NEEDS REVIEW').length;
      const autoProcessedCount = demoData.filter(d => d.review_status === 'AUTO-PROCESSED').length;
      const uniqueImages = new Set(demoData.map(d => d.filename)).size;

      setLastBatchSummary({
        imagesProcessed: uniqueImages,
        animalsDetected: demoData.length,
        needsReview: needsReviewCount,
        autoProcessed: autoProcessedCount,
      });

      onAnalysisComplete(demoData, true);
    } catch (err: any) {
      setErrorMessages([`Could not load cached results: ${err?.message || 'Check public/demo_results.json'}`]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-text">Upload & Analyze</h2>
        <p className="text-sm text-muted mt-1">
          Upload camera-trap survey frames for MegaDetector localization, ResNet-50 species classification, and active learning review prioritization.
        </p>
      </div>

      {/* Offline Demo Banner */}
      {isDemoMode && (
        <div className="p-3.5 bg-cyan-950/40 border border-cyan-400/40 rounded-hud flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-cyan-400" />
            <div>
              <span className="text-sm font-semibold text-cyan-200">Showing cached results (offline demo)</span>
              <p className="text-xs text-cyan-400/80">
                Loaded verified pre-computed outputs from public/demo_results.json for reliable offline evaluation.
              </p>
            </div>
          </div>
          <button
            onClick={() => setIsDemoMode(false)}
            className="text-xs px-2.5 py-1 rounded bg-cyan-900/50 hover:bg-cyan-800/50 text-cyan-200 border border-cyan-700/60"
          >
            Clear Demo
          </button>
        </div>
      )}

      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`hud-viewfinder border-2 border-dashed rounded-hud p-10 flex flex-col items-center justify-center cursor-pointer transition-all ${
          isDragging
            ? 'border-cyan-400 bg-raised/70'
            : 'border-border bg-surface/80 hover:bg-surface hover:border-muted/50'
        }`}
      >
        <span className="hud-viewfinder-tr" />
        <span className="hud-viewfinder-bl" />
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="w-14 h-14 rounded-full bg-raised flex items-center justify-center border border-border mb-4">
          <UploadCloud className="w-7 h-7 text-emerald-accent" />
        </div>
        <p className="text-base font-semibold text-text">
          Drag & drop camera-trap images, or <span className="text-emerald-accent underline underline-offset-2">browse files</span>
        </p>
        <p className="text-xs text-muted mt-1.5 font-normal">
          Supported formats: JPEG, PNG. MegaDetector V6 will automatically localize animal detections.
        </p>
      </div>

      {/* File Chips */}
      {selectedFiles.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="hud-label">SELECTED FILES ({selectedFiles.length})</span>
            <button
              onClick={() => setSelectedFiles([])}
              className="text-xs text-muted hover:text-rose-400 transition-colors"
            >
              Clear All
            </button>
          </div>
          <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto p-2 bg-bg/50 rounded-hud border border-border">
            {selectedFiles.map(file => (
              <div
                key={file.name}
                className="flex items-center gap-2 px-3 py-1.5 bg-raised border border-border rounded-hud text-xs text-text group"
              >
                <FileImage className="w-3.5 h-3.5 text-muted group-hover:text-cyan-400" />
                <span className="truncate max-w-[180px] font-mono">{file.name}</span>
                <span className="text-[10px] text-muted">({(file.size / 1024).toFixed(0)} KB)</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(file.name);
                  }}
                  className="text-muted hover:text-rose-400 transition-colors p-0.5"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap items-center gap-4">
        <button
          onClick={handleAnalyze}
          disabled={selectedFiles.length === 0 || isProcessing}
          className={`flex items-center gap-2.5 px-6 py-3 rounded-hud font-semibold text-sm transition-all shadow-md ${
            selectedFiles.length === 0 || isProcessing
              ? 'bg-raised text-muted cursor-not-allowed border border-border'
              : 'bg-emerald-accent text-bg hover:bg-emerald-400 active:scale-[0.98]'
          }`}
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Analyzing Batch...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Analyze {selectedFiles.length > 0 ? `(${selectedFiles.length}) Images` : 'Images'}</span>
            </>
          )}
        </button>

        <button
          onClick={handleLoadDemo}
          disabled={isProcessing}
          className="flex items-center gap-2 px-4 py-3 rounded-hud font-medium text-xs text-muted hover:text-text bg-raised hover:bg-surface border border-border transition-colors"
          title="Load pre-computed sample inferences from public/demo_results.json"
        >
          <Database className="w-4 h-4 text-cyan-400" />
          <span>Load Cached Results (Offline Demo)</span>
        </button>
      </div>

      {/* Live Per-File Progress Bar */}
      {isProcessing && currentProgress && (
        <div className="p-4 bg-raised border border-border rounded-hud space-y-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="hud-label">ANALYSIS PROGRESS</span>
            <span className="hud-value-mono font-medium text-text">
              {currentProgress.current} / {currentProgress.total} Files
            </span>
          </div>
          <div className="w-full bg-bg h-2 rounded-full overflow-hidden border border-border">
            <div
              className="bg-emerald-accent h-full transition-all duration-300"
              style={{ width: `${(currentProgress.current / currentProgress.total) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted font-mono truncate">
            Processing: <span className="text-text">{currentProgress.currentFileName}</span>
          </p>
        </div>
      )}

      {/* Errors display */}
      {errorMessages.length > 0 && (
        <div className="p-4 bg-rose-950/30 border border-rose-500/40 rounded-hud space-y-2">
          <div className="flex items-center gap-2 text-rose-400 text-xs font-semibold">
            <AlertCircle className="w-4 h-4" />
            <span>Processing Issues Detected:</span>
          </div>
          <ul className="text-xs text-rose-200/90 font-mono space-y-1 max-h-32 overflow-y-auto pl-6 list-disc">
            {errorMessages.map((msg, idx) => (
              <li key={idx}>{msg}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Summary Strip (Visible after processing) */}
      {lastBatchSummary && (
        <div className="p-5 bg-surface border border-border rounded-hud space-y-4">
          <div className="flex items-center justify-between">
            <span className="hud-label text-emerald-accent flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4" /> BATCH PROCESSING SUMMARY
            </span>
            <button
              onClick={onNavigateToBatch}
              className="flex items-center gap-1.5 text-xs font-semibold text-emerald-accent hover:text-emerald-300 transition-colors"
            >
              <span>View Batch Results</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">IMAGES PROCESSED</span>
              <div className="hud-value-mono text-xl font-bold text-text mt-1">
                {lastBatchSummary.imagesProcessed}
              </div>
            </div>

            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">ANIMALS DETECTED</span>
              <div className="hud-value-mono text-xl font-bold text-cyan-400 mt-1">
                {lastBatchSummary.animalsDetected}
              </div>
            </div>

            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">NEEDS REVIEW</span>
              <div className="hud-value-mono text-xl font-bold text-rose-400 mt-1">
                {lastBatchSummary.needsReview}
              </div>
            </div>

            <div className="p-3 bg-raised border border-border rounded-hud">
              <span className="hud-label">AUTO-PROCESSED</span>
              <div className="hud-value-mono text-xl font-bold text-emerald-accent mt-1">
                {lastBatchSummary.autoProcessed}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
