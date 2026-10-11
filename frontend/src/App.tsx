import React, { useState, useEffect } from 'react';
import type { PageId, Detection, HealthStatus } from './types';
import { fetchHealth } from './api/client';
import { Sidebar } from './components/layout/Sidebar';
import { UploadPage } from './components/upload/UploadPage';
import { BatchResultsPage } from './components/results/BatchResultsPage';
import { AboutPage } from './components/about/AboutPage';
import { DetailDrawer } from './components/detail/DetailDrawer';
import { AlertCircle, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('upload');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetectionIndex, setSelectedDetectionIndex] = useState<number | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus>({
    status: 'initializing',
    models_loaded: false,
    detector_loaded: false,
    classifier_loaded: false,
    device: 'cpu',
    review_threshold: 0.60,
    classes: [],
  });

  const checkHealth = async () => {
    const res = await fetchHealth();
    setHealth(res);
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);

    const params = new URLSearchParams(window.location.search);
    const pageParam = params.get('page');
    if (pageParam === 'batch' || pageParam === 'about' || pageParam === 'upload') {
      setCurrentPage(pageParam as PageId);
    }
    if (params.get('demo') === '1' || pageParam === 'batch') {
      import('./api/client').then(({ loadDemoResults }) => {
        loadDemoResults().then(data => {
          setDetections(data);
          setIsDemoMode(true);
          const detParam = params.get('detail');
          if (detParam !== null) {
            const idx = parseInt(detParam, 10);
            if (!isNaN(idx) && idx >= 0 && idx < data.length) {
              setSelectedDetectionIndex(idx);
            }
          }
        }).catch(() => {});
      });
    }

    return () => clearInterval(interval);
  }, []);

  const handleAnalysisComplete = (newDetections: Detection[], isDemo: boolean) => {
    setDetections(newDetections);
    setIsDemoMode(isDemo);
  };

  const handleSelectDetection = (_detection: Detection, index: number) => {
    setSelectedDetectionIndex(index);
  };

  const handleCloseDetail = () => {
    setSelectedDetectionIndex(null);
  };

  const handlePrevDetail = () => {
    if (selectedDetectionIndex !== null && selectedDetectionIndex > 0) {
      setSelectedDetectionIndex(selectedDetectionIndex - 1);
    }
  };

  const handleNextDetail = () => {
    if (selectedDetectionIndex !== null && selectedDetectionIndex < detections.length - 1) {
      setSelectedDetectionIndex(selectedDetectionIndex + 1);
    }
  };

  const selectedDetection = selectedDetectionIndex !== null ? detections[selectedDetectionIndex] : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-text">
      {/* Fixed Left Sidebar */}
      <Sidebar
        currentPage={currentPage}
        onSelectPage={setCurrentPage}
        health={health}
        batchCount={detections.length}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto min-w-0">
        {/* Backend Offline Banner */}
        {health.status === 'offline' && !isDemoMode && (
          <div className="bg-rose-950/80 border-b border-rose-500/50 px-6 py-2.5 flex items-center justify-between text-xs text-rose-200 shrink-0">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>
                Backend server is unreachable on port 8000. Start the server with <code className="font-mono bg-rose-900/60 px-1 py-0.5 rounded">npm run dev</code> or test with cached results.
              </span>
            </div>
            <button
              onClick={checkHealth}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-900 hover:bg-rose-800 text-rose-100 font-medium transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Page Render */}
        <div className="flex-1 pb-12">
          {currentPage === 'upload' && (
            <UploadPage
              onAnalysisComplete={handleAnalysisComplete}
              onNavigateToBatch={() => setCurrentPage('batch')}
              isDemoMode={isDemoMode}
              setIsDemoMode={setIsDemoMode}
            />
          )}

          {currentPage === 'batch' && (
            <BatchResultsPage
              detections={detections}
              onSelectDetection={handleSelectDetection}
              onNavigateToUpload={() => setCurrentPage('upload')}
            />
          )}

          {currentPage === 'about' && <AboutPage />}
        </div>
      </main>

      {/* Detail Drawer Modal */}
      {selectedDetection && (
        <DetailDrawer
          detection={selectedDetection}
          currentIndex={selectedDetectionIndex!}
          totalCount={detections.length}
          onClose={handleCloseDetail}
          onPrev={handlePrevDetail}
          onNext={handleNextDetail}
        />
      )}
    </div>
  );
};

export default App;
