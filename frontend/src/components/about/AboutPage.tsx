import React from 'react';
import { 
  DATASET_INFO, 
  CLASSIFIER_INFO, 
  ACTIVE_LEARNING_INFO, 
  FULL_DATA_REFERENCE, 
  ROUND3_RESULTS_TABLE, 
  HONEST_NOTE 
} from '../../data/modelInfo';
import { Database, Cpu, Layers, AlertCircle } from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Page Title */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-text">About the Model</h2>
        <p className="text-sm text-muted mt-1">
          Technical specifications, sequence-aware experimental setup, and verified test results for RareCam.
        </p>
      </div>

      {/* Honest Note Panel */}
      <div className="p-4 bg-amber-950/30 border border-amber-500/40 rounded-hud flex items-start gap-3.5">
        <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="hud-label text-amber-400 font-bold">RESEARCH INTEGRITY & METHODOLOGICAL NOTE</span>
          <p className="text-xs text-amber-200/90 leading-relaxed font-mono">
            {HONEST_NOTE}
          </p>
        </div>
      </div>

      {/* 3 Core Spec Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Dataset */}
        <div className="p-5 bg-surface border border-border rounded-hud space-y-3">
          <div className="flex items-center gap-2 text-emerald-accent">
            <Database className="w-4 h-4" />
            <span className="hud-label text-emerald-accent font-bold">DATASET & SPLITS</span>
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <span className="text-muted block">Subset:</span>
              <span className="font-semibold text-text">{DATASET_INFO.name}</span>
            </div>
            <div>
              <span className="text-muted block">Source Images:</span>
              <span className="hud-value-mono font-medium text-text">{DATASET_INFO.sourceImages.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-muted block">MegaDetector Detections:</span>
              <span className="hud-value-mono font-medium text-text">{DATASET_INFO.imagesWithAnimals.toLocaleString()} images ({DATASET_INFO.animalCrops.toLocaleString()} animal crops)</span>
            </div>
            <div>
              <span className="text-muted block">Sequence-Aware Split:</span>
              <span className="hud-value-mono font-medium text-text">
                Train: {DATASET_INFO.splits.train} | Val: {DATASET_INFO.splits.val} | Test: {DATASET_INFO.splits.test}
              </span>
            </div>
            <div className="pt-1">
              <span className="text-muted block mb-1">Target Species (6):</span>
              <div className="flex flex-wrap gap-1">
                {DATASET_INFO.classes.map(c => (
                  <span key={c} className="px-1.5 py-0.5 rounded bg-raised border border-border text-[10px] font-mono text-cyan-300">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Classifier */}
        <div className="p-5 bg-surface border border-border rounded-hud space-y-3">
          <div className="flex items-center gap-2 text-cyan-400">
            <Cpu className="w-4 h-4" />
            <span className="hud-label text-cyan-400 font-bold">CLASSIFIER RECIPE</span>
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <span className="text-muted block">Architecture:</span>
              <span className="font-semibold text-text">{CLASSIFIER_INFO.architecture}</span>
            </div>
            <div>
              <span className="text-muted block">Optimizer & LR:</span>
              <span className="hud-value-mono font-medium text-text">{CLASSIFIER_INFO.optimizer} (lr: {CLASSIFIER_INFO.learningRate})</span>
            </div>
            <div>
              <span className="text-muted block">Batch Size & Resolution:</span>
              <span className="hud-value-mono font-medium text-text">{CLASSIFIER_INFO.batchSize} crops | {CLASSIFIER_INFO.inputResolution}</span>
            </div>
            <div>
              <span className="text-muted block">Epochs:</span>
              <span className="font-medium text-text">{CLASSIFIER_INFO.epochs}</span>
            </div>
          </div>
        </div>

        {/* Active Learning */}
        <div className="p-5 bg-surface border border-border rounded-hud space-y-3">
          <div className="flex items-center gap-2 text-amber-500">
            <Layers className="w-4 h-4" />
            <span className="hud-label text-amber-500 font-bold">ACTIVE LEARNING BUDGET</span>
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <span className="text-muted block">Seed Pool:</span>
              <span className="hud-value-mono font-medium text-text">{ACTIVE_LEARNING_INFO.seedSize} labeled samples</span>
            </div>
            <div>
              <span className="text-muted block">Unlabeled Pool:</span>
              <span className="hud-value-mono font-medium text-text">{ACTIVE_LEARNING_INFO.poolSize} samples</span>
            </div>
            <div>
              <span className="text-muted block">Rounds & Budget:</span>
              <span className="font-medium text-text">{ACTIVE_LEARNING_INFO.rounds} rounds × {ACTIVE_LEARNING_INFO.budgetPerRound} sequences/round</span>
            </div>
            <div>
              <span className="text-muted block">Final Round 3 Budget:</span>
              <span className="hud-value-mono font-semibold text-emerald-accent">
                {ACTIVE_LEARNING_INFO.finalLabeledBudget} samples ({ACTIVE_LEARNING_INFO.labeledPercentage})
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Results Benchmark Tables */}
      <div className="space-y-4">
        <div>
          <span className="hud-label">VERIFIED ROUND 3 TEST BENCHMARK</span>
          <p className="text-xs text-muted mt-0.5">
            Evaluated on the 970-crop sequence-aware test split at 1,394 labeled samples (30.2% data budget).
          </p>
        </div>

        <div className="border border-border rounded-hud overflow-hidden bg-surface shadow">
          <table className="w-full text-left text-xs">
            <thead className="bg-raised border-b border-border text-muted">
              <tr>
                <th className="p-3.5 font-semibold hud-label">ACQUISITION STRATEGY</th>
                <th className="p-3.5 font-semibold hud-label">TEST ACCURACY</th>
                <th className="p-3.5 font-semibold hud-label">MACRO-F1</th>
                <th className="p-3.5 font-semibold hud-label">MOUNTAIN LION RECALL (RAREST)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              {ROUND3_RESULTS_TABLE.map((row) => (
                <tr key={row.strategy} className="hover:bg-raised/40 transition-colors">
                  <td className="p-3.5 font-semibold text-text font-sans">{row.strategy}</td>
                  <td className="p-3.5 text-text font-bold">{row.accuracy.toFixed(2)}%</td>
                  <td className="p-3.5 text-text">{row.macroF1.toFixed(2)}%</td>
                  <td className="p-3.5 text-cyan-300 font-bold">{row.mountainLionRecall.toFixed(2)}%</td>
                </tr>
              ))}
              {/* Full-data Reference Row */}
              <tr className="bg-raised/60 border-t-2 border-border font-semibold text-muted">
                <td className="p-3.5 text-emerald-accent font-sans">Full-Data Reference (4,617 crops)</td>
                <td className="p-3.5 text-emerald-accent">{FULL_DATA_REFERENCE.accuracy.toFixed(2)}%</td>
                <td className="p-3.5 text-emerald-accent">{FULL_DATA_REFERENCE.macroF1.toFixed(2)}%</td>
                <td className="p-3.5 text-emerald-accent">{FULL_DATA_REFERENCE.mountainLionRecall.toFixed(2)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
