import React, { useState, useMemo } from 'react';
import { 
  Search, Download, ArrowUpDown, ArrowUp, ArrowDown, 
  AlertTriangle, CheckCircle2, ChevronRight, BarChart2 
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import type { Detection } from '../../types';

interface BatchResultsPageProps {
  detections: Detection[];
  onSelectDetection: (detection: Detection, index: number) => void;
  onNavigateToUpload: () => void;
}

type SortField = 'species' | 'classifier_confidence' | 'priority' | 'daac' | 'uncertainty' | 'rarity' | 'review_status';

export const BatchResultsPage: React.FC<BatchResultsPageProps> = ({
  detections,
  onSelectDetection,
  onNavigateToUpload,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSpecies, setSelectedSpecies] = useState<string>('all');
  const [selectedReviewStatus, setSelectedReviewStatus] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField | null>(null);
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  // Unique species list for filtering
  const allSpecies = useMemo(() => {
    const set = new Set(detections.map(d => d.species));
    return Array.from(set).sort();
  }, [detections]);

  // Filtering and sorting
  const filteredDetections = useMemo(() => {
    let list = detections.filter(d => {
      const matchSearch = d.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          d.species.toLowerCase().includes(searchQuery.toLowerCase());
      const matchSpecies = selectedSpecies === 'all' || d.species === selectedSpecies;
      const matchStatus = selectedReviewStatus === 'all' || d.review_status === selectedReviewStatus;
      return matchSearch && matchSpecies && matchStatus;
    });

    if (sortField) {
      list.sort((a, b) => {
        let valA: any = a[sortField];
        let valB: any = b[sortField];
        if (valA === null || valA === undefined) valA = -Infinity;
        if (valB === null || valB === undefined) valB = -Infinity;
        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
      });
    } else {
      // Default ordering: Needs Review first, then Priority descending
      list.sort((a, b) => {
        if (a.review_status !== b.review_status) {
          return a.review_status === 'NEEDS REVIEW' ? -1 : 1;
        }
        return b.priority - a.priority;
      });
    }

    return list;
  }, [detections, searchQuery, selectedSpecies, selectedReviewStatus, sortField, sortAsc]);

  // Species distribution data for Recharts
  const speciesDistributionData = useMemo(() => {
    const counts: Record<string, number> = {};
    detections.forEach(d => {
      const sp = d.species.replace('_', ' ');
      counts[sp] = (counts[sp] || 0) + 1;
    });
    return Object.entries(counts).map(([name, count]) => ({
      name,
      count,
    }));
  }, [detections]);

  // Confidence histogram buckets (0.0 to 1.0 in 10 buckets)
  const confidenceHistogramData = useMemo(() => {
    const buckets = [
      { range: '0-20%', min: 0.0, max: 0.2, count: 0 },
      { range: '20-40%', min: 0.2, max: 0.4, count: 0 },
      { range: '40-60%', min: 0.4, max: 0.6, count: 0 },
      { range: '60-80%', min: 0.6, max: 0.8, count: 0 },
      { range: '80-100%', min: 0.8, max: 1.01, count: 0 },
    ];
    detections.forEach(d => {
      const conf = d.classifier_confidence;
      const bucket = buckets.find(b => conf >= b.min && conf < b.max);
      if (bucket) bucket.count++;
    });
    return buckets;
  }, [detections]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      if (!sortAsc) {
        setSortAsc(true);
      } else {
        setSortField(null);
      }
    } else {
      setSortField(field);
      setSortAsc(false); // default descending
    }
  };

  const exportCSV = () => {
    if (detections.length === 0) return;
    const headers = [
      'filename',
      'animal_index',
      'species',
      'classifier_confidence',
      'detector_confidence',
      'uncertainty_nats',
      'rarity',
      'priority',
      'daac_score',
      'review_status',
    ];

    const rows = filteredDetections.map(d => [
      `"${d.filename}"`,
      d.animal_index,
      `"${d.species}"`,
      d.classifier_confidence,
      d.detector_confidence,
      d.uncertainty,
      d.rarity,
      d.priority,
      d.daac ?? '',
      `"${d.review_status}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `rarecam_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (detections.length === 0) {
    return (
      <div className="p-12 max-w-xl mx-auto text-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-raised border border-border flex items-center justify-center mx-auto text-muted">
          <BarChart2 className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-text">No Batch Results Yet</h3>
        <p className="text-sm text-muted">
          Process images on the Upload & Analyze page or load cached results to inspect detections here.
        </p>
        <button
          onClick={onNavigateToUpload}
          className="px-5 py-2.5 rounded-hud bg-emerald-accent text-bg font-semibold text-sm hover:bg-emerald-400 transition-colors"
        >
          Go to Upload & Analyze
        </button>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Page Header and Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-text">Batch Results</h2>
          <p className="text-sm text-muted mt-0.5">
            Showing {filteredDetections.length} of {detections.length} animal detections. Rows flagged for review are pinned to the top.
          </p>
        </div>

        <button
          onClick={exportCSV}
          className="flex items-center gap-2 px-4 py-2 rounded-hud bg-raised hover:bg-surface border border-border text-xs font-semibold text-text transition-colors self-start md:self-auto"
        >
          <Download className="w-4 h-4 text-emerald-accent" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Visual Analytics Strip (Species Distribution + Confidence Histogram) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Species Distribution Chart */}
        <div className="p-4 bg-surface border border-border rounded-hud">
          <span className="hud-label block mb-2">SPECIES DISTRIBUTION</span>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={speciesDistributionData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis 
                  dataKey="name" 
                  tick={{ fill: '#8CA3AB', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis 
                  allowDecimals={false}
                  tick={{ fill: '#8CA3AB', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#111B1F', borderColor: '#22353C', borderRadius: '8px', color: '#E6EEF0' }}
                  itemStyle={{ color: '#10B981' }}
                />
                <Bar dataKey="count" fill="#10B981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Classifier Confidence Histogram */}
        <div className="p-4 bg-surface border border-border rounded-hud">
          <span className="hud-label block mb-2">CLASSIFIER CONFIDENCE SPREAD</span>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confidenceHistogramData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <XAxis 
                  dataKey="range" 
                  tick={{ fill: '#8CA3AB', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                />
                <YAxis 
                  allowDecimals={false}
                  tick={{ fill: '#8CA3AB', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#111B1F', borderColor: '#22353C', borderRadius: '8px', color: '#E6EEF0' }}
                  itemStyle={{ color: '#38BDF8' }}
                />
                <Bar dataKey="count" fill="#38BDF8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-3 bg-surface border border-border rounded-hud flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 text-muted absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename or species..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-raised border border-border rounded-hud text-xs text-text placeholder-muted focus:outline-none focus:border-cyan-400 font-mono"
          />
        </div>

        {/* Species Filter */}
        <select
          value={selectedSpecies}
          onChange={(e) => setSelectedSpecies(e.target.value)}
          className="px-3 py-1.5 bg-raised border border-border rounded-hud text-xs text-text focus:outline-none focus:border-cyan-400 capitalize font-mono"
        >
          <option value="all">All Species ({detections.length})</option>
          {allSpecies.map(sp => (
            <option key={sp} value={sp}>
              {sp.replace('_', ' ')}
            </option>
          ))}
        </select>

        {/* Review Status Filter */}
        <select
          value={selectedReviewStatus}
          onChange={(e) => setSelectedReviewStatus(e.target.value)}
          className="px-3 py-1.5 bg-raised border border-border rounded-hud text-xs text-text focus:outline-none focus:border-cyan-400 font-mono"
        >
          <option value="all">All Statuses</option>
          <option value="NEEDS REVIEW">Needs Review</option>
          <option value="AUTO-PROCESSED">Auto-Processed</option>
        </select>
      </div>

      {/* Detections Data Table */}
      <div className="border border-border rounded-hud overflow-hidden bg-surface shadow-lg">
        <div className="max-h-[520px] overflow-y-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-raised sticky top-0 z-10 border-b border-border text-muted">
              <tr>
                <th className="p-3 font-semibold hud-label w-16 text-center">PREVIEW</th>
                <th className="p-3 font-semibold hud-label">FILENAME</th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('species')}>
                  <div className="flex items-center gap-1">
                    <span>SPECIES</span>
                    {sortField === 'species' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('classifier_confidence')}>
                  <div className="flex items-center gap-1">
                    <span>CONFIDENCE</span>
                    {sortField === 'classifier_confidence' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('priority')}>
                  <div className="flex items-center gap-1">
                    <span>PRIORITY</span>
                    {sortField === 'priority' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('daac')}>
                  <div className="flex items-center gap-1">
                    <span>DAAC (EXPERIMENTAL)</span>
                    {sortField === 'daac' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('uncertainty')}>
                  <div className="flex items-center gap-1">
                    <span>UNCERTAINTY</span>
                    {sortField === 'uncertainty' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('rarity')}>
                  <div className="flex items-center gap-1">
                    <span>RARITY</span>
                    {sortField === 'rarity' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label cursor-pointer hover:text-text" onClick={() => toggleSort('review_status')}>
                  <div className="flex items-center gap-1">
                    <span>REVIEW STATUS</span>
                    {sortField === 'review_status' ? (sortAsc ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 text-muted/50" />}
                  </div>
                </th>
                <th className="p-3 font-semibold hud-label w-10"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {filteredDetections.map((detection, idx) => {
                const isNeedsReview = detection.review_status === 'NEEDS REVIEW';
                const pc = detection.classifier_confidence;

                return (
                  <tr
                    key={`${detection.filename}-${detection.animal_index}-${idx}`}
                    onClick={() => onSelectDetection(detection, idx)}
                    className="hover:bg-raised/80 transition-colors cursor-pointer group"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelectDetection(detection, idx);
                      }
                    }}
                  >
                    {/* Thumbnail Preview */}
                    <td className="p-2.5 text-center">
                      <div className="w-12 h-12 rounded bg-bg overflow-hidden border border-border mx-auto flex items-center justify-center">
                        <img
                          src={detection.crop_image}
                          alt={detection.species}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      </div>
                    </td>

                    {/* Filename with tooltip */}
                    <td className="p-3 font-mono text-text">
                      <span title={detection.filename} className="truncate max-w-[140px] block">
                        {detection.filename}
                      </span>
                    </td>

                    {/* Species */}
                    <td className="p-3 font-medium text-text capitalize">
                      {detection.species.replace('_', ' ')}
                    </td>

                    {/* Classifier Confidence (Bar + Number) */}
                    <td className="p-3 min-w-[130px]">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-bg h-2 rounded-full overflow-hidden border border-border/80">
                          <div
                            className={`h-full ${
                              pc >= 0.8 ? 'bg-emerald-accent' : pc >= 0.6 ? 'bg-amber-500' : 'bg-rose-500'
                            }`}
                            style={{ width: `${pc * 100}%` }}
                          />
                        </div>
                        <span className="hud-value-mono text-xs font-semibold">
                          {(pc * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>

                    {/* Priority Score (Bar + Number) */}
                    <td className="p-3 min-w-[120px]">
                      <div className="flex items-center gap-2">
                        <div className="w-14 bg-bg h-2 rounded-full overflow-hidden border border-border/80">
                          <div
                            className="h-full bg-cyan-400"
                            style={{ width: `${Math.min(100, (detection.priority / 5) * 100)}%` }}
                          />
                        </div>
                        <span className="hud-value-mono text-xs font-medium">
                          {detection.priority.toFixed(3)}
                        </span>
                      </div>
                    </td>

                    {/* DAAC (experimental) */}
                    <td className="p-3 font-mono text-xs">
                      {detection.daac !== null ? (
                        <div className="flex items-center gap-2">
                          <div className="w-12 bg-bg h-2 rounded-full overflow-hidden border border-border/80">
                            <div
                              className="h-full bg-emerald-accent"
                              style={{ width: `${detection.daac * 100}%` }}
                            />
                          </div>
                          <span>{detection.daac.toFixed(3)}</span>
                        </div>
                      ) : (
                        <span className="text-muted/60">—</span>
                      )}
                    </td>

                    {/* Uncertainty */}
                    <td className="p-3 font-mono text-xs text-muted">
                      {detection.uncertainty.toFixed(3)} {detection.uncertainty_unit}
                    </td>

                    {/* Rarity */}
                    <td className="p-3 font-mono text-xs text-muted">
                      {detection.rarity.toFixed(3)}
                    </td>

                    {/* Review Status Badge */}
                    <td className="p-3">
                      {isNeedsReview ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-rose-950/40 text-rose-400 border border-rose-500/40">
                          <AlertTriangle className="w-3 h-3 text-rose-400 shrink-0" />
                          <span>Needs Review</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-950/40 text-emerald-accent border border-emerald-500/40">
                          <CheckCircle2 className="w-3 h-3 text-emerald-accent shrink-0" />
                          <span>Auto-Processed</span>
                        </span>
                      )}
                    </td>

                    {/* Row Open Arrow */}
                    <td className="p-3 text-right">
                      <ChevronRight className="w-4 h-4 text-muted group-hover:text-cyan-400 transition-colors inline" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
