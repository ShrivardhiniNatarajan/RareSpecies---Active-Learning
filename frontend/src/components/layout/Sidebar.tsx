import React from 'react';
import { Upload, Table, Info } from 'lucide-react';
import type { PageId, HealthStatus } from '../../types';

interface SidebarProps {
  currentPage: PageId;
  onSelectPage: (page: PageId) => void;
  health: HealthStatus;
  batchCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onSelectPage,
  health,
  batchCount,
}) => {
  const navItems = [
    { id: 'upload' as PageId, label: 'Upload & Analyze', icon: Upload },
    { id: 'batch' as PageId, label: 'Batch Results', icon: Table, badge: batchCount > 0 ? batchCount : undefined },
    { id: 'about' as PageId, label: 'About the Model', icon: Info },
  ];

  const getStatusColor = () => {
    if (health.status === 'ok') return 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]';
    if (health.status === 'initializing') return 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.7)]';
    return 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.7)]';
  };

  const getStatusText = () => {
    if (health.status === 'ok') return `ONLINE (${health.device.toUpperCase()})`;
    if (health.status === 'initializing') return 'INITIALIZING...';
    return 'BACKEND OFFLINE';
  };

  return (
    <aside className="w-64 bg-surface border-r border-border flex flex-col h-screen shrink-0 select-none">
      {/* Brand Header */}
      <div className="p-6 border-b border-border text-center flex flex-col items-center">
        <div className="relative mb-3">
          <img
            src="/logo.jpg"
            alt="RareCam Logo Badge"
            className="w-20 h-20 rounded-full object-cover border-2 border-emerald-accent shadow-lg shadow-black/50"
            onError={(e) => {
              // fallback if image path fails
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-emerald-accent">RareCam</h1>
        <p className="text-xs text-muted mt-1 leading-snug font-normal max-w-[200px]">
          Active Learning for Camera-Trap Wildlife Monitoring
        </p>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-4 space-y-1.5" aria-label="Main Navigation">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id)}
              className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-hud text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-raised text-emerald-accent border border-emerald-accent/30 shadow-sm'
                  : 'text-text/80 hover:bg-raised/60 hover:text-text border border-transparent'
              }`}
              aria-current={isActive ? 'page' : undefined}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-accent' : 'text-muted'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge !== undefined && (
                <span className="hud-value-mono text-xs px-2 py-0.5 rounded-full bg-emerald-accent/20 text-emerald-accent border border-emerald-accent/40 font-semibold">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Backend Status Indicator */}
      <div className="p-4 border-t border-border bg-bg/50">
        <div className="flex items-center gap-2.5 px-3 py-2 rounded-hud bg-surface border border-border">
          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${getStatusColor()}`} />
          <div className="flex flex-col min-w-0">
            <span className="hud-label text-[10px]">SYSTEM STATUS</span>
            <span className="hud-value-mono text-xs text-text font-semibold truncate">
              {getStatusText()}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
