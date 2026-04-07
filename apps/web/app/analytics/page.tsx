'use client';

import { useRouter } from 'next/navigation';

export default function AnalyticsPage() {
  const router = useRouter();

    return (
      <>
        <div className="antialiased min-h-screen selection:bg-brand-500/30 selection:text-brand-400 overflow-x-hidden relative">
          <div className="noise-overlay"></div>

    {/* Top Navigation Header */}
    <header className="fixed top-0 left-0 right-0 z-40 glass-panel border-b-0 border-slate-800/80">
        <div className="max-w-[90rem] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            {/* Brand */}
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400 to-indigo-600 flex items-center justify-center shadow-[0_0_15px_rgba(34,211,238,0.3)] relative overflow-hidden">
                    <i className="ph ph-scan text-white text-xl z-10"></i>
                    {/* Animated scan line in logo */}
                    <div className="absolute inset-0 bg-white/20 h-[2px] w-full animate-scan-line"></div>
                </div>
                <span className="font-bold text-lg tracking-tight text-white">Aura<span className="text-brand-400">Sense</span></span>
            </div>

            {/* Navigation Links */}
            <nav className="hidden md:flex items-center gap-1">
                <button onClick={() => router.push('/')} className="px-4 py-2 text-sm font-medium rounded-md text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-all">
                    Dashboard Overview
                </button>
                <button className="px-4 py-2 text-sm font-medium rounded-md bg-white/10 text-brand-400 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)] transition-all">
                    Analytics
                </button>
                <button onClick={() => router.push('/settings')} className="px-4 py-2 text-sm font-medium rounded-md text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-all">
                    Settings
                </button>
            </nav>

            {/* Right Actions */}
            <div className="flex items-center gap-4">
                <button className="text-slate-400 hover:text-white transition-colors relative">
                    <i className="ph ph-bell text-xl"></i>
                </button>
                <div className="h-5 w-px bg-slate-800"></div>
                <button className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                    <img src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80" alt="Admin" className="w-8 h-8 rounded-full border border-slate-700" />
                </button>
            </div>
        </div>
    </header>

    {/* Main Content */}
    <main className="max-w-[90rem] mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12 space-y-6">
        
        {/* Hero Section & Filters */}
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
            <div>
                <div className="flex items-center gap-2 mb-1">
                    <i className="ph ph-chart-line-up text-brand-500"></i>
                    <span className="text-sm font-medium text-brand-500 tracking-wide uppercase">Historical Data & Insights</span>
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">Attendance Analytics</h1>
                <p className="text-slate-400 mt-2 max-w-2xl">Analyze facial recognition trends, identify anomalies, and export compliance reports.</p>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
                {/* Filters */}
                <div className="flex items-center p-1 rounded-lg bg-slate-900/80 border border-slate-800 backdrop-blur-md">
                    <button className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium bg-slate-800 text-white shadow-sm border border-slate-700">
                        <i className="ph ph-calendar-blank"></i> Last 30 Days
                    </button>
                    <button className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors">
                        <i className="ph ph-student"></i> All Classes
                    </button>
                    <button className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors">
                        <i className="ph ph-map-pin"></i> All Zones
                    </button>
                </div>

                {/* Export CTA */}
                <button className="flex items-center gap-2 px-4 py-2.5 bg-brand-500 hover:bg-brand-400 text-slate-950 rounded-lg text-sm font-semibold transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_25px_rgba(6,182,212,0.5)] active:scale-95 group">
                    <i className="ph ph-download-simple text-lg group-hover:-translate-y-0.5 transition-transform"></i>
                    Export Report
                    <span className="border-l border-slate-950/20 pl-2 ml-1 text-xs opacity-80 font-normal">CSV / PDF</span>
                </button>
            </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
            {/* Metric 1 */}
            <div className="glass-panel rounded-2xl p-5 border-l-2 border-l-brand-500 animate-fade-in-up">
                <div className="flex justify-between items-start mb-2">
                    <p className="text-sm font-medium text-slate-400">Avg. Attendance Rate</p>
                    <div className="w-8 h-8 rounded-full bg-brand-500/10 flex items-center justify-center">
                        <i className="ph ph-users text-brand-400 text-lg"></i>
                    </div>
                </div>
                <div className="flex items-baseline gap-2">
                    <h3 className="text-3xl font-bold text-white">94.2%</h3>
                    <span className="flex items-center text-xs font-medium text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                        <i className="ph ph-arrow-up-right mr-1"></i> 2.1%
                    </span>
                </div>
                <p className="text-xs text-slate-500 mt-2">vs. previous 30 days</p>
            </div>

            {/* Metric 2 */}
            <div className="glass-panel rounded-2xl p-5 border-l-2 border-l-indigo-500 animate-fade-in-up delay-100">
                <div className="flex justify-between items-start mb-2">
                    <p className="text-sm font-medium text-slate-400">Total Valid Scans</p>
                    <div className="w-8 h-8 rounded-full bg-indigo-500/10 flex items-center justify-center">
                        <i className="ph ph-scan text-indigo-400 text-lg"></i>
                    </div>
                </div>
                <div className="flex items-baseline gap-2">
                    <h3 className="text-3xl font-bold text-white">142.5k</h3>
                    <span className="flex items-center text-xs font-medium text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                        <i className="ph ph-arrow-up-right mr-1"></i> 8.4%
                    </span>
                </div>
                <p className="text-xs text-slate-500 mt-2">Unique successful verifications</p>
            </div>

            {/* Metric 3 */}
            <div className="glass-panel rounded-2xl p-5 border-l-2 border-l-emerald-500 animate-fade-in-up delay-200">
                <div className="flex justify-between items-start mb-2">
                    <p className="text-sm font-medium text-slate-400">Overall AI Confidence</p>
                    <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center">
                        <i className="ph ph-bounding-box text-emerald-400 text-lg"></i>
                    </div>
                </div>
                <div className="flex items-baseline gap-2">
                    <h3 className="text-3xl font-bold text-white">97.8%</h3>
                    <span className="flex items-center text-xs font-medium text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                        <i className="ph ph-minus mr-1"></i> 0.0%
                    </span>
                </div>
                <p className="text-xs text-slate-500 mt-2">Stable processing threshold</p>
            </div>

            {/* Metric 4 */}
            <div className="glass-panel rounded-2xl p-5 border-l-2 border-l-red-500 animate-fade-in-up delay-300">
                <div className="flex justify-between items-start mb-2">
                    <p className="text-sm font-medium text-slate-400">Detection Anomaly Rate</p>
                    <div className="w-8 h-8 rounded-full bg-red-500/10 flex items-center justify-center">
                        <i className="ph ph-warning-circle text-red-400 text-lg"></i>
                    </div>
                </div>
                <div className="flex items-baseline gap-2">
                    <h3 className="text-3xl font-bold text-white">1.2%</h3>
                    <span className="flex items-center text-xs font-medium text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                        <i className="ph ph-arrow-down-right mr-1"></i> 0.3%
                    </span>
                </div>
                <p className="text-xs text-slate-500 mt-2">Spoofs, misalignments, failures</p>
            </div>
        </div>

        {/* Main Chart: Trends Over Time */}
        <div className="glass-panel rounded-2xl border border-slate-800 p-6">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center mb-8 gap-4">
                <div>
                    <h2 className="text-lg font-semibold text-white">Attendance Trends Overview</h2>
                    <p className="text-sm text-slate-400">Daily verification success vs. anomalies over the selected period.</p>
                </div>
                <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-brand-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]"></div>
                        <span className="text-slate-300">Successful Verifications</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                        <span className="text-slate-500">Anomalies</span>
                    </div>
                </div>
            </div>

            {/* SVG Area Chart Visualization */}
            <div className="w-full h-[300px] relative">
                {/* Y-Axis Labels */}
                <div className="absolute left-0 top-0 bottom-8 w-10 flex flex-col justify-between text-xs text-slate-500 z-10">
                    <span>100%</span>
                    <span>75%</span>
                    <span>50%</span>
                    <span>25%</span>
                    <span>0%</span>
                </div>
                {/* Grid lines */}
                <div className="absolute left-10 right-0 top-0 bottom-8 flex flex-col justify-between z-0">
                    <div className="w-full h-px bg-slate-800/50"></div>
                    <div className="w-full h-px bg-slate-800/50"></div>
                    <div className="w-full h-px bg-slate-800/50"></div>
                    <div className="w-full h-px bg-slate-800/50"></div>
                    <div className="w-full h-px bg-slate-800"></div>
                </div>

                {/* SVG Graph container */}
                <div className="absolute left-10 right-0 top-2 bottom-8 z-10">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full overflow-visible">
                        {/* Definitions for Gradients */}
                        <defs>
                            <linearGradient id="brandGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
                            </linearGradient>
                            <linearGradient id="slateGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#334155" stopOpacity="0.2" />
                                <stop offset="100%" stopColor="#334155" stopOpacity="0" />
                            </linearGradient>
                        </defs>

                        {/* Anomalies Area & Line (Bottom layer) */}
                        <path d="M0,95 L10,92 L20,96 L30,90 L40,94 L50,88 L60,95 L70,85 L80,93 L90,90 L100,94 L100,100 L0,100 Z" fill="url(#slateGradient)" />
                        <path d="M0,95 L10,92 L20,96 L30,90 L40,94 L50,88 L60,95 L70,85 L80,93 L90,90 L100,94" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="2,2" />

                        {/* Success Area & Line (Top layer) */}
                        {/* The points match a roughly ~90-98% range */}
                        <path d="M0,15 L10,8 L20,12 L30,5 L40,10 L50,18 L60,8 L70,12 L80,4 L90,8 L100,10 L100,100 L0,100 Z" fill="url(#brandGradient)" />
                        <path d="M0,15 L10,8 L20,12 L30,5 L40,10 L50,18 L60,8 L70,12 L80,4 L90,8 L100,10" fill="none" stroke="#06b6d4" strokeWidth="2" style={{ filter: 'drop-shadow(0 0 4px rgba(6,182,212,0.5))' }} />
                        
                        {/* Hover points (mockup) */}
                        <circle cx="80" cy="4" r="3" fill="#020617" stroke="#06b6d4" strokeWidth="2" className="cursor-pointer" />
                    </svg>
                    
                    {/* Tooltip mockup */}
                    <div className="absolute right-[15%] top-[-10px] bg-slate-900 border border-slate-700 px-3 py-2 rounded-lg shadow-xl text-xs flex flex-col gap-1 z-20">
                        <span className="text-slate-400 font-medium">Oct 18, 2023</span>
                        <div className="flex items-center gap-2 text-brand-400">
                            <i className="ph-fill ph-check-circle"></i> 97.4% Success
                        </div>
                    </div>
                </div>

                {/* X-Axis Labels */}
                <div className="absolute left-10 right-0 bottom-0 h-8 flex justify-between items-end text-xs text-slate-500 z-10">
                    <span>Oct 01</span>
                    <span>Oct 05</span>
                    <span>Oct 10</span>
                    <span>Oct 15</span>
                    <span>Oct 20</span>
                    <span>Oct 25</span>
                    <span>Today</span>
                </div>
            </div>
        </div>

        {/* Secondary Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Confidence Score Distribution */}
            <div className="glass-panel rounded-2xl border border-slate-800 p-6 flex flex-col">
                <div className="mb-6">
                    <h2 className="text-lg font-semibold text-white">Confidence Score Distribution</h2>
                    <p className="text-sm text-slate-400">Breakdown of AI certainty across all successful scans.</p>
                </div>
                
                <div className="flex-1 flex items-end gap-2 mt-4 pb-6 border-b border-slate-800 relative h-64">
                    {/* Y-Axis */}
                    <div className="absolute left-0 top-0 bottom-6 w-8 flex flex-col justify-between text-[10px] text-slate-500">
                        <span>40k</span>
                        <span>20k</span>
                        <span>0</span>
                    </div>
                    {/* Bars */}
                    <div className="flex-1 flex items-end justify-between ml-10 h-full w-full">
                        {/* <85% Bar */}
                        <div className="w-1/5 flex flex-col items-center group relative h-full justify-end">
                            <div className="w-full max-w-[40px] bg-slate-800 group-hover:bg-slate-700 rounded-t-sm transition-colors" style={{ height: '12%' }}></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-400 font-mono">< 85%</span>
                        </div>
                        {/* 85-90% Bar */}
                        <div className="w-1/5 flex flex-col items-center group relative h-full justify-end">
                            <div className="w-full max-w-[40px] bg-indigo-500/40 group-hover:bg-indigo-500/60 rounded-t-sm transition-colors" style={{ height: '35%' }}></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-400 font-mono">85-90%</span>
                        </div>
                        {/* 90-95% Bar */}
                        <div className="w-1/5 flex flex-col items-center group relative h-full justify-end">
                            <div className="w-full max-w-[40px] bg-brand-500/60 group-hover:bg-brand-500/80 rounded-t-sm transition-colors" style={{ height: '75%' }}></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-400 font-mono">90-95%</span>
                        </div>
                        {/* 95-99% Bar */}
                        <div className="w-1/5 flex flex-col items-center group relative h-full justify-end">
                            <div className="w-full max-w-[40px] bg-brand-400 rounded-t-sm shadow-[0_0_15px_rgba(34,211,238,0.3)] transition-all" style={{ height: '95%' }}></div>
                            <div className="absolute -top-8 bg-slate-800 text-white text-xs px-2 py-1 rounded border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity">Most Scans</div>
                            <span className="absolute -bottom-6 text-[10px] text-brand-400 font-mono font-medium">95-99%</span>
                        </div>
                        {/* >99% Bar */}
                        <div className="w-1/5 flex flex-col items-center group relative h-full justify-end">
                            <div className="w-full max-w-[40px] bg-emerald-400/80 group-hover:bg-emerald-400 rounded-t-sm transition-colors" style={{ height: '40%' }}></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-400 font-mono">> 99%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Top Classes by Attendance */}
            <div className="glass-panel rounded-2xl border border-slate-800 p-6 flex flex-col">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-lg font-semibold text-white">Top Classes by Rate</h2>
                        <p className="text-sm text-slate-400">Highest consistent attendance.</p>
                    </div>
                    <button className="text-sm text-brand-400 hover:text-white transition-colors">View All</button>
                </div>
                
                <div className="flex-1 space-y-5 flex flex-col justify-center">
                    
                    {/* Class Row */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-sm font-medium text-white flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-brand-500"></span> CS-401 (Advanced AI)
                            </span>
                            <span className="text-sm font-mono text-brand-400">98.5%</span>
                        </div>
                        <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div className="bg-gradient-to-r from-brand-600 to-brand-400 h-full rounded-full" style={{ width: '98.5%' }}></div>
                        </div>
                    </div>

                    {/* Class Row */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-sm font-medium text-slate-200 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-indigo-500"></span> BIO-204 (Genetics Lab)
                            </span>
                            <span className="text-sm font-mono text-slate-300">96.2%</span>
                        </div>
                        <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div className="bg-indigo-500 h-full rounded-full" style={{ width: '96.2%' }}></div>
                        </div>
                    </div>

                    {/* Class Row */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-sm font-medium text-slate-300 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-emerald-500"></span> MAT-302 (Calculus III)
                            </span>
                            <span className="text-sm font-mono text-slate-300">94.8%</span>
                        </div>
                        <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div className="bg-emerald-500 h-full rounded-full" style={{ width: '94.8%' }}></div>
                        </div>
                    </div>

                    {/* Class Row */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-sm font-medium text-slate-400 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-amber-500"></span> ENG-101 (Literature)
                            </span>
                            <span className="text-sm font-mono text-slate-400">91.4%</span>
                        </div>
                        <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div className="bg-amber-500 h-full rounded-full opacity-80" style={{ width: '91.4%' }}></div>
                        </div>
                    </div>

                    {/* Class Row */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-sm font-medium text-slate-400 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-slate-600"></span> PHY-101 (Intro Physics)
                            </span>
                            <span className="text-sm font-mono text-slate-400">88.7%</span>
                        </div>
                        <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div className="bg-slate-600 h-full rounded-full" style={{ width: '88.7%' }}></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {/* Detected Anomalies Pattern Summary */}
        <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-lg shadow-black/20 mt-6">
            <div className="px-6 py-5 border-b border-slate-800 bg-slate-900/40 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <div className="flex items-center gap-2">
                        <h2 className="text-lg font-semibold text-white">Pattern Anomalies Detected</h2>
                        <span className="flex h-2 w-2 relative">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                        </span>
                    </div>
                    <p className="text-sm text-slate-400 mt-1">Aggregated issues requiring administrative review.</p>
                </div>
            </div>

            <div className="p-0">
                {/* Anomaly Item 1 */}
                <div className="p-5 border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center flex-shrink-0 mt-1 sm:mt-0">
                            <i className="ph ph-mask-happy text-xl text-red-400"></i>
                        </div>
                        <div>
                            <h4 className="text-sm font-medium text-white">Repeated Spoofing Attempts</h4>
                            <p className="text-xs text-slate-400 mt-1 max-w-xl">Multiple "2D Image Detected" flags associated with Camera 04 (North Gate) during morning rush (08:30 - 09:00 AM).</p>
                            <div className="flex items-center gap-2 mt-2 text-[10px]">
                                <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">14 incidents</span>
                                <span className="text-slate-500">Last 7 days</span>
                            </div>
                        </div>
                    </div>
                    <button className="text-sm text-slate-300 hover:text-white px-4 py-2 border border-slate-700 rounded-lg bg-slate-900/50 hover:bg-slate-800 transition-colors self-start sm:self-center flex-shrink-0">
                        Investigate Logs
                    </button>
                </div>

                {/* Anomaly Item 2 */}
                <div className="p-5 border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center flex-shrink-0 mt-1 sm:mt-0">
                            <i className="ph ph-faders-horizontal text-xl text-amber-400"></i>
                        </div>
                        <div>
                            <h4 className="text-sm font-medium text-white">Consistent Low Confidence Zone</h4>
                            <p className="text-xs text-slate-400 mt-1 max-w-xl">Camera 02 in the Main Auditorium consistently returns confidence scores below 85% for users wearing glasses.</p>
                            <div className="flex items-center gap-2 mt-2 text-[10px]">
                                <span className="bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded border border-amber-500/30">Hardware/Lighting Check Recommended</span>
                            </div>
                        </div>
                    </div>
                    <button className="text-sm text-slate-300 hover:text-white px-4 py-2 border border-slate-700 rounded-lg bg-slate-900/50 hover:bg-slate-800 transition-colors self-start sm:self-center flex-shrink-0">
                        View Analytics
                    </button>
                </div>

                {/* Anomaly Item 3 */}
                <div className="p-5 hover:bg-slate-800/20 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0 mt-1 sm:mt-0">
                            <i className="ph ph-map-trifold text-xl text-indigo-400"></i>
                        </div>
                        <div>
                            <h4 className="text-sm font-medium text-white">Location Misalignment (Class Cutting)</h4>
                            <p className="text-xs text-slate-400 mt-1 max-w-xl">Detected 24 students scheduled for mandatory Lab sessions scanning into the Cafeteria zone simultaneously.</p>
                            <div className="flex items-center gap-2 mt-2 text-[10px]">
                                <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">Affected Class: CHM-201</span>
                            </div>
                        </div>
                    </div>
                    <button className="text-sm text-slate-300 hover:text-white px-4 py-2 border border-slate-700 rounded-lg bg-slate-900/50 hover:bg-slate-800 transition-colors self-start sm:self-center flex-shrink-0">
                        Export Student List
                    </button>
                </div>
            </div>
        </div>

    </main>

    {/* Footer */}
    <footer className="border-t border-slate-800/50 bg-slate-950/80 backdrop-blur-sm mt-12 py-8">
        <div className="max-w-[90rem] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-slate-500 text-sm">
                <i className="ph ph-shield-check text-brand-500 text-lg"></i>
                <span>AuraSense Attendance System v2.4.1</span>
            </div>
            <div className="flex items-center gap-6 text-sm">
                <a href="#" className="text-slate-400 hover:text-white transition-colors">Documentation & Help</a>
                <a href="#" className="text-slate-400 hover:text-white transition-colors">Contact Support</a>
                <a href="#" className="text-slate-400 hover:text-white transition-colors">Privacy Policy</a>
            </div>
        </div>
    </footer>
        </div>
      </>
    );
}
