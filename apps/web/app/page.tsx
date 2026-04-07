'use client';

import { useRouter } from 'next/navigation';

export default function HomePage() {
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
                <button className="px-4 py-2 text-sm font-medium rounded-md bg-white/10 text-brand-400 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)] transition-all">
                    Dashboard Overview
                </button>
                <button onClick={() => router.push('/analytics')} className="px-4 py-2 text-sm font-medium rounded-md text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-all">
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
                    <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border border-slate-950"></span>
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
        
        {/* Hero Section */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div>
                <div className="flex items-center gap-3 mb-1">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                    </span>
                    <span className="text-sm font-medium text-emerald-500 tracking-wide uppercase">System Active • Live Monitoring</span>
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">Campus Attendance</h1>
                <p className="text-slate-400 mt-2 max-w-2xl">Real-time facial recognition tracking across all designated zones. Showing data for today.</p>
            </div>
            <div className="flex items-center gap-3">
                <button className="flex items-center gap-2 px-4 py-2.5 glass-panel rounded-lg text-sm font-medium text-slate-200 hover:bg-slate-800/80 transition-all border border-slate-700">
                    <i className="ph ph-calendar-blank text-lg"></i>
                    Today, Oct 24
                    <i className="ph ph-caret-down text-slate-400"></i>
                </button>
                <button onClick={() => router.push('/settings')} className="flex items-center gap-2 px-4 py-2.5 bg-brand-500 hover:bg-brand-400 text-slate-950 rounded-lg text-sm font-semibold transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_25px_rgba(6,182,212,0.5)] active:scale-95">
                    <i className="ph ph-sliders-horizontal text-lg"></i>
                    Configure Settings
                </button>
            </div>
        </div>

        {/* Stats Summary Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Stat 1: Total Attendance */}
            <div className="glass-panel rounded-2xl p-5 hover:border-slate-700 transition-colors relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <i className="ph-fill ph-users text-6xl text-brand-400"></i>
                </div>
                <div className="flex items-center justify-between mb-4 relative z-10">
                    <h3 className="text-sm font-medium text-slate-400">Total Present</h3>
                    <span className="px-2 py-1 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 flex items-center gap-1 border border-emerald-500/20">
                        <i className="ph ph-trend-up"></i> 12%
                    </span>
                </div>
                <div className="relative z-10">
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-white">1,248</span>
                        <span className="text-sm text-slate-500">/ 1,350</span>
                    </div>
                    {/* Progress bar */}
                    <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
                        <div className="bg-gradient-to-r from-brand-600 to-brand-400 h-1.5 rounded-full" style={{ width: '92%' }}></div>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">92.4% overall attendance rate</p>
                </div>
            </div>

            {/* Stat 2: Avg Confidence */}
            <div className="glass-panel rounded-2xl p-5 hover:border-slate-700 transition-colors relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <i className="ph-fill ph-bounding-box text-6xl text-indigo-400"></i>
                </div>
                <div className="flex items-center justify-between mb-4 relative z-10">
                    <h3 className="text-sm font-medium text-slate-400">Avg. Match Confidence</h3>
                    <span className="px-2 py-1 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 flex items-center gap-1 border border-emerald-500/20">
                        High
                    </span>
                </div>
                <div className="relative z-10">
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-white">98.6%</span>
                    </div>
                    {/* Distribution visual */}
                    <div className="flex w-full h-1.5 mt-4 rounded-full overflow-hidden gap-0.5">
                        <div className="bg-emerald-500 h-full" style={{ width: '85%' }} title="> 95% Confidence"></div>
                        <div className="bg-yellow-500 h-full" style={{ width: '12%' }} title="85% - 95% Confidence"></div>
                        <div className="bg-red-500 h-full" style={{ width: '3%' }} title="< 85% Confidence"></div>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">Target threshold: >90%</p>
                </div>
            </div>

            {/* Stat 3: Class Breakdown Summary */}
            <div className="glass-panel rounded-2xl p-5 hover:border-slate-700 transition-colors relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <i className="ph-fill ph-chalkboard-teacher text-6xl text-amber-400"></i>
                </div>
                <div className="flex items-center justify-between mb-4 relative z-10">
                    <h3 className="text-sm font-medium text-slate-400">Active Classes</h3>
                    <span className="px-2 py-1 rounded text-xs font-medium bg-slate-800 text-slate-300 flex items-center gap-1 border border-slate-700">
                        Live Now
                    </span>
                </div>
                <div className="relative z-10">
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-white">42</span>
                        <span className="text-sm text-slate-500">sessions</span>
                    </div>
                    <div className="flex items-center gap-3 mt-4">
                        <div className="flex -space-x-2">
                            <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center text-[10px] text-indigo-300">CS</div>
                            <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-[10px] text-emerald-300">BIO</div>
                            <div className="w-6 h-6 rounded-full bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-[10px] text-amber-300">ENG</div>
                        </div>
                        <span className="text-xs text-slate-400">Scanning in 14 zones</span>
                    </div>
                </div>
            </div>

            {/* Stat 4: Alerts/Anomalies */}
            <div className="glass-panel rounded-2xl p-5 hover:border-red-900/50 transition-colors relative overflow-hidden group">
                <div className="absolute inset-0 bg-red-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <i className="ph-fill ph-warning-octagon text-6xl text-red-500"></i>
                </div>
                <div className="flex items-center justify-between mb-4 relative z-10">
                    <h3 className="text-sm font-medium text-slate-400">Pending Alerts</h3>
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                    </span>
                </div>
                <div className="relative z-10">
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-white">7</span>
                        <span className="text-sm text-slate-500">requires review</span>
                    </div>
                    <div className="flex flex-col gap-1 mt-3">
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-slate-400">Low Confidence (< 80%)</span>
                            <span className="text-red-400 font-medium">4</span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-slate-400">Spoofing Suspected</span>
                            <span className="text-amber-400 font-medium">3</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {/* Middle Section: Analytics Chart & Notifications */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Attendance Trends Chart (Visual Mock) */}
            <div className="lg:col-span-2 glass-panel rounded-2xl border border-slate-800 p-6 flex flex-col h-[400px]">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-lg font-semibold text-white">Arrival Distribution</h2>
                        <p className="text-sm text-slate-400">Check-in volume over time (08:00 AM - 12:00 PM)</p>
                    </div>
                    <button onClick={() => router.push('/analytics')} className="text-sm text-brand-400 hover:text-brand-300 font-medium flex items-center gap-1 transition-colors">
                        Full Analytics <i className="ph ph-arrow-right"></i>
                    </button>
                </div>
                
                {/* CSS Based Bar Chart Visualization */}
                <div className="flex-1 flex items-end gap-2 sm:gap-4 mt-4 pb-6 border-b border-slate-800 relative">
                    {/* Y-Axis labels */}
                    <div className="absolute left-0 top-0 bottom-6 w-8 flex flex-col justify-between text-[10px] text-slate-500">
                        <span>300</span>
                        <span>200</span>
                        <span>100</span>
                        <span>0</span>
                    </div>
                    {/* Chart Bars Container offset for Y-axis */}
                    <div className="flex-1 flex items-end gap-2 sm:gap-4 ml-8 h-full pt-4">
                        {/* Bar Items (Heights simulated via inline style) */}
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '10%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-500">8AM</span>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '45%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '85%' }}>
                            <div className="w-full bg-gradient-to-t from-brand-600 to-brand-400 rounded-t-sm h-full shadow-[0_0_15px_rgba(34,211,238,0.2)]"></div>
                            {/* Tooltip */}
                            <div className="absolute -top-10 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 border border-slate-700 text-xs py-1 px-2 rounded pointer-events-none whitespace-nowrap z-10">240 Scans</div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-300 font-medium">9AM</span>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '60%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '30%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-500">10AM</span>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '15%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '25%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-500">11AM</span>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '10%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                        </div>
                        <div className="w-full flex flex-col items-center group relative cursor-pointer" style={{ height: '5%' }}>
                            <div className="w-full bg-slate-800 hover:bg-slate-700 rounded-t-sm h-full transition-colors relative"></div>
                            <span className="absolute -bottom-6 text-[10px] text-slate-500">12PM</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Discrepancy Notifications */}
            <div className="glass-panel rounded-2xl border border-slate-800 flex flex-col h-[400px]">
                <div className="p-6 border-b border-slate-800 flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-semibold text-white">Live Discrepancies</h2>
                        <p className="text-sm text-slate-400">Needs manual verification</p>
                    </div>
                    <div className="bg-red-500/10 text-red-400 text-xs font-bold px-2 py-1 rounded border border-red-500/20">
                        7 New
                    </div>
                </div>
                
                <div className="flex-1 overflow-y-auto no-scrollbar p-2">
                    <div className="space-y-1">
                        {/* Alert Item 1 */}
                        <div className="p-3 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group flex gap-3 items-start border border-transparent hover:border-slate-700/50">
                            <div className="relative w-10 h-10 rounded overflow-hidden bg-slate-800 flex-shrink-0 border border-slate-700">
                                <img alt="" src="https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover opacity-60 filter grayscale group-hover:grayscale-0 transition-all" />
                                <div className="absolute inset-0 ring-1 ring-inset ring-red-500/50"></div>
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-medium text-white truncate">Unknown Match</p>
                                    <span className="text-[10px] text-slate-500">Just now</span>
                                </div>
                                <p className="text-xs text-slate-400 mt-0.5 truncate">CS-101 (Camera 3)</p>
                                <div className="flex items-center gap-2 mt-1.5">
                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">Low Conf: 42%</span>
                                    <button className="text-[10px] text-slate-300 hover:text-white underline decoration-slate-500 underline-offset-2">Review</button>
                                </div>
                            </div>
                        </div>

                        {/* Alert Item 2 */}
                        <div className="p-3 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group flex gap-3 items-start border border-transparent hover:border-slate-700/50">
                            <div className="relative w-10 h-10 rounded overflow-hidden bg-slate-800 flex-shrink-0 border border-slate-700">
                                <img alt="" src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-medium text-white truncate">Elena Rostova</p>
                                    <span className="text-[10px] text-slate-500">5m ago</span>
                                </div>
                                <p className="text-xs text-slate-400 mt-0.5 truncate">Lab Zone B</p>
                                <div className="flex items-center gap-2 mt-1.5">
                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">Mask Detected</span>
                                    <span className="text-[10px] text-slate-500">Conf: 78%</span>
                                </div>
                            </div>
                        </div>

                        {/* Alert Item 3 */}
                        <div className="p-3 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group flex gap-3 items-start border border-transparent hover:border-slate-700/50">
                            <div className="relative w-10 h-10 rounded overflow-hidden bg-slate-800 flex-shrink-0 border border-slate-700 flex items-center justify-center">
                                <i className="ph ph-user-circle-minus text-2xl text-slate-500"></i>
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-medium text-white truncate">Spoofing Attempt?</p>
                                    <span className="text-[10px] text-slate-500">12m ago</span>
                                </div>
                                <p className="text-xs text-slate-400 mt-0.5 truncate">Main Entrance (Cam 1)</p>
                                <div className="flex items-center gap-2 mt-1.5">
                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">2D Image Detected</span>
                                    <button className="text-[10px] text-slate-300 hover:text-white underline decoration-slate-500 underline-offset-2">Review Video</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="p-3 border-t border-slate-800 bg-slate-900/50 backdrop-blur text-center">
                    <button className="text-xs text-slate-400 hover:text-white transition-colors">View All Notifications</button>
                </div>
            </div>
        </div>

        {/* Detailed Attendance Table */}
        <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden flex flex-col shadow-lg shadow-black/20 mt-6">
            
            {/* Table Header / Controls */}
            <div className="p-5 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/40">
                <div>
                    <h2 className="text-lg font-semibold text-white">Live Roster Feed</h2>
                    <p className="text-sm text-slate-400">Streaming recent recognized faces.</p>
                </div>
                
                <div className="flex items-center gap-3">
                    {/* Search Input */}
                    <div className="relative">
                        <i className="ph ph-magnifying-glass absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"></i>
                        <input type="text" placeholder="Search student or ID..." className="w-full sm:w-64 bg-slate-950/50 border border-slate-700 rounded-lg py-2 pl-9 pr-4 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-all" />
                    </div>
                    {/* Filter Button */}
                    <button className="px-3 py-2 border border-slate-700 bg-slate-950/50 rounded-lg text-slate-300 hover:bg-slate-800 hover:text-white transition-all flex items-center gap-2 text-sm">
                        <i className="ph ph-faders"></i>
                        <span className="hidden sm:inline">Filters</span>
                    </button>
                </div>
            </div>

            {/* Table Container */}
            <div className="overflow-x-auto">
                <table className="w-full text-left whitespace-nowrap text-sm border-collapse">
                    <thead className="bg-slate-950/50 text-slate-400 border-b border-slate-800 text-xs uppercase tracking-wider font-semibold">
                        <tr>
                            <th className="px-6 py-4">Student Info</th>
                            <th className="px-6 py-4">Scan Time & Location</th>
                            <th className="px-6 py-4">Target Class</th>
                            <th className="px-6 py-4 w-48">AI Confidence</th>
                            <th className="px-6 py-4 text-right">Status</th>
                            <th className="px-6 py-4 w-12"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                        
                        {/* Row 1 (High Confidence) */}
                        <tr className="hover:bg-slate-800/30 transition-colors group">
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                    <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-slate-800 border border-slate-700">
                                        <img alt="" src="https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover" />
                                        {/* Bounding box overlay mock */}
                                        <div className="absolute inset-[4px] border border-brand-400/50 rounded-sm"></div>
                                    </div>
                                    <div>
                                        <div className="font-medium text-white">Marcus Johnson</div>
                                        <div className="text-xs text-slate-500 font-mono mt-0.5">ID: STD-8492</div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="text-slate-300">09:14:22 AM</div>
                                <div className="text-xs text-slate-500 mt-0.5">Physics Lab / Cam 02</div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
                                    PHY-201
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-brand-400 font-mono text-xs w-8 text-right">99.2%</span>
                                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-brand-500 shadow-[0_0_10px_rgba(6,182,212,0.8)]" style={{ width: '99.2%' }}></div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                    <i className="ph-fill ph-check-circle"></i> Present
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right text-slate-500 group-hover:text-white transition-colors cursor-pointer">
                                <i className="ph ph-dots-three text-xl"></i>
                            </td>
                        </tr>

                        {/* Row 2 (Medium Confidence) */}
                        <tr className="hover:bg-slate-800/30 transition-colors group">
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                    <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-slate-800 border border-slate-700">
                                        <img alt="" src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover" />
                                        <div className="absolute inset-[4px] border border-amber-400/50 rounded-sm"></div>
                                    </div>
                                    <div>
                                        <div className="font-medium text-white">Sarah Chen</div>
                                        <div className="text-xs text-slate-500 font-mono mt-0.5">ID: STD-7311</div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="text-slate-300">09:12:05 AM</div>
                                <div className="text-xs text-slate-500 mt-0.5">Main Hall / Entrance</div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
                                    BIO-101
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-amber-400 font-mono text-xs w-8 text-right">86.4%</span>
                                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]" style={{ width: '86.4%' }}></div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                    <i className="ph-fill ph-check-circle"></i> Present
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right text-slate-500 group-hover:text-white transition-colors cursor-pointer">
                                <i className="ph ph-dots-three text-xl"></i>
                            </td>
                        </tr>

                        {/* Row 3 (High Confidence) */}
                        <tr className="hover:bg-slate-800/30 transition-colors group">
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                    <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-slate-800 border border-slate-700">
                                        <img alt="" src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover" />
                                        <div className="absolute inset-[4px] border border-brand-400/50 rounded-sm"></div>
                                    </div>
                                    <div>
                                        <div className="font-medium text-white">David Miller</div>
                                        <div className="text-xs text-slate-500 font-mono mt-0.5">ID: STD-9023</div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="text-slate-300">09:10:44 AM</div>
                                <div className="text-xs text-slate-500 mt-0.5">Room 402 / Cam 01</div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400"></div>
                                    LIT-305
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-brand-400 font-mono text-xs w-8 text-right">98.9%</span>
                                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-brand-500 shadow-[0_0_10px_rgba(6,182,212,0.8)]" style={{ width: '98.9%' }}></div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                    <i className="ph-fill ph-check-circle"></i> Present
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right text-slate-500 group-hover:text-white transition-colors cursor-pointer">
                                <i className="ph ph-dots-three text-xl"></i>
                            </td>
                        </tr>

                        {/* Row 4 (Wrong Class/Location warning) */}
                        <tr className="hover:bg-slate-800/30 transition-colors group bg-red-950/10">
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                    <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-slate-800 border border-slate-700">
                                        <img alt="" src="https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=100&h=100&fit=crop&q=80" className="w-full h-full object-cover" />
                                        <div className="absolute inset-[4px] border border-brand-400/50 rounded-sm"></div>
                                    </div>
                                    <div>
                                        <div className="font-medium text-white">James Wilson</div>
                                        <div className="text-xs text-slate-500 font-mono mt-0.5">ID: STD-6442</div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="text-slate-300">09:05:12 AM</div>
                                <div className="text-xs text-red-400 mt-0.5">Gymnasium (Wrong Zone)</div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700 opacity-50">
                                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
                                    PHY-201
                                </div>
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-brand-400 font-mono text-xs w-8 text-right">97.1%</span>
                                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-brand-500 shadow-[0_0_10px_rgba(6,182,212,0.8)]" style={{ width: '97.1%' }}></div>
                                    </div>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20" title="Recognized, but not scheduled for this location">
                                    <i className="ph-fill ph-warning"></i> Misaligned
                                </span>
                            </td>
                            <td className="px-6 py-4 text-right text-slate-500 group-hover:text-white transition-colors cursor-pointer">
                                <i className="ph ph-dots-three text-xl"></i>
                            </td>
                        </tr>

                    </tbody>
                </table>
            </div>
            
            {/* Pagination */}
            <div className="px-6 py-4 border-t border-slate-800 bg-slate-900/40 flex items-center justify-between text-sm">
                <div className="text-slate-500">
                    Showing <span className="text-white font-medium">1</span> to <span className="text-white font-medium">4</span> of <span className="text-white font-medium">1,248</span> results
                </div>
                <div className="flex gap-1">
                    <button className="px-3 py-1.5 border border-slate-700 bg-slate-800/50 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 transition-colors disabled:opacity-50" disabled>Previous</button>
                    <button className="px-3 py-1.5 border border-slate-700 bg-slate-800/50 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 transition-colors">Next</button>
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
