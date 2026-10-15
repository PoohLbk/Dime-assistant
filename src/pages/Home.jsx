import { useState, useEffect, useMemo } from "react";
import { TrendingUp, Activity, Newspaper, ScanLine, Sparkles } from "lucide-react";
import { base44 } from "@/api/base44Client";
import { Image } from "@/components/ui/image";
import ControlPanel from "@/components/ControlPanel";
import MetricsBar from "@/components/MetricsBar";
import TrendBanner from "@/components/TrendBanner";
import SignalPanel from "@/components/SignalPanel";
import TechnicalChart from "@/components/TechnicalChart";
import NewsSentiment from "@/components/NewsSentiment";
import MarketScreener from "@/components/MarketScreener";
import { generateSampleAnalysis } from "@/lib/sampleData";

const HERO_IMAGE = "https://media.base44.com/images/public/6acf42499ab131f374363007/5936a22c0_generated_b1095ffe.jpg";

const TABS = [
  { id: "technical", label: "Technical Analysis & Golden Zone", icon: Activity },
  { id: "news", label: "News Sentiment Analysis", icon: Newspaper },
  { id: "screener", label: "Auto Market Screener", icon: ScanLine },
];

export default function Home() {
  const [ticker, setTicker] = useState("PLTR");
  const [interval, setInterval] = useState("1d");
  const [period, setPeriod] = useState("6mo");
  const [showDemandSupply, setShowDemandSupply] = useState(true);
  const [activeTab, setActiveTab] = useState("technical");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    base44.functions
      .invoke("fetchStockData", { ticker, period, interval })
      .then((res) => {
        if (cancelled) return;
        const data = res.data;
        if (data.error) {
          setError(data.error);
          setAnalysis(generateSampleAnalysis(ticker, period, interval));
        } else {
          const df = data.df.map((d) => ({ ...d, date: new Date(d.date) }));
          setAnalysis({ df, results: data.results });
        }
      })
      .catch(() => {
        if (cancelled) return;
        setError("ไม่สามารถดึงข้อมูลได้ — แสดงข้อมูลตัวอย่างชั่วคราว");
        setAnalysis(generateSampleAnalysis(ticker, period, interval));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [ticker, period, interval]);

  const sampleFallback = useMemo(
    () => generateSampleAnalysis(ticker, period, interval),
    [ticker, period, interval]
  );
  const current = analysis || sampleFallback;

  return (
    <div className="min-h-screen bg-[#07080A] text-[#D9DEE3] font-body flex flex-col md:flex-row antialiased selection:bg-[#2ED88A]/30 selection:text-[#2ED88A]">
      <ControlPanel
        ticker={ticker}
        setTicker={setTicker}
        interval={interval}
        setInterval={setInterval}
        period={period}
        setPeriod={setPeriod}
        showDemandSupply={showDemandSupply}
        setShowDemandSupply={setShowDemandSupply}
      />

      <main className="relative flex-1 min-w-0">
        {/* Backdrop Glow & Hero Overlay */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-[320px] overflow-hidden z-0">
          <Image src={HERO_IMAGE} alt="" className="w-full h-full object-cover opacity-35 mix-blend-luminosity scale-105 filter blur-[1px]" />
          <div className="absolute inset-0 bg-gradient-to-b from-[#07080A]/20 via-[#07080A]/80 to-[#07080A]" />
          <div className="absolute -top-24 right-10 w-96 h-96 bg-[#2ED88A]/10 rounded-full blur-3xl pointer-events-none" />
        </div>

        <div className="relative z-[1] px-5 md:px-8 pt-7 pb-12 space-y-6">
          {/* Header Bar */}
          <header className="animate-in fade-in slide-in-from-top-3 duration-500 flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#1E232A]/80">
            <div className="flex items-center gap-3.5">
              <div className="w-11 h-11 shrink-0 rounded-xl border border-[#2ED88A]/40 bg-[#2ED88A]/[0.08] shadow-[0_0_25px_rgba(46,216,138,0.25)] flex items-center justify-center transition-transform hover:scale-105 duration-200">
                <TrendingUp className="w-6 h-6 text-[#2ED88A]" strokeWidth={2.5} />
              </div>
              <div>
                <h1 className="m-0 text-xl md:text-2xl font-black tracking-wide leading-tight text-[#F2F5F7] flex items-center gap-2">
                  EMERALD <span className="text-[#2ED88A] drop-shadow-[0_0_12px_rgba(46,216,138,0.4)]">ANALYTICS</span>
                </h1>
                <p className="m-0 mt-0.5 text-[10px] md:text-[11px] font-bold uppercase tracking-widest text-[#7A838D] flex items-center gap-2">
                  <span>Technical Analysis</span> • <span>Golden Zone</span> • <span>News Sentiment</span>
                </p>
              </div>
            </div>

            {loading && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#2ED88A]/30 bg-[#2ED88A]/10 text-xs font-semibold text-[#2ED88A] animate-pulse">
                <span className="w-2.5 h-2.5 rounded-full bg-[#2ED88A] animate-ping" />
                Fetching Real-time Market Data...
              </div>
            )}
          </header>

          {error && (
            <div className="flex items-center gap-2.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] px-4 py-3 text-xs font-medium text-[#FF8A8A] backdrop-blur-sm animate-in fade-in duration-300">
              <Sparkles className="w-4 h-4 shrink-0 text-[#FF5252]" />
              <span>{error}</span>
            </div>
          )}

          {/* Key Metrics Strip */}
          <MetricsBar key={ticker} res={current.results} />

          {/* Trend Banner Indicator */}
          <TrendBanner res={current.results} />

          {/* Signal Detector */}
          <SignalPanel res={current.results} ticker={ticker} interval={interval} />

          {/* Navigation Tabs */}
          <div className="pt-2">
            <div className="flex flex-wrap gap-2 border-b border-[#1E232A]">
              {TABS.map((t) => {
                const Icon = t.icon;
                const isActive = activeTab === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setActiveTab(t.id)}
                    className={`flex items-center gap-2 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-all duration-200 border-b-2 -mb-px rounded-t-lg ${
                      isActive
                        ? "text-[#2ED88A] border-[#2ED88A] bg-[#2ED88A]/[0.06]"
                        : "text-[#7A838D] border-transparent hover:text-[#D9DEE3] hover:bg-[#12161C]/50"
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? "text-[#2ED88A]" : "text-[#7A838D]"}`} />
                    <span>{t.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab Workspace */}
            <div className="mt-4 rounded-b-2xl border border-[#1E232A] bg-[#0C0F12]/90 p-4 md:p-6 shadow-2xl backdrop-blur-xl">
              {activeTab === "technical" && (
                <TechnicalChart df={current.df} res={current.results} showDemandSupply={showDemandSupply} />
              )}
              {activeTab === "news" && <NewsSentiment ticker={ticker} />}
              {activeTab === "screener" && <MarketScreener period={period} interval={interval} />}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
