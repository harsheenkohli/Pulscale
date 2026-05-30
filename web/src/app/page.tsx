"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
  Cell,
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
} from "recharts";
import {
  Activity,
  Heart,
  Moon,
  Sun,
  TrendingUp,
  Upload,
  PenLine,
  Github,
  FileText,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Menu,
  X,
  Zap,
} from "lucide-react";
import Image from "next/image";

const API = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type Subject = {
  subject_id: string;
  n_days: number;
  rhr_baseline: number;
  sleep_baseline: number;
};
type History = {
  subject_id: string;
  dates: string[];
  rhr: (number | null)[];
  rhr_baseline: (number | null)[];
  sleep_efficiency: (number | null)[];
  sleep_efficiency_baseline: (number | null)[];
  steps: (number | null)[];
  strain_proxy: (number | null)[];
};
type Prediction = {
  subject_id: string;
  target_date: string;
  rhr: { point: number; lower: number; upper: number; baseline: number };
  sleep_efficiency: { point: number; lower: number; upper: number; baseline: number };
  warning_level: "green" | "yellow" | "red";
  warning_message: string;
};
type Recommendation = {
  subject_id: string;
  recommended_max_slider: number;
  sweep: {
    slider: number;
    warning_level: string;
    rhr_point: number;
    rhr_lower: number;
    rhr_upper: number;
  }[];
};
type Mode = "sample" | "manual" | "upload" | "googlefit";

const SLIDER_LABELS: Record<number, string> = {
  1: "Rest day",
  2: "Very light",
  3: "Light walk",
  4: "Easy run",
  5: "Moderate run",
  6: "Tempo",
  7: "Hard run",
  8: "Very hard",
  9: "Race pace",
  10: "Max effort",
};

// Effort level → pictogram base name (theme suffix applied at render time)
const EFFORT_IMG: Record<number, string> = {
  1: "rest",  2: "rest",
  3: "walk",  4: "walk",
  5: "jog",   6: "jog",
  7: "run",   8: "run",
  9: "sprint", 10: "sprint",
};

const MANUAL_LABELS: Record<string, string> = {
  rhr_baseline: "Your usual resting heart rate (bpm)",
  yesterday_rhr: "Yesterday's resting heart rate",
  sleep_baseline: "Your usual sleep score (%)",
  yesterday_sleep: "Last night's sleep score",
  typical_steps: "Your usual daily steps",
  yesterday_steps: "Yesterday's steps",
};

export default function Page() {
  const [subjects, setSubjects] = useState<Subject[] | null>(null);
  const [activeId, setActiveId] = useState<string>("p01");
  const [history, setHistory] = useState<History | null>(null);
  const [slider, setSlider] = useState<number>(5);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("sample");
  const [manualForm, setManualForm] = useState({
    rhr_baseline: 60,
    yesterday_rhr: 62,
    sleep_baseline: 92,
    yesterday_sleep: 90,
    typical_steps: 10000,
    yesterday_steps: 12000,
  });
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [googleFitFile, setGoogleFitFile] = useState<File | null>(null);
  const [googleFitRhr, setGoogleFitRhr] = useState({ rhr_baseline: 60, yesterday_rhr: 62 });
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Theme init from localStorage / system
  useEffect(() => {
    const stored = localStorage.getItem("ps-theme") as "dark" | "light" | null;
    if (stored === "dark" || stored === "light") {
      setTheme(stored);
    } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
      setTheme("light");
    }
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ps-theme", theme);
  }, [theme]);

  // Load subjects
  useEffect(() => {
    fetch(`${API}/sample-subjects`)
      .then((r) => r.json())
      .then((data: Subject[]) => {
        setSubjects(data);
        if (data.length > 0) setActiveId(data[0].subject_id);
      })
      .catch((e) => setError(`Could not reach the server: ${e.message}`));
  }, []);

  // Sample mode: history + recommendation
  useEffect(() => {
    if (!activeId || mode !== "sample") return;
    setLoading(true);
    Promise.all([
      fetch(`${API}/sample-subject/${activeId}`).then((r) => r.json()),
      fetch(`${API}/recommend-workout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject_id: activeId }),
      }).then((r) => r.json()),
    ])
      .then(([h, rec]: [History, Recommendation]) => {
        setHistory(h);
        setRecommendation(rec);
        setSlider(Math.max(1, rec.recommended_max_slider));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [activeId, mode]);

  // Sample mode: prediction on slider change
  useEffect(() => {
    if (!activeId || mode !== "sample") return;
    fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: activeId, planned_strain_slider: slider }),
    })
      .then((r) => r.json())
      .then(setPrediction)
      .catch((e) => setError(e.message));
  }, [activeId, slider, mode]);

  async function runManualPredict() {
    setLoading(true);
    setError(null);
    try {
      const [predRes, recRes] = await Promise.all([
        fetch(`${API}/predict-from-stats`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...manualForm, planned_strain_slider: slider }),
        }),
        fetch(`${API}/recommend-from-stats`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(manualForm),
        }),
      ]);
      const data = await predRes.json();
      const rec = await recRes.json();
      setPrediction({
        subject_id: "manual",
        target_date: "tomorrow",
        rhr: data.rhr,
        sleep_efficiency: data.sleep_efficiency,
        warning_level: data.warning_level,
        warning_message: data.warning_message,
      });
      setRecommendation(rec);
      setSlider(Math.max(1, rec.recommended_max_slider));
      setHistory(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runGoogleFit() {
    if (!googleFitFile) return;
    setLoading(true);
    setError(null);
    setUploadStatus(`Reading ${googleFitFile.name}...`);
    try {
      const fd = new FormData();
      fd.append("file", googleFitFile);
      fd.append("rhr_baseline", String(googleFitRhr.rhr_baseline));
      fd.append("yesterday_rhr", String(googleFitRhr.yesterday_rhr));
      fd.append("planned_strain_slider", String(slider));
      const res = await fetch(`${API}/upload-google-fit`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.rhr) {
        setPrediction({
          subject_id: "googlefit",
          target_date: "tomorrow",
          rhr: data.rhr,
          sleep_efficiency: data.sleep_efficiency,
          warning_level: data.warning_level,
          warning_message: data.warning_message,
        });
        if (data.history) setHistory(data.history);
        if (data.sweep) {
          setRecommendation({
            subject_id: "googlefit",
            recommended_max_slider: data.recommended_max_slider ?? 5,
            sweep: data.sweep,
          });
          setSlider(Math.max(1, data.recommended_max_slider ?? 5));
        }
        setUploadStatus(data.note);
      } else {
        setPrediction(null);
        setUploadStatus(data.detail ?? data.note ?? "Upload failed.");
      }
    } catch (e: any) {
      setError(e.message);
      setUploadStatus(null);
    } finally {
      setLoading(false);
    }
  }

  async function runUpload(file: File) {
    setLoading(true);
    setError(null);
    setUploadStatus(`Reading ${file.name}...`);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(
        `${API}/upload-apple-health?planned_strain_slider=${slider}`,
        { method: "POST", body: fd },
      );
      const data = await res.json();
      if (data.rhr) {
        setPrediction({
          subject_id: "upload",
          target_date: "tomorrow",
          rhr: data.rhr,
          sleep_efficiency: data.sleep_efficiency,
          warning_level: data.warning_level,
          warning_message: data.warning_message,
        });
        if (data.history) setHistory(data.history);
        else setHistory(null);
        if (data.sweep) {
          setRecommendation({
            subject_id: "upload",
            recommended_max_slider: data.recommended_max_slider ?? 5,
            sweep: data.sweep,
          });
          setSlider(Math.max(1, data.recommended_max_slider ?? 5));
        }
        setUploadStatus(data.note);
      } else {
        setPrediction(null);
        setUploadStatus(data.note);
      }
    } catch (e: any) {
      setError(e.message);
      setUploadStatus(null);
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo(() => {
    if (!history) return [];
    return history.dates.map((d, i) => ({
      date: d.slice(5),
      rhr: history.rhr[i],
      baseline: history.rhr_baseline[i],
    }));
  }, [history]);

  const sleepChartData = useMemo(() => {
    if (!history) return [];
    return history.dates.map((d, i) => ({
      date: d.slice(5),
      sleep: history.sleep_efficiency[i],
      baseline: history.sleep_efficiency_baseline[i],
    }));
  }, [history]);

  const stepsChartData = useMemo(() => {
    if (!history) return [];
    const last14 = history.dates.slice(-14);
    return last14.map((d, i) => {
      const idx = history.dates.length - 14 + i;
      return {
        day: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"][new Date(d).getDay()],
        steps: Math.round((history.steps[idx] ?? 0) as number),
      };
    });
  }, [history]);

  // Activity breakdown from steps + strain — used for routine donut
  const activityBreakdown = useMemo(() => {
    if (!history) return null;
    const stepsArr = history.steps.filter((s) => s !== null) as number[];
    const strainArr = history.strain_proxy.filter((s) => s !== null) as number[];
    if (stepsArr.length < 7) return null;
    // If all steps are 0, no step data was in the export — don't show misleading 100% rest
    const hasStepData = stepsArr.some((s) => s > 0);
    if (!hasStepData) return null;
    const avgSteps = stepsArr.reduce((a, b) => a + b, 0) / stepsArr.length;
    const avgStrain = strainArr.length > 0 ? strainArr.reduce((a, b) => a + b, 0) / strainArr.length : 0;
    let active = 0, moderate = 0, rest = 0;
    stepsArr.forEach((s, i) => {
      const strain = strainArr[i] ?? 0;
      if (s > avgSteps * 1.2 || strain > avgStrain * 1.4) active++;
      else if (s > avgSteps * 0.6 || strain > avgStrain * 0.6) moderate++;
      else rest++;
    });
    const total = active + moderate + rest;
    return [
      { name: "Active", value: Math.round((active / total) * 100), fill: "var(--accent)" },
      { name: "Moderate", value: Math.round((moderate / total) * 100), fill: "var(--green)" },
      { name: "Rest", value: Math.round((rest / total) * 100), fill: "var(--surface-3)" },
    ];
  }, [history]);

  const chartDataWithPrediction = useMemo(() => {
    if (!chartData || !prediction || mode !== "sample") return chartData;
    return [
      ...chartData,
      {
        date: prediction.target_date.slice(5),
        rhr: null,
        baseline: prediction.rhr.baseline,
        prediction_rhr: prediction.rhr.point,
        prediction_band: [prediction.rhr.lower, prediction.rhr.upper],
      } as any,
    ];
  }, [chartData, prediction, mode]);

  const isAlt = mode === "googlefit" || mode === "manual";

  // ----- Sidebar body -----
  const SidebarBody = () => (
    <div className="flex flex-col h-full min-h-0">
      {/* Mode tabs */}
      <div className="mb-5 shrink-0">
        <p className="text-[10px] text-text-muted uppercase tracking-widest mb-2 px-1 font-medium">
          How are you using this?
        </p>
        <div className="flex flex-col gap-0.5">
          {(
            [
              ["upload",    "/logos/apple-watch.png", "Apple Watch user",          "Upload your Health export"],
              ["googlefit", "/logos/google-fit.png",  "Google Fit user",           "Upload your Fit export"],
              ["manual",    "/logos/wearable.png",    "Other wearable or no watch","Type your numbers in"],
              ["sample",    "/logos/browse.png",      "Just browsing",             "See a live example"],
            ] as const
          ).map(([m, logo, label, desc]) => (
            <button
              key={m}
              onClick={() => {
                setMode(m);
                setSidebarOpen(false);
              }}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all text-left ${
                mode === m
                  ? "bg-surface-2 text-accent"
                  : "text-text-soft hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <img src={logo} alt="" className="w-7 h-7 object-contain shrink-0 rounded opacity-80" />
              <span className="flex flex-col gap-0.5 flex-1">
                <span className="font-medium leading-tight">{label}</span>
                <span className={`text-[11px] leading-tight ${mode === m ? "text-accent/60" : "text-text-muted"}`}>{desc}</span>
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-border mb-4 shrink-0" />

      {/* Mode-specific inputs */}
      <div className="flex-1 overflow-y-auto thin-scroll min-h-0 pb-2">
        {mode === "sample" && (
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-widest mb-2 px-1 font-medium">
              Example users
            </p>
            <p className="text-xs text-text-soft mb-3 leading-relaxed px-1">
              Real anonymized data from 12 people wearing a Fitbit for 5 months.
            </p>
            <div className="flex flex-col gap-0.5">
              {subjects?.map((s) => (
                <button
                  key={s.subject_id}
                  onClick={() => setActiveId(s.subject_id)}
                  className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-sm transition-all ${
                    s.subject_id === activeId
                      ? "bg-surface-2 text-accent"
                      : "border-transparent text-text-soft hover:text-text hover:bg-surface-2/60"
                  }`}
                >
                  <span className="font-medium">{s.subject_id}</span>
                  <span className="text-xs text-text-muted tabular">
                    {s.rhr_baseline} bpm
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === "manual" && (
          <div className="flex flex-col gap-3">
            <button
              onClick={runManualPredict}
              disabled={loading}
              className="w-full px-4 py-2.5 bg-accent hover:bg-accent-strong text-white rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
            >
              {loading ? "Working..." : "Get my forecast"}
            </button>
          </div>
        )}

        {mode === "upload" && (
          <div>
            <label className="flex flex-col items-center justify-center px-4 py-6 border-2 border-dashed border-border-strong rounded-xl cursor-pointer hover:border-accent transition-colors bg-surface-2/40">
              <input
                type="file"
                accept=".xml,.zip"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    runUpload(f);
                    e.target.value = "";
                  }
                }}
                className="hidden"
              />
              <Upload size={22} className="mb-2 text-text-muted" />
              <span className="text-xs text-text-soft">Click to upload</span>
              <span className="text-xs text-text-muted mt-1">.zip or export.xml</span>
            </label>
            {uploadStatus && (
              <p className="mt-3 text-xs text-text-soft p-3 bg-surface-2 rounded-xl border border-border leading-relaxed">
                {uploadStatus}
              </p>
            )}
          </div>
        )}

        {mode === "googlefit" && (
          <div className="flex flex-col gap-4">
            <label className="flex flex-col items-center justify-center px-4 py-5 border-2 border-dashed border-border-strong rounded-xl cursor-pointer hover:border-accent transition-colors bg-surface-2/40">
              <input
                type="file"
                accept=".zip,.csv"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) { setGoogleFitFile(f); e.target.value = ""; }
                }}
                className="hidden"
              />
              <Activity size={22} className="mb-2 text-text-muted" />
              <span className="text-xs text-text-soft">Click to upload</span>
              <span className="text-xs text-text-muted mt-1">.zip Takeout export</span>
            </label>
            {googleFitFile && (
              <p className="text-[11px] text-accent px-1 truncate">{googleFitFile.name}</p>
            )}

            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-text-soft px-1">Usual resting HR (bpm)</label>
                <input
                  type="number"
                  min={35} max={95}
                  value={googleFitRhr.rhr_baseline}
                  onChange={(e) => setGoogleFitRhr((f) => ({ ...f, rhr_baseline: parseInt(e.target.value) || 60 }))}
                  className="w-full px-3 py-2 bg-surface-2 border border-border rounded-xl text-sm text-text tabular outline-none focus:border-accent transition-colors"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-text-soft px-1">Yesterday's resting HR (bpm)</label>
                <input
                  type="number"
                  min={35} max={95}
                  value={googleFitRhr.yesterday_rhr}
                  onChange={(e) => setGoogleFitRhr((f) => ({ ...f, yesterday_rhr: parseInt(e.target.value) || 62 }))}
                  className="w-full px-3 py-2 bg-surface-2 border border-border rounded-xl text-sm text-text tabular outline-none focus:border-accent transition-colors"
                />
              </div>
            </div>

            <button
              onClick={runGoogleFit}
              disabled={loading || !googleFitFile}
              className="w-full px-4 py-2.5 bg-accent hover:bg-accent-strong text-white rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
            >
              {loading ? "Working..." : "Get forecast"}
            </button>

            {uploadStatus && (
              <p className="text-[11px] text-text-muted leading-relaxed px-1">{uploadStatus}</p>
            )}
          </div>
        )}
      </div>

      {/* Footer links */}
      <div className="mt-3 pt-3 border-t border-border flex flex-col gap-0.5 shrink-0">
        <a
          href="https://github.com/"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm text-text-soft hover:text-text hover:bg-surface-2 transition-colors"
        >
          <Github size={14} /> GitHub
        </a>
        <a
          href="#"
          className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm text-text-soft hover:text-text hover:bg-surface-2 transition-colors"
        >
          <FileText size={14} /> Paper
        </a>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-bg text-text">
      {/* Top header */}
      <header className="sticky top-0 z-20 border-b border-border/60 bg-bg/75 backdrop-blur-xl h-14 flex items-center px-4 md:px-6">
        <div className="flex items-center justify-between w-full gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 text-text-soft hover:text-text rounded-xl hover:bg-surface-2 transition-colors"
              aria-label="Toggle menu"
            >
              {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
            <Image
              src="/pulscale-logo.png"
              alt="Pulscale"
              width={32}
              height={32}
              className="rounded-lg"
            />
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight">Pulscale</div>
              <div className="text-[11px] text-text-muted hidden sm:block leading-none mt-0.5">
                Daily recovery forecaster
              </div>
            </div>
          </div>
          <button
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm text-text-soft hover:text-text border border-border hover:border-border-strong hover:bg-surface-2 transition-all"
            aria-label="Toggle light/dark mode"
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            <span className="hidden sm:inline text-xs">
              {theme === "dark" ? "Light" : "Dark"}
            </span>
          </button>
        </div>
      </header>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-10 bg-black/50 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Body: sidebar + main */}
      <div className="flex" style={{ minHeight: "calc(100vh - 3.5rem)" }}>
        {/* Sidebar */}
        <aside
          className={[
            "fixed lg:static inset-y-14 left-0 z-10",
            "w-64 xl:w-72",
            "flex flex-col",
            "bg-bg/95 lg:bg-transparent backdrop-blur-xl lg:backdrop-blur-none",
            "border-r border-border",
            "p-4",
            "overflow-hidden",
            "transition-transform duration-300 ease-in-out",
            sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          ].join(" ")}
        >
          <SidebarBody />
        </aside>

        {/* Main dashboard */}
        <main className="flex-1 min-w-0 overflow-y-auto p-4 md:p-6 lg:p-7">
          {/* Greeting */}
          <div className="mb-6">
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
              Tomorrow&apos;s recovery
            </h1>
            <p className="text-sm text-text-soft mt-1.5">
              Your resting heart rate and sleep score, forecast for tomorrow.
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-4 p-3 rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 text-sm flex items-start gap-2">
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Dashboard grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 mb-6">

            {/* ── ALT TOP ROW (googlefit / manual): 3 equal stat cards ── */}
            {isAlt && (
              <div className="lg:col-span-12 grid grid-cols-1 sm:grid-cols-3 gap-3">
                <StatCard icon={<Heart size={20} />} label="Resting heart rate" value={prediction?.rhr.point} unit="bpm"
                  sub={prediction ? `${prediction.rhr.lower.toFixed(0)} to ${prediction.rhr.upper.toFixed(0)} bpm` : undefined}
                  accentColor="accent" />
                <StatCard icon={<Moon size={20} />} label="Sleep score" value={prediction?.sleep_efficiency.point} unit="%"
                  sub={prediction ? `${prediction.sleep_efficiency.lower.toFixed(0)} to ${prediction.sleep_efficiency.upper.toFixed(0)}%` : undefined}
                  accentColor="green" />
                <StatusBentoCard prediction={prediction} />
              </div>
            )}

            {/* ── GOOGLE FIT MIDDLE ROW: steps (8 cols) + day types (4 cols) ── */}
            {mode === "googlefit" && (
              <>
                <div className="lg:col-span-8 lg:self-start bento p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="p-1.5 rounded-lg bg-accent-soft text-accent shrink-0"><Activity size={13} /></div>
                    <h3 className="text-sm font-semibold">Daily steps, last 14 days</h3>
                  </div>
                  <div className="h-48">
                    {!history || stepsChartData.every(d => d.steps === 0) ? (
                      <NoChartPlaceholder metric="step count" mode={mode} />
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stepsChartData} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                          <defs>
                            <linearGradient id="stepsHighAlt" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="var(--accent)" stopOpacity={1} />
                              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.4} />
                            </linearGradient>
                            <linearGradient id="stepsLowAlt" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="var(--text-soft)" stopOpacity={0.7} />
                              <stop offset="100%" stopColor="var(--text-soft)" stopOpacity={0.35} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke="var(--border)" strokeDasharray="2 6" vertical={false} />
                          <XAxis dataKey="day" tick={{ fontSize: 9, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontSize: 9, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} width={36} tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 10, color: "var(--text)", fontSize: 11, boxShadow: "0 8px 24px rgba(0,0,0,0.3)" }} formatter={(v: number) => [v.toLocaleString(), "steps"]} />
                          <Bar dataKey="steps" radius={[5, 5, 0, 0]} name="Steps">
                            {stepsChartData.map((entry, i) => {
                              const avg = stepsChartData.reduce((s, d) => s + d.steps, 0) / stepsChartData.length;
                              return <Cell key={i} fill={entry.steps >= avg ? "url(#stepsHighAlt)" : "url(#stepsLowAlt)"} />;
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                <div className="lg:col-span-4 lg:self-start bento p-5 flex flex-col">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="p-1.5 rounded-lg bg-accent-soft text-accent shrink-0"><TrendingUp size={13} /></div>
                    <h3 className="text-sm font-semibold">Day types</h3>
                  </div>
                  <p className="text-[10px] text-text-muted mb-3 leading-relaxed">Based on steps and strain.</p>
                  <div className="flex-1 flex flex-col items-center justify-center">
                    {activityBreakdown ? (
                      <>
                        <div className="relative w-32 h-32">
                          <ResponsiveContainer width="100%" height="100%">
                            <RadialBarChart innerRadius={38} outerRadius={60} data={activityBreakdown} startAngle={90} endAngle={-270}>
                              <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
                              {activityBreakdown.map((entry, i) => (
                                <RadialBar key={i} dataKey="value" cornerRadius={4} fill={entry.fill} background={i === 0 ? { fill: "var(--surface-3)" } : undefined} angleAxisId={0} />
                              ))}
                            </RadialBarChart>
                          </ResponsiveContainer>
                          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                            <span className="text-xl font-bold tabular leading-none">{activityBreakdown[0].value}</span>
                            <span className="text-[10px] text-text-muted">% active</span>
                          </div>
                        </div>
                        <div className="flex flex-col gap-1 mt-3 w-full">
                          {activityBreakdown.map((d) => (
                            <div key={d.name} className="flex items-center justify-between text-xs">
                              <div className="flex items-center gap-1.5">
                                <div className="w-2 h-2 rounded-full" style={{ background: d.fill }} />
                                <span className="text-text-soft">{d.name}</span>
                              </div>
                              <span className="text-text-muted tabular">{d.value}%</span>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-text-muted text-center leading-relaxed">Not enough data yet. Need 7+ days.</p>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* ── ALT BOTTOM ROW (googlefit / manual): 3 action cards ── */}
            {isAlt && (
              <div className="lg:col-span-12 grid grid-cols-1 sm:grid-cols-3 gap-3 lg:items-start">
                <SuggestedCard recommendation={recommendation} mode={mode} />
                <ForecastPrecisionCard prediction={prediction} />
                <WorkoutPlanCard slider={slider} setSlider={setSlider} prediction={prediction} recommendation={recommendation} theme={theme} />
              </div>
            )}

            {/* ── NORMAL LAYOUT (sample / upload) ── */}
            {!isAlt && (
              <div className="lg:col-span-8 flex flex-col gap-3 lg:self-start">
                <div className="grid grid-cols-2 gap-3">
                  <StatCard icon={<Heart size={20} />} label="Resting heart rate" value={prediction?.rhr.point} unit="bpm"
                    sub={prediction ? `${prediction.rhr.lower.toFixed(0)} to ${prediction.rhr.upper.toFixed(0)} bpm` : undefined}
                    usual={prediction?.rhr.baseline ? `${prediction.rhr.baseline.toFixed(0)} bpm usual` : undefined}
                    accentColor="accent" />
                  <StatCard icon={<Moon size={20} />} label="Sleep score" value={prediction?.sleep_efficiency.point} unit="%"
                    sub={prediction ? `${prediction.sleep_efficiency.lower.toFixed(0)} to ${prediction.sleep_efficiency.upper.toFixed(0)}%` : undefined}
                    usual={prediction?.sleep_efficiency.baseline ? `${prediction.sleep_efficiency.baseline.toFixed(0)}% usual` : undefined}
                    accentColor="green" />
                </div>

                <div className="bento p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Heart size={13} className="text-accent" />
                    <h3 className="text-sm font-semibold">Resting heart rate, 60 days</h3>
                  </div>
                  <div className="h-56" key={history?.subject_id ?? mode}>
                    {!history ? <NoChartPlaceholder metric="resting heart rate" mode={mode} /> :
                     loading ? <ChartLoader /> : (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartDataWithPrediction} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
                          <defs>
                            <linearGradient id="rhrGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.4} />
                              <stop offset="75%" stopColor="var(--accent)" stopOpacity={0.05} />
                              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.12} />
                              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke="var(--border)" strokeDasharray="2 8" vertical={false} />
                          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={40} />
                          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} width={36} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 12, color: "var(--text)", fontSize: 12, boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }} labelStyle={{ color: "var(--text-soft)" }} cursor={{ stroke: "var(--accent)", strokeWidth: 1, strokeDasharray: "4 4" }} />
                          <Area dataKey="prediction_band" fill="url(#bandGrad)" stroke="none" name="90% range" isAnimationActive={false} />
                          <Area dataKey="rhr" stroke="var(--accent)" strokeWidth={2.5} fill="url(#rhrGrad)" dot={false} activeDot={{ r: 5, fill: "var(--accent)", stroke: "var(--bg)", strokeWidth: 2 }} name="Resting HR" animationBegin={100} animationDuration={1400} animationEasing="ease-out" connectNulls={false} />
                          <Line dataKey="baseline" stroke="var(--text-muted)" strokeDasharray="4 6" strokeWidth={1.5} dot={false} name="30-day avg" animationBegin={300} animationDuration={1600} animationEasing="ease-out" />
                          <Line dataKey="prediction_rhr" stroke="var(--accent)" strokeWidth={0} dot={{ r: 7, fill: "var(--accent)", stroke: "var(--bg)", strokeWidth: 3 }} name="Tomorrow" isAnimationActive={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-5 gap-3">
                  <div className="bento p-5 col-span-3">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="p-1.5 rounded-lg bg-accent-soft text-accent shrink-0"><Activity size={13} /></div>
                      <h3 className="text-sm font-semibold">Daily steps, last 14 days</h3>
                    </div>
                    <div className="h-40">
                      {!history || stepsChartData.every(d => d.steps === 0) ? (
                        <NoChartPlaceholder metric="step count" mode={mode} />
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={stepsChartData} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                            <defs>
                              <linearGradient id="stepsHigh" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--accent)" stopOpacity={1} />
                                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.4} />
                              </linearGradient>
                              <linearGradient id="stepsLow" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--text-soft)" stopOpacity={0.7} />
                                <stop offset="100%" stopColor="var(--text-soft)" stopOpacity={0.35} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid stroke="var(--border)" strokeDasharray="2 6" vertical={false} />
                            <XAxis dataKey="day" tick={{ fontSize: 9, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fontSize: 9, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} width={36} tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                            <Tooltip contentStyle={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 10, color: "var(--text)", fontSize: 11, boxShadow: "0 8px 24px rgba(0,0,0,0.3)" }} formatter={(v: number) => [v.toLocaleString(), "steps"]} />
                            <Bar dataKey="steps" radius={[5, 5, 0, 0]} name="Steps">
                              {stepsChartData.map((entry, i) => {
                                const avg = stepsChartData.reduce((s, d) => s + d.steps, 0) / stepsChartData.length;
                                return <Cell key={i} fill={entry.steps >= avg ? "url(#stepsHigh)" : "url(#stepsLow)"} />;
                              })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>
                  <div className="bento p-5 col-span-2 flex flex-col">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-1.5 rounded-lg bg-accent-soft text-accent shrink-0"><TrendingUp size={13} /></div>
                      <h3 className="text-sm font-semibold">Day types</h3>
                    </div>
                    <p className="text-[10px] text-text-muted mb-3 leading-relaxed">
                      Based on steps and strain. No workout type data available.
                    </p>
                    <div className="flex-1 flex flex-col items-center justify-center">
                      {activityBreakdown ? (
                        <>
                          <div className="relative w-32 h-32">
                            <ResponsiveContainer width="100%" height="100%">
                              <RadialBarChart innerRadius={38} outerRadius={60} data={activityBreakdown} startAngle={90} endAngle={-270}>
                                <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
                                {activityBreakdown.map((entry, i) => (
                                  <RadialBar key={i} dataKey="value" cornerRadius={4} fill={entry.fill} background={i === 0 ? { fill: "var(--surface-3)" } : undefined} angleAxisId={0} />
                                ))}
                              </RadialBarChart>
                            </ResponsiveContainer>
                            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                              <span className="text-xl font-bold tabular leading-none">{activityBreakdown[0].value}</span>
                              <span className="text-[10px] text-text-muted">% active</span>
                            </div>
                          </div>
                          <div className="flex flex-col gap-1 mt-3 w-full">
                            {activityBreakdown.map((d) => (
                              <div key={d.name} className="flex items-center justify-between text-xs">
                                <div className="flex items-center gap-1.5">
                                  <div className="w-2 h-2 rounded-full" style={{ background: d.fill }} />
                                  <span className="text-text-soft">{d.name}</span>
                                </div>
                                <span className="text-text-muted tabular">{d.value}%</span>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : (
                        <p className="text-xs text-text-muted text-center leading-relaxed">
                          Not enough data yet. Need 7+ days of history.
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="bento p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Moon size={13} className="text-green-400" />
                    <h3 className="text-sm font-semibold">Sleep score, 60 days</h3>
                  </div>
                  <div className="h-44" key={history?.subject_id ?? mode}>
                    {!history ? <NoChartPlaceholder metric="sleep score" mode={mode} /> :
                     loading ? <ChartLoader /> : (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart
                          data={[...sleepChartData, ...(prediction && mode === "sample" ? [{ date: prediction.target_date.slice(5), sleep: null, baseline: prediction.sleep_efficiency.baseline, prediction_sleep: prediction.sleep_efficiency.point }] : [])]}
                          margin={{ top: 8, right: 12, left: 4, bottom: 0 }}
                        >
                          <defs>
                            <linearGradient id="sleepGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="var(--green)" stopOpacity={0.4} />
                              <stop offset="75%" stopColor="var(--green)" stopOpacity={0.05} />
                              <stop offset="100%" stopColor="var(--green)" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke="var(--border)" strokeDasharray="2 8" vertical={false} />
                          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={40} />
                          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} width={36} />
                          <Tooltip contentStyle={{ backgroundColor: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 12, color: "var(--text)", fontSize: 12, boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }} labelStyle={{ color: "var(--text-soft)" }} cursor={{ stroke: "var(--green)", strokeWidth: 1, strokeDasharray: "4 4" }} />
                          <Area dataKey="sleep" stroke="var(--green)" strokeWidth={2.5} fill="url(#sleepGrad)" dot={false} activeDot={{ r: 5, fill: "var(--green)", stroke: "var(--bg)", strokeWidth: 2 }} name="Sleep %" animationBegin={100} animationDuration={1400} animationEasing="ease-out" connectNulls={false} />
                          <Line dataKey="baseline" stroke="var(--text-muted)" strokeDasharray="4 6" strokeWidth={1.5} dot={false} name="30-day avg" animationBegin={300} animationDuration={1600} animationEasing="ease-out" />
                          <Line dataKey="prediction_sleep" stroke="var(--green)" strokeWidth={0} dot={{ r: 7, fill: "var(--green)", stroke: "var(--bg)", strokeWidth: 3 }} name="Tomorrow" isAnimationActive={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Normal layout right column */}
            {!isAlt && (
              <div className="lg:col-span-4 flex flex-col gap-3 lg:self-start">
                <StatusBentoCard prediction={prediction} />
                <SuggestedCard recommendation={recommendation} mode={mode} />
                <ForecastPrecisionCard prediction={prediction} />
                <WorkoutPlanCard slider={slider} setSlider={setSlider} prediction={prediction} recommendation={recommendation} theme={theme} />
              </div>
            )}
          </div>

          {/* Footer */}
          <footer className="border-t border-border pt-5 text-xs text-text-muted">
            <p className="leading-relaxed">
              Trained on PMData (Thambawita et al., MMSys 2020). 3-model ensemble.
              90% ranges from split conformal prediction (Angelopoulos and Bates, 2021).
              Tested on held-out users:{" "}
              <span className="font-mono tabular text-text-soft">0.886 resting HR</span>{" "}
              /{" "}
              <span className="font-mono tabular text-text-soft">0.905 sleep</span>{" "}
              at 90% target.
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}

// ---- Sub-components ----

function StatCard({
  icon,
  label,
  value,
  unit,
  sub,
  usual,
  accentColor = "accent",
}: {
  icon: React.ReactNode;
  label: string;
  value: number | undefined;
  unit: string;
  sub?: string;
  usual?: string;
  accentColor?: "accent" | "green" | "yellow";
}) {
  const colorMap = {
    accent: "bg-accent-soft text-accent",
    green: "bg-green-500/10 text-green-400",
    yellow: "bg-amber-500/10 text-amber-400",
  };
  return (
    <div className="bento p-5 flex flex-col gap-3">
      <div className={`w-11 h-11 rounded-2xl flex items-center justify-center shrink-0 ${colorMap[accentColor]}`}>
        {icon}
      </div>
      <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold leading-tight">
        {label}
      </p>
      {value !== undefined ? (
        <div>
          <div className="flex items-end gap-1.5 leading-none">
            <span className="stat-num">{value.toFixed(1)}</span>
            <span className="text-text-muted text-base font-medium mb-1">{unit}</span>
          </div>
          {sub && <p className="text-xs text-text-soft mt-2">{sub}</p>}
          {usual && <p className="text-[11px] text-text-muted mt-1">{usual}</p>}
        </div>
      ) : (
        <div className="stat-num text-surface-3">--</div>
      )}
    </div>
  );
}

function StatusBentoCard({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) {
    return (
      <div className="bento p-5 flex flex-col gap-3">
        <div className="w-11 h-11 rounded-2xl flex items-center justify-center bg-surface-3 text-text-muted">
          <AlertCircle size={20} />
        </div>
        <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold">Status</p>
        <div className="stat-num text-surface-3">--</div>
      </div>
    );
  }
  const p = {
    green: {
      outer: "bg-gradient-to-br from-green-500/14 via-green-500/6 to-transparent",
      badge: "bg-green-500/20 text-green-400",
      title: "text-ok",
      sub: "text-ok opacity-70",
      icon: <CheckCircle2 size={20} />,
      label: "On track",
    },
    yellow: {
      outer: "bg-gradient-to-br from-amber-500/14 via-amber-500/6 to-transparent",
      badge: "bg-amber-500/20 text-amber-400",
      title: "text-amber-200",
      sub: "text-amber-400/70",
      icon: <AlertTriangle size={20} />,
      label: "Take it easy",
    },
    red: {
      outer: "bg-gradient-to-br from-red-500/14 via-red-500/6 to-transparent",
      badge: "bg-red-500/20 text-red-400",
      title: "text-red-200",
      sub: "text-red-400/70",
      icon: <AlertCircle size={20} />,
      label: "Rest day",
    },
  }[prediction.warning_level];
  return (
    <div className={`bento p-5 flex flex-col gap-3 ${p.outer}`}>
      <div className={`w-11 h-11 rounded-2xl flex items-center justify-center ${p.badge}`}>
        {p.icon}
      </div>
      <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold">Status</p>
      <div>
        <p className={`text-3xl font-bold tracking-tight ${p.title}`}>{p.label}</p>
        <p className={`text-xs mt-2 leading-relaxed ${p.sub}`}>{prediction.warning_message}</p>
      </div>
    </div>
  );
}

function SuggestedCard({
  recommendation,
  mode,
}: {
  recommendation: Recommendation | null;
  mode: Mode;
}) {
  return (
    <div className="bento p-5 flex flex-col gap-3">
      <div className="w-11 h-11 rounded-2xl flex items-center justify-center bg-accent-soft text-accent">
        <Zap size={20} />
      </div>
      <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold">
        Suggested effort
      </p>
      {recommendation ? (
        <div>
          <div className="flex items-end gap-1.5 leading-none">
            <span className="stat-num">{recommendation.recommended_max_slider}</span>
            <span className="text-text-muted text-base font-medium mb-1">/10</span>
          </div>
          <p className="text-xs text-text-soft mt-2">
            {SLIDER_LABELS[recommendation.recommended_max_slider] ?? "Custom"}
          </p>
          <p className="text-[11px] text-text-muted mt-1">max before caution zone</p>
        </div>
      ) : (
        <div>
          <div className="stat-num text-surface-3">--</div>
          <p className="text-xs text-text-muted mt-2">
            {mode === "sample" ? "Loading..." : "Get a forecast to see this"}
          </p>
        </div>
      )}
    </div>
  );
}

function NoChartPlaceholder({ metric, mode }: { metric: string; mode: Mode }) {
  const msg =
    mode === "upload"
      ? `Upload your Apple Health export to see your ${metric} chart`
      : mode === "googlefit"
      ? `Upload your Google Fit export to see your ${metric} chart`
      : mode === "manual"
      ? `${metric.charAt(0).toUpperCase() + metric.slice(1)} history isn't available in manual mode — use Apple Watch upload for charts`
      : `${metric.charAt(0).toUpperCase() + metric.slice(1)} chart will appear here after you get a forecast`;
  return (
    <div className="h-full flex flex-col items-center justify-center gap-2 opacity-50">
      <TrendingUp size={28} className="text-text-muted" />
      <p className="text-sm text-text-muted text-center">{msg}</p>
    </div>
  );
}

function ChartLoader() {
  return (
    <div className="h-full flex items-center justify-center text-text-muted text-sm">
      Loading...
    </div>
  );
}

function ForecastPrecisionCard({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) {
    return (
      <div className="bento p-5 flex flex-col gap-3">
        <div className="w-11 h-11 rounded-2xl flex items-center justify-center bg-surface-3 text-text-muted">
          <TrendingUp size={20} />
        </div>
        <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold">Forecast confidence</p>
        <div className="stat-num text-surface-3">--</div>
      </div>
    );
  }

  const hrWidth = prediction.rhr.upper - prediction.rhr.lower;
  const sleepWidth = prediction.sleep_efficiency.upper - prediction.sleep_efficiency.lower;
  const hrPrecision = Math.max(0, Math.round(100 - (hrWidth / 10) * 100));
  const sleepPrecision = Math.max(0, Math.round(100 - (sleepWidth / 20) * 100));
  const barColor = (p: number) => p >= 70 ? "bg-green-400" : p >= 45 ? "bg-amber-400" : "bg-red-400";

  return (
    <div className="bento p-5 flex flex-col gap-3">
      <div className="w-11 h-11 rounded-2xl flex items-center justify-center bg-accent-soft text-accent">
        <TrendingUp size={20} />
      </div>
      <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold">Forecast confidence</p>
      <div className="flex flex-col gap-2.5">
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-text-soft">HR</span>
            <span className="tabular text-text-muted">±{(hrWidth / 2).toFixed(1)} bpm</span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor(hrPrecision)}`} style={{ width: `${hrPrecision}%` }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-text-soft">Sleep</span>
            <span className="tabular text-text-muted">±{(sleepWidth / 2).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor(sleepPrecision)}`} style={{ width: `${sleepPrecision}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

function EffortPictogram({
  slider,
  theme,
}: {
  slider: number;
  theme: "dark" | "light";
}) {
  const base = EFFORT_IMG[slider] ?? "jog";
  const src = `/activity/${base}-${theme}.png`;

  return (
    <div className="flex items-center justify-center">
      <img
        key={src}
        src={src}
        alt={SLIDER_LABELS[slider]}
        className="w-28 h-28 object-contain transition-opacity duration-300"
      />
    </div>
  );
}

function SweepBars({
  sweep,
  activeSlider,
}: {
  sweep: Recommendation["sweep"];
  activeSlider: number;
}) {
  const values = sweep.map((s) => s.rhr_point);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const colorMap: Record<string, string> = {
    green: "bg-green-400",
    yellow: "bg-amber-400",
    red: "bg-red-400",
  };

  return (
    <div>
      <p className="text-[10px] text-text-muted uppercase tracking-widest font-semibold mb-2">
        HR impact by effort level
      </p>
      <div className="flex items-end gap-1 h-16">
        {sweep.map((pt) => {
          const isActive = pt.slider === activeSlider;
          const isNear = Math.abs(pt.slider - activeSlider) === 1;
          const heightPct = 20 + ((pt.rhr_point - min) / range) * 80;
          const color = colorMap[pt.warning_level] ?? "bg-accent";
          return (
            <div key={pt.slider} className="flex-1 flex flex-col items-center gap-0.5">
              <div
                className={`w-full rounded-md transition-all duration-150 ${color} ${
                  isActive ? "opacity-100 shadow-lg" : isNear ? "opacity-50" : "opacity-20"
                }`}
                style={{ height: `${heightPct}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[9px] text-text-muted mt-1 tabular">
        <span>1</span>
        <span>5</span>
        <span>10</span>
      </div>
    </div>
  );
}

function LiveForecastRow({
  prediction,
  sweep,
  slider,
}: {
  prediction: Prediction;
  sweep: Recommendation["sweep"] | undefined;
  slider: number;
}) {
  const restRhr = sweep?.find((s) => s.slider === 1)?.rhr_point;
  const delta = restRhr !== undefined ? prediction.rhr.point - restRhr : null;
  const warnColor =
    prediction.warning_level === "green"
      ? "text-green-400"
      : prediction.warning_level === "yellow"
      ? "text-amber-400"
      : "text-red-400";
  const warnBg =
    prediction.warning_level === "green"
      ? "bg-green-500/10 border-green-500/20"
      : prediction.warning_level === "yellow"
      ? "bg-amber-500/10 border-amber-500/20"
      : "bg-red-500/10 border-red-500/20";

  return (
    <div className={`rounded-xl border p-3 ${warnBg}`}>
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <span className={`text-2xl font-bold tabular ${warnColor}`}>
            {prediction.rhr.point.toFixed(1)}
          </span>
          <span className="text-text-muted text-xs ml-1">bpm tomorrow</span>
        </div>
        {delta !== null && (
          <span className={`text-xs font-semibold tabular ${delta > 0 ? "text-red-400" : "text-green-400"}`}>
            {delta > 0 ? "+" : ""}{delta.toFixed(1)} vs rest
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-[10px] text-text-muted tabular shrink-0">
          {prediction.rhr.lower.toFixed(0)}
        </span>
        <div className="flex-1 h-1.5 rounded-full bg-surface-3 relative overflow-hidden">
          <div
            className={`absolute inset-y-0 left-0 rounded-full transition-all duration-300 ${
              prediction.warning_level === "green" ? "bg-green-400" :
              prediction.warning_level === "yellow" ? "bg-amber-400" : "bg-red-400"
            }`}
            style={{
              width: `${((prediction.rhr.point - prediction.rhr.lower) / (prediction.rhr.upper - prediction.rhr.lower)) * 100}%`,
            }}
          />
        </div>
        <span className="text-[10px] text-text-muted tabular shrink-0">
          {prediction.rhr.upper.toFixed(0)} bpm
        </span>
      </div>
    </div>
  );
}


function WorkoutPlanCard({
  slider,
  setSlider,
  prediction,
  recommendation,
  theme,
}: {
  slider: number;
  setSlider: (v: number) => void;
  prediction: Prediction | null;
  recommendation: Recommendation | null;
  theme: "dark" | "light";
}) {
  return (
    <div className="bento p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="p-1.5 rounded-lg bg-accent-soft text-accent shrink-0">
          <Activity size={14} />
        </div>
        <h3 className="text-sm font-semibold">Workout plan</h3>
      </div>
      <EffortPictogram slider={slider} theme={theme} />
      {prediction && (
        <LiveForecastRow prediction={prediction} sweep={recommendation?.sweep} slider={slider} />
      )}
      <div>
        <div className="flex items-end gap-1 mb-1">
          <span className="stat-num" style={{ fontSize: "3rem" }}>{slider}</span>
          <span className="text-text-muted text-lg font-medium mb-0.5">/10</span>
          <span className="text-text-soft text-sm font-medium ml-2 mb-1">{SLIDER_LABELS[slider]}</span>
        </div>
        <input type="range" min={1} max={10} step={1} value={slider}
          onChange={(e) => setSlider(parseInt(e.target.value))}
          className="pulscale-slider" />
        <div className="flex justify-between text-[10px] text-text-muted tabular mt-1.5">
          <span>Rest</span><span>Easy</span><span>Mod</span><span>Hard</span><span>Max</span>
        </div>
      </div>
    </div>
  );
}
