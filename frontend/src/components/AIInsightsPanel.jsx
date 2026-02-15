import React, { useState, useEffect } from 'react';
import { getQuickTip, analyzePatterns, getAIStatus } from '../api/client';

/**
 * AIInsightsPanel — Displays Gemini-powered health insights.
 *
 * Two modes:
 *  1. Quick Tip (Flash): Instant feedback after daily log submission
 *  2. Deep Analysis (Pro): Multi-day pattern analysis on the Insights page
 *
 * Props:
 *  - mode: 'quick-tip' | 'deep-analysis'
 *  - dailyLog: (for quick-tip) the daily log data just submitted
 *  - userId: (for deep-analysis) user ID to fetch history
 *  - autoLoad: trigger analysis on mount
 */
export default function AIInsightsPanel({ mode = 'quick-tip', dailyLog = null, userId = null, autoLoad = false }) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [aiAvailable, setAiAvailable] = useState(null);

    useEffect(() => {
        getAIStatus()
            .then(res => setAiAvailable(res.data.ai_available))
            .catch(() => setAiAvailable(false));
    }, []);

    useEffect(() => {
        if (autoLoad && mode === 'quick-tip' && dailyLog) {
            fetchQuickTip();
        }
        if (autoLoad && mode === 'deep-analysis' && userId) {
            fetchDeepAnalysis();
        }
    }, [autoLoad, dailyLog, userId]);

    const fetchQuickTip = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await getQuickTip(dailyLog);
            setResult(res.data.data);
        } catch (err) {
            setError('Could not generate tip. Using fallback insights.');
            // Try to still show something useful
            setResult({
                source: 'fallback',
                tip: {
                    greeting: 'Thanks for logging today! 👋',
                    quick_tip: 'Keep up your daily logging streak — consistency helps Hea learn your patterns.',
                    encouragement: 'Every entry makes your insights smarter ✨',
                },
                success: false,
            });
        }
        setLoading(false);
    };

    const fetchDeepAnalysis = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await analyzePatterns(userId, 7);
            setResult(res.data.data);
        } catch (err) {
            setError('Could not run deep analysis. Try again later.');
        }
        setLoading(false);
    };

    // ─── Loading State ───────────────────────────────
    if (loading) {
        return (
            <div className="card" style={{ textAlign: 'center', padding: 40 }}>
                <div className="pulse" style={{ fontSize: '2rem', marginBottom: 12 }}>
                    {mode === 'quick-tip' ? '⚡' : '🧠'}
                </div>
                <p style={{ fontWeight: 600, color: 'var(--hea-text-dark)' }}>
                    {mode === 'quick-tip' ? 'Generating your wellness tip...' : 'Analyzing your health patterns...'}
                </p>
                <p className="text-sm text-muted" style={{ marginTop: 4 }}>
                    Powered by Gemini AI
                </p>
            </div>
        );
    }

    // ─── Quick Tip View ──────────────────────────────
    if (mode === 'quick-tip' && result?.tip) {
        const { tip, source } = result;
        return (
            <div className="card animate-in" style={{
                background: 'linear-gradient(135deg, #F0F7FF 0%, #E8F5E9 100%)',
                border: '1px solid rgba(59, 107, 245, 0.15)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                    <span style={{ fontSize: '1.4rem' }}>⚡</span>
                    <h3 style={{ margin: 0, fontSize: '1rem' }}>AI Wellness Tip</h3>
                    {source === 'gemini-flash' && (
                        <span style={{
                            fontSize: '0.65rem', padding: '2px 8px',
                            background: 'var(--hea-primary-light)', color: 'var(--hea-primary)',
                            borderRadius: 'var(--radius-full)', fontWeight: 600,
                        }}>
                            Gemini Flash
                        </span>
                    )}
                </div>

                <p style={{ fontSize: '1.05rem', fontWeight: 500, color: 'var(--hea-text-dark)', marginBottom: 12 }}>
                    {tip.greeting}
                </p>

                <div style={{
                    background: 'rgba(255,255,255,0.7)', borderRadius: 'var(--radius-md)',
                    padding: '14px 16px', marginBottom: 12,
                }}>
                    <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--hea-text-body)' }}>
                        💡 {tip.quick_tip}
                    </p>
                </div>

                <p className="text-sm" style={{ color: 'var(--hea-primary)', fontWeight: 500 }}>
                    {tip.encouragement}
                </p>

                {!result?.success && !autoLoad && (
                    <button className="btn btn-secondary btn-sm" style={{ marginTop: 12 }}
                        onClick={fetchQuickTip}>
                        🔄 Retry with AI
                    </button>
                )}
            </div>
        );
    }

    // ─── Deep Analysis View ──────────────────────────
    if (mode === 'deep-analysis' && result?.analysis) {
        const { analysis, source } = result;
        return (
            <div className="card animate-in" style={{
                background: 'linear-gradient(135deg, #F5F0FF 0%, #EBF0FF 100%)',
                border: '1px solid rgba(106, 90, 205, 0.15)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
                    <span style={{ fontSize: '1.4rem' }}>🧠</span>
                    <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Deep Pattern Analysis</h3>
                    {source === 'gemini-pro' && (
                        <span style={{
                            fontSize: '0.65rem', padding: '2px 8px',
                            background: '#F0EAFF', color: '#6A5ACD',
                            borderRadius: 'var(--radius-full)', fontWeight: 600,
                        }}>
                            Gemini Pro
                        </span>
                    )}
                </div>

                {/* Summary */}
                <p style={{
                    fontSize: '1.05rem', fontWeight: 500, lineHeight: 1.6,
                    color: 'var(--hea-text-dark)', marginBottom: 16,
                    paddingBottom: 16, borderBottom: '1px solid rgba(0,0,0,0.06)',
                }}>
                    {analysis.summary}
                </p>

                {/* Patterns Detected */}
                {analysis.patterns && analysis.patterns.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: 8, color: 'var(--hea-text-dark)' }}>
                            📊 Patterns Detected
                        </h4>
                        {analysis.patterns.map((pattern, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'flex-start', gap: 8,
                                padding: '8px 12px', marginBottom: 6,
                                background: 'rgba(255,255,255,0.6)', borderRadius: 'var(--radius-sm)',
                                borderLeft: '3px solid var(--hea-primary)',
                            }}>
                                <span className="text-sm" style={{ lineHeight: 1.5 }}>{pattern}</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Recommendations */}
                {analysis.recommendations && analysis.recommendations.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: 8, color: 'var(--hea-text-dark)' }}>
                            💡 Recommendations
                        </h4>
                        {analysis.recommendations.map((rec, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'flex-start', gap: 8,
                                padding: '8px 12px', marginBottom: 6,
                                background: 'rgba(255,255,255,0.6)', borderRadius: 'var(--radius-sm)',
                                borderLeft: '3px solid var(--risk-low)',
                            }}>
                                <span className="text-sm" style={{ lineHeight: 1.5 }}>{rec}</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Risk Note */}
                {analysis.risk_note && (
                    <div style={{
                        padding: '12px 16px', borderRadius: 'var(--radius-md)',
                        background: 'rgba(255,255,255,0.7)', marginBottom: 12,
                    }}>
                        <p className="text-sm" style={{ lineHeight: 1.6 }}>
                            🛡️ <strong>Risk Note:</strong> {analysis.risk_note}
                        </p>
                    </div>
                )}

                {/* Disclaimer */}
                <p className="text-sm text-muted" style={{ marginTop: 12, fontStyle: 'italic', lineHeight: 1.5 }}>
                    {analysis.disclaimer || '⚕️ Hea is a wellness tool, not medical advice. Consult a healthcare professional for medical concerns.'}
                </p>

                <button className="btn btn-secondary btn-sm" style={{ marginTop: 16 }}
                    onClick={fetchDeepAnalysis}>
                    🔄 Refresh Analysis
                </button>
            </div>
        );
    }

    // ─── Default / CTA State ─────────────────────────
    return (
        <div className="card animate-in" style={{
            textAlign: 'center', padding: 32,
            background: 'linear-gradient(135deg, #F8FAFC 0%, #EFF5F0 100%)',
        }}>
            <span style={{ fontSize: '2rem' }}>{mode === 'quick-tip' ? '⚡' : '🧠'}</span>
            <h3 style={{ marginTop: 12, marginBottom: 8 }}>
                {mode === 'quick-tip' ? 'Get a Wellness Tip' : 'AI Pattern Analysis'}
            </h3>
            <p className="text-sm text-muted" style={{ marginBottom: 16 }}>
                {mode === 'quick-tip'
                    ? 'Submit your daily log to get an instant AI-powered wellness tip.'
                    : 'Analyze your recent health data for patterns and trends.'}
            </p>
            {aiAvailable === false && (
                <p className="text-sm" style={{ color: 'var(--risk-weak)', marginBottom: 12 }}>
                    ⚠️ AI service initializing... Rule-based insights available.
                </p>
            )}
            <button
                className="btn btn-primary btn-sm"
                onClick={mode === 'quick-tip' ? fetchQuickTip : fetchDeepAnalysis}
                disabled={loading}
            >
                {mode === 'quick-tip' ? '⚡ Generate Tip' : '🧠 Run Analysis'}
            </button>
        </div>
    );
}
