import React, { useState } from 'react';

const RISK_COLORS = {
    LOW: { bg: 'var(--risk-low-bg)', color: 'var(--risk-low)', badge: 'badge-low' },
    WEAK: { bg: 'var(--risk-weak-bg)', color: 'var(--risk-weak)', badge: 'badge-weak' },
    MODERATE: { bg: 'var(--risk-moderate-bg)', color: 'var(--risk-moderate)', badge: 'badge-moderate' },
    HIGH: { bg: 'var(--risk-high-bg)', color: 'var(--risk-high)', badge: 'badge-high' },
};

export default function RiskInsightCard({ assessment, onFeedback }) {
    const [showDetails, setShowDetails] = useState(false);

    if (!assessment) {
        return (
            <div className="card text-center" style={{ padding: 48 }}>
                <span style={{ fontSize: '3rem' }}>📊</span>
                <h3 style={{ marginTop: 16, marginBottom: 8 }}>No assessment yet</h3>
                <p className="text-muted">Submit your daily health log to get your first risk insight.</p>
            </div>
        );
    }

    const { risk_level, confidence_score, explanation_text, signal_details, created_at } = assessment;
    const riskStyle = RISK_COLORS[risk_level] || RISK_COLORS.LOW;
    const confidencePct = Math.round(confidence_score * 100);

    return (
        <div className="card animate-in" style={{ overflow: 'hidden' }}>
            {/* Risk Level Header */}
            <div style={{
                background: riskStyle.bg,
                margin: '-32px -32px 24px -32px',
                padding: '28px 32px',
                borderBottom: `2px solid ${riskStyle.color}`,
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <p className="text-sm text-muted" style={{ marginBottom: 4 }}>Your Risk Assessment</p>
                        <h2 style={{ color: riskStyle.color, marginBottom: 4 }}>
                            {risk_level === 'LOW' && '✅ '}
                            {risk_level === 'WEAK' && '🔍 '}
                            {risk_level === 'MODERATE' && '⚠️ '}
                            {risk_level === 'HIGH' && '🚨 '}
                            {risk_level} Risk
                        </h2>
                    </div>
                    <span className={`badge ${riskStyle.badge}`}>{risk_level}</span>
                </div>
            </div>

            {/* Confidence Score */}
            <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span className="text-sm" style={{ fontWeight: 600 }}>Confidence</span>
                    <span className="text-sm" style={{ fontWeight: 700, color: riskStyle.color }}>{confidencePct}%</span>
                </div>
                <div className="progress-bar-container">
                    <div className="progress-bar-fill" style={{
                        width: `${confidencePct}%`,
                        background: riskStyle.color,
                    }} />
                </div>
            </div>

            {/* Explanation */}
            <div style={{
                background: 'var(--hea-bg-light)',
                borderRadius: 'var(--radius-md)',
                padding: 16,
                marginBottom: 20,
                lineHeight: 1.6,
            }}>
                <p>{explanation_text}</p>
            </div>

            {/* Signal Details Toggle */}
            {signal_details && (
                <div>
                    <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => setShowDetails(!showDetails)}
                        style={{ width: '100%', justifyContent: 'space-between' }}
                    >
                        <span>📋 View Signal Details</span>
                        <span style={{ transform: showDetails ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>▼</span>
                    </button>

                    {showDetails && (
                        <div className="signal-list" style={{ marginTop: 12 }}>
                            {(signal_details.signals || []).map((signal, i) => {
                                const categoryLabels = {
                                    nlp: { icon: '💬', label: 'Symptom Analysis' },
                                    timeseries: { icon: '📈', label: 'Daily Patterns' },
                                    trained_model: { icon: '🧠', label: 'ML Model' },
                                    behavioral: { icon: '🔄', label: 'Behavioral' },
                                };
                                const cat = categoryLabels[signal.category] || { icon: '📋', label: 'Analysis' };
                                const relevance = signal.weight >= 0.7 ? 'High' : signal.weight >= 0.4 ? 'Moderate' : 'Low';
                                const relColor = signal.weight >= 0.7 ? 'var(--risk-high)' : signal.weight >= 0.4 ? 'var(--risk-moderate)' : 'var(--risk-low)';
                                return (
                                    <div key={i} className={`signal-item importance-${signal.weight >= 0.7 ? 'high' : signal.weight >= 0.4 ? 'moderate' : 'low'}`}>
                                        <span className="signal-icon">{cat.icon}</span>
                                        <div className="signal-text">
                                            <div style={{ fontWeight: 500, fontSize: '0.9rem' }}>{signal.signal}</div>
                                            <span className="signal-source">
                                                {cat.label} •{' '}
                                                <span style={{ color: relColor, fontWeight: 600 }}>
                                                    {relevance} Relevance
                                                </span>
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* Feedback Actions */}
            {onFeedback && assessment.feedback_received === 'none' && (
                <div style={{ marginTop: 20, borderTop: '1px solid var(--hea-border)', paddingTop: 16 }}>
                    <p className="text-sm text-muted mb-md">Was this assessment helpful?</p>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button className="btn btn-success btn-sm" onClick={() => onFeedback('confirm', assessment.id)}>
                            👍 Accurate
                        </button>
                        <button className="btn btn-secondary btn-sm" onClick={() => onFeedback('adjust', assessment.id)}>
                            ✏️ Partially
                        </button>
                        <button className="btn btn-danger btn-sm" onClick={() => onFeedback('reject', assessment.id)}>
                            👎 Not Relevant
                        </button>
                    </div>
                </div>
            )}

            {/* Timestamp */}
            <p className="text-sm text-muted" style={{ marginTop: 16, textAlign: 'right' }}>
                {created_at ? new Date(created_at).toLocaleString() : ''}
            </p>
        </div>
    );
}
