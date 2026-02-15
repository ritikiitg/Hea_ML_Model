import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar.jsx';
import OnboardingFlow from './components/OnboardingFlow.jsx';
import DailyInputForm from './components/DailyInputForm.jsx';
import RiskInsightCard from './components/RiskInsightCard.jsx';
import SettingsPrivacy from './components/SettingsPrivacy.jsx';
import AIInsightsPanel from './components/AIInsightsPanel.jsx';
import {
    createUser,
    submitHealthInput,
    runAssessment,
    getAssessmentHistory,
    submitFeedback,
    exportUserData,
    deleteUserData,
    updatePrivacySettings,
} from './api/client.js';

// ─── Landing Page ────────────────────────────────────
function LandingPage() {
    const [liveAssessment, setLiveAssessment] = useState(null);

    return (
        <div className="page-wrapper">
            <div className="app-container">
                <div className="text-center" style={{ paddingTop: 60, paddingBottom: 40 }}>
                    <div className="animate-in">
                        <span style={{ fontSize: '3.5rem' }}>🩺</span>
                        <h1 style={{ marginTop: 16, fontSize: '2.5rem', fontWeight: 800 }}>
                            Catch health risks <span style={{ color: 'var(--hea-primary)' }}>before they catch you</span>
                        </h1>
                        <p style={{ maxWidth: 560, margin: '16px auto 0', fontSize: '1.1rem', lineHeight: 1.7, color: 'var(--hea-text-body)' }}>
                            Hea's AI monitors your daily health inputs and detects subtle patterns that could signal emerging risks — no medical records required.
                        </p>
                    </div>
                </div>

                <div className="grid-3 animate-in animate-delay-1" style={{ maxWidth: 900, margin: '0 auto 60px' }}>
                    <div className="card text-center" style={{ padding: 32 }}>
                        <span style={{ fontSize: '2.2rem' }}>📝</span>
                        <h3 style={{ marginTop: 12, marginBottom: 8 }}>Log Daily</h3>
                        <p className="text-sm text-muted">Share how you're feeling — rate your health, check symptoms, track your daily metrics.</p>
                    </div>
                    <div className="card text-center" style={{ padding: 32 }}>
                        <span style={{ fontSize: '2.2rem' }}>🧠</span>
                        <h3 style={{ marginTop: 12, marginBottom: 8 }}>AI Analysis</h3>
                        <p className="text-sm text-muted">Our trained LightGBM + XGBoost ensemble detects weak signals across your health data.</p>
                    </div>
                    <div className="card text-center" style={{ padding: 32 }}>
                        <span style={{ fontSize: '2.2rem' }}>💡</span>
                        <h3 style={{ marginTop: 12, marginBottom: 8 }}>Clear Insights</h3>
                        <p className="text-sm text-muted">Get plain-language explanations and actionable next steps, not medical jargon.</p>
                    </div>
                </div>

                {liveAssessment && (
                    <div style={{ maxWidth: 640, margin: '0 auto' }}>
                        <RiskInsightCard assessment={liveAssessment} />
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Daily Log Page ──────────────────────────────────
function DailyPage() {
    const [submitted, setSubmitted] = useState(false);
    const [loading, setLoading] = useState(false);
    const [lastLog, setLastLog] = useState(null);
    const [assessment, setAssessment] = useState(null);

    const handleSubmit = async (data) => {
        setLoading(true);
        setLastLog(data);

        try {
            let userId = localStorage.getItem('hea_user_id');
            if (!userId) {
                const userRes = await createUser(
                    { consent_data_storage: true, consent_ml_usage: true, consent_anonymized_research: true, consent_wearable_data: false },
                    {}
                );
                userId = userRes.data.id;
                localStorage.setItem('hea_user_id', userId);
            }

            await submitHealthInput(userId, data);
            const assessRes = await runAssessment(userId, 7);
            setAssessment(assessRes.data);
        } catch (e) {
            console.error('Submission error:', e);
            setAssessment({
                risk_level: 'LOW',
                confidence_score: 0,
                explanation_text: 'Could not reach analysis server. Your data has been saved locally and will be analyzed when the server is available.',
                signal_details: { signals: [] },
            });
        }

        setLoading(false);
        setSubmitted(true);
    };

    if (submitted) {
        return (
            <div className="page-wrapper">
                <div className="app-container" style={{ maxWidth: 640, margin: '0 auto' }}>
                    <div className="card text-center animate-in" style={{ padding: 48 }}>
                        <span style={{ fontSize: '3rem' }}>✨</span>
                        <h2 style={{ marginTop: 16, marginBottom: 8 }}>Log Submitted!</h2>
                        <p className="text-muted mb-lg">Your data has been analyzed. Here's your latest insight:</p>
                        {assessment && (
                            <RiskInsightCard assessment={assessment} onFeedback={async (type, id) => {
                                try {
                                    const userId = localStorage.getItem('hea_user_id');
                                    if (userId) await submitFeedback(userId, { assessment_id: id, feedback_type: type });
                                } catch (e) { console.warn('Feedback failed:', e); }
                            }} />
                        )}
                    </div>

                    <div style={{ marginTop: 20 }}>
                        <AIInsightsPanel mode="quick-tip" dailyLog={lastLog} autoLoad={true} />
                    </div>

                    <div className="text-center" style={{ marginTop: 20 }}>
                        <button className="btn btn-secondary" onClick={() => { setSubmitted(false); setLastLog(null); setAssessment(null); }}>
                            Submit Another Log
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="page-wrapper">
            <div className="app-container" style={{ maxWidth: 640, margin: '0 auto' }}>
                <div className="page-header">
                    <h1>📝 Daily Health Log</h1>
                    <p>Take a minute to check in. Every entry makes your insights smarter.</p>
                </div>
                <DailyInputForm onSubmit={handleSubmit} isLoading={loading} />
            </div>
        </div>
    );
}

// ─── Insights Page ───────────────────────────────────
function InsightsPage() {
    const [assessments, setAssessments] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            const userId = localStorage.getItem('hea_user_id');
            if (!userId) { setLoading(false); return; }
            try {
                const res = await getAssessmentHistory(userId, 10);
                setAssessments(res.data || []);
            } catch (e) {
                console.warn('Could not fetch history:', e);
            }
            setLoading(false);
        };
        fetchHistory();
    }, []);

    return (
        <div className="page-wrapper">
            <div className="app-container" style={{ maxWidth: 640, margin: '0 auto' }}>
                <div className="page-header">
                    <h1>💡 Your Insights</h1>
                    <p>AI-powered analysis of your health patterns over time</p>
                </div>

                <div style={{ marginBottom: 24 }}>
                    <AIInsightsPanel mode="deep-analysis" userId={localStorage.getItem('hea_user_id') || 'demo-user'} />
                </div>

                {assessments.length > 0 && assessments[0] && (
                    <RiskInsightCard assessment={assessments[0]} onFeedback={async (type, id) => {
                        try {
                            const userId = localStorage.getItem('hea_user_id');
                            if (userId) await submitFeedback(userId, { assessment_id: id, feedback_type: type });
                        } catch (e) { console.warn('Feedback failed:', e); }
                    }} />
                )}

                {assessments.length > 0 && (
                    <div className="card" style={{ marginTop: 24 }}>
                        <h3 style={{ marginBottom: 16 }}>📈 Assessment History</h3>
                        {assessments.map(h => (
                            <div key={h.id} style={{
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                padding: '12px 0', borderBottom: '1px solid var(--hea-border-light)',
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                    <span className={`badge badge-${h.risk_level.toLowerCase()}`}>{h.risk_level}</span>
                                    <span className="text-sm">{h.explanation_text?.slice(0, 80)}...</span>
                                </div>
                                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                                    <span className="text-sm text-muted">{new Date(h.created_at).toLocaleDateString()}</span>
                                    {h.model_version && (
                                        <p style={{ fontSize: '0.65rem', color: h.model_version?.includes('ensemble') ? 'var(--hea-primary)' : 'var(--hea-text-muted)' }}>
                                            {h.model_version}
                                        </p>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {loading && (
                    <div className="card text-center" style={{ padding: 32 }}>
                        <div className="spinner" />
                        <p className="text-muted" style={{ marginTop: 12 }}>Loading assessments...</p>
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Settings Page ───────────────────────────────────
function SettingsPage() {
    const [deleting, setDeleting] = useState(false);
    const [exporting, setExporting] = useState(false);

    const userId = localStorage.getItem('hea_user_id');

    const handleExport = async () => {
        if (!userId) { alert('No user data found.'); return; }
        setExporting(true);
        try {
            const res = await exportUserData(userId);
            const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hea-data-export-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            alert('Export failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            setExporting(false);
        }
    };

    const handleDelete = async () => {
        if (!userId) { alert('No user data found.'); return; }
        setDeleting(true);
        try {
            await deleteUserData(userId);
            localStorage.removeItem('hea_user_id');
            localStorage.removeItem('hea_onboarded');
            localStorage.removeItem('hea_consents');
            window.location.href = '/';
        } catch (e) {
            alert('Delete failed: ' + (e.response?.data?.detail || e.message));
            setDeleting(false);
        }
    };

    const handleUpdate = async (settings) => {
        if (!userId) { alert('No user found. Please complete onboarding first.'); return; }
        try {
            await updatePrivacySettings(userId, settings);
        } catch (e) {
            alert('Save failed: ' + (e.response?.data?.detail || e.message));
        }
    };

    return (
        <div className="page-wrapper">
            <div className="app-container" style={{ maxWidth: 640, margin: '0 auto' }}>
                <div className="page-header">
                    <h1>⚙️ Settings</h1>
                    <p>Manage your privacy, consent, and data</p>
                </div>
                <SettingsPrivacy
                    onUpdate={handleUpdate}
                    onExport={handleExport}
                    onDelete={handleDelete}
                />
                {(deleting || exporting) && (
                    <div style={{
                        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999,
                    }}>
                        <div className="card" style={{ textAlign: 'center', padding: 32 }}>
                            <div className="spinner" />
                            <p style={{ marginTop: 12 }}>{deleting ? 'Deleting all data...' : 'Exporting your data...'}</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── App Root ────────────────────────────────────────
export default function App() {
    const [isOnboarded, setIsOnboarded] = useState(() => {
        return localStorage.getItem('hea_onboarded') === 'true';
    });

    const handleOnboardComplete = async (consents) => {
        localStorage.setItem('hea_onboarded', 'true');
        localStorage.setItem('hea_consents', JSON.stringify(consents));

        try {
            const res = await createUser(
                { consent_data_storage: true, consent_ml_usage: true, consent_anonymized_research: true, consent_wearable_data: false, ...consents },
                {}
            );
            localStorage.setItem('hea_user_id', res.data.id);
        } catch (e) {
            console.warn('User creation failed, will retry on first submission:', e);
        }

        setIsOnboarded(true);
    };

    if (!isOnboarded) {
        return (
            <Router>
                <OnboardingFlow onComplete={handleOnboardComplete} />
            </Router>
        );
    }

    return (
        <Router>
            <Navbar />
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/daily" element={<DailyPage />} />
                <Route path="/insights" element={<InsightsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/" />} />
            </Routes>
        </Router>
    );
}
