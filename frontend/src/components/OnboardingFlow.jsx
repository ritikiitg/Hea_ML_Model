import React, { useState } from 'react';

const CONSENT_STEPS = [
    {
        title: "Welcome to Hea",
        subtitle: "Your AI-powered health companion",
        description: "We'll help you catch health risks early by analyzing patterns in your daily inputs. No medical records needed — just honest self-reporting.",
    },
    {
        title: "How We Use Your Data",
        subtitle: "Transparency is our priority",
        points: [
            "We collect daily health inputs you voluntarily share (symptoms, mood, sleep, etc.)",
            "Our AI models analyze patterns to detect weak health signals over time",
            "Your data is stored securely with encryption and UK data residency",
            "We never share identifiable data with third parties",
            "You can export or delete your data at any time (GDPR compliant)",
        ],
    },
    {
        title: "Your Consent",
        subtitle: "You're always in control",
        consents: [
            { key: 'consent_data_storage', label: 'Store my health data securely', description: 'Required to use Hea. Data encrypted at rest.', required: true },
            { key: 'consent_ml_usage', label: 'Use AI to analyze my health patterns', description: 'Enables risk detection and personalized insights.', required: false },
            { key: 'consent_anonymized_research', label: 'Contribute anonymized data to health research', description: 'De-identified summaries only. Helps improve detection.', required: false },
            { key: 'consent_wearable_data', label: 'Connect wearable device data (optional)', description: 'If you opt in later, we can integrate step/sleep data.', required: false },
        ],
    },
];

export default function OnboardingFlow({ onComplete }) {
    const [step, setStep] = useState(0);
    const [consents, setConsents] = useState({
        consent_data_storage: false,
        consent_ml_usage: false,
        consent_anonymized_research: false,
        consent_wearable_data: false,
    });

    const current = CONSENT_STEPS[step];

    const handleConsent = (key) => {
        setConsents(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const canProceed = () => {
        if (step === 2) return consents.consent_data_storage;
        return true;
    };

    const handleNext = () => {
        if (step < CONSENT_STEPS.length - 1) {
            setStep(step + 1);
        } else {
            onComplete(consents);
        }
    };

    return (
        <div className="page-wrapper" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ maxWidth: 560, width: '100%' }}>
                {/* Step indicator */}
                <div className="steps-indicator">
                    {CONSENT_STEPS.map((_, i) => (
                        <div key={i} className={`step-dot ${i === step ? 'active' : i < step ? 'completed' : ''}`} />
                    ))}
                </div>

                <div className="card animate-in" key={step}>
                    <h2 style={{ marginBottom: 4 }}>{current.title}</h2>
                    <p className="text-muted" style={{ marginBottom: 28 }}>{current.subtitle}</p>

                    {/* Step 1: Welcome */}
                    {step === 0 && (
                        <div>
                            <p style={{ marginBottom: 20, lineHeight: 1.7 }}>{current.description}</p>
                            <div className="card-glass" style={{ padding: 16, marginBottom: 8 }}>
                                <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                                    <span style={{ fontSize: '1.5rem' }}>🔒</span>
                                    <div>
                                        <strong>Privacy First</strong>
                                        <p className="text-sm text-muted">All data encrypted & stored in the UK</p>
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                                    <span style={{ fontSize: '1.5rem' }}>🧠</span>
                                    <div>
                                        <strong>AI-Powered Insights</strong>
                                        <p className="text-sm text-muted">Detect patterns humans miss</p>
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: 12 }}>
                                    <span style={{ fontSize: '1.5rem' }}>💬</span>
                                    <div>
                                        <strong>Simple Language</strong>
                                        <p className="text-sm text-muted">No medical jargon — clear, helpful explanations</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Step 2: Data Usage */}
                    {step === 1 && (
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            {current.points.map((point, i) => (
                                <li key={i} className="animate-in" style={{ animationDelay: `${i * 0.08}s`, display: 'flex', gap: 12, marginBottom: 14, alignItems: 'flex-start' }}>
                                    <span style={{ color: 'var(--risk-low)', fontWeight: 700, fontSize: '1.1rem' }}>✓</span>
                                    <span style={{ lineHeight: 1.5 }}>{point}</span>
                                </li>
                            ))}
                        </ul>
                    )}

                    {/* Step 3: Consent Toggles */}
                    {step === 2 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                            {current.consents.map((c) => (
                                <div key={c.key}
                                    style={{
                                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        padding: 16, borderRadius: 'var(--radius-md)',
                                        border: `1.5px solid ${consents[c.key] ? 'var(--hea-primary)' : 'var(--hea-border)'}`,
                                        background: consents[c.key] ? 'var(--hea-primary-light)' : 'transparent',
                                        transition: 'all 0.2s ease',
                                        cursor: 'pointer',
                                    }}
                                    onClick={() => handleConsent(c.key)}
                                >
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <strong style={{ fontSize: '0.95rem' }}>{c.label}</strong>
                                            {c.required && <span className="badge badge-high" style={{ fontSize: '0.65rem', padding: '2px 8px' }}>Required</span>}
                                        </div>
                                        <p className="text-sm text-muted" style={{ marginTop: 4 }}>{c.description}</p>
                                    </div>
                                    <div className={`toggle-switch ${consents[c.key] ? 'active' : ''}`} />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Navigation */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 32 }}>
                        {step > 0 ? (
                            <button className="btn btn-secondary" onClick={() => setStep(step - 1)}>Back</button>
                        ) : <div />}
                        <button className="btn btn-primary" onClick={handleNext} disabled={!canProceed()}>
                            {step < CONSENT_STEPS.length - 1 ? 'Continue' : 'Get Started →'}
                        </button>
                    </div>
                </div>

                <p className="text-center text-sm text-muted" style={{ marginTop: 20 }}>
                    Hea is a wellness companion, not a medical device. Always consult a healthcare professional.
                </p>
            </div>
        </div>
    );
}
