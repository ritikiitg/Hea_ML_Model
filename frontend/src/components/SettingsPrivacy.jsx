import React, { useState } from 'react';

export default function SettingsPrivacy({ settings, onUpdate, onExport, onDelete }) {
    const [localSettings, setLocalSettings] = useState(settings || {
        consent_data_storage: true,
        consent_ml_usage: true,
        consent_anonymized_research: false,
        consent_wearable_data: false,
        data_retention_days: 365,
    });
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [saved, setSaved] = useState(false);

    const handleToggle = (key) => {
        if (key === 'consent_data_storage') return; // Can't disable required consent
        setLocalSettings(prev => ({ ...prev, [key]: !prev[key] }));
        setSaved(false);
    };

    const handleSave = () => {
        onUpdate(localSettings);
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    const consentItems = [
        { key: 'consent_data_storage', label: 'Data Storage', desc: 'Required. Your health data is stored securely with encryption.', required: true, icon: '🔒' },
        { key: 'consent_ml_usage', label: 'AI Analysis', desc: 'Allow our AI to analyze patterns in your health data.', required: false, icon: '🧠' },
        { key: 'consent_anonymized_research', label: 'Research Data', desc: 'Contribute de-identified data to health research.', required: false, icon: '🔬' },
        { key: 'consent_wearable_data', label: 'Wearable Data', desc: 'Connect wearable device to enrich health insights.', required: false, icon: '⌚' },
    ];

    return (
        <div>
            {/* Privacy & Consent */}
            <div className="card" style={{ marginBottom: 24 }}>
                <h3 style={{ marginBottom: 4 }}>🔐 Privacy & Consent</h3>
                <p className="text-sm text-muted mb-lg">Control how your data is used. Changes take effect immediately.</p>

                {consentItems.map(item => (
                    <div key={item.key} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '16px 0', borderBottom: '1px solid var(--hea-border-light)',
                    }}>
                        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', flex: 1 }}>
                            <span style={{ fontSize: '1.3rem' }}>{item.icon}</span>
                            <div>
                                <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    {item.label}
                                    {item.required && <span className="text-sm text-muted">(Required)</span>}
                                </div>
                                <p className="text-sm text-muted" style={{ marginTop: 2 }}>{item.desc}</p>
                            </div>
                        </div>
                        <div
                            className={`toggle-switch ${localSettings[item.key] ? 'active' : ''}`}
                            onClick={() => handleToggle(item.key)}
                            style={item.required ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
                        />
                    </div>
                ))}

                {/* Data Retention */}
                <div style={{ marginTop: 20 }}>
                    <label style={{ fontWeight: 600, display: 'block', marginBottom: 8 }}>
                        📅 Data Retention Period
                    </label>
                    <select
                        className="input-field"
                        value={localSettings.data_retention_days}
                        onChange={(e) => {
                            setLocalSettings(prev => ({ ...prev, data_retention_days: Number(e.target.value) }));
                            setSaved(false);
                        }}
                        style={{ maxWidth: 250 }}
                    >
                        <option value={90}>90 days</option>
                        <option value={180}>6 months</option>
                        <option value={365}>1 year</option>
                        <option value={730}>2 years</option>
                    </select>
                </div>

                <div style={{ marginTop: 24, display: 'flex', gap: 12 }}>
                    <button className="btn btn-primary" onClick={handleSave}>
                        {saved ? '✓ Saved' : 'Save Settings'}
                    </button>
                </div>
            </div>

            {/* Data Management */}
            <div className="card" style={{ marginBottom: 24 }}>
                <h3 style={{ marginBottom: 4 }}>📦 Your Data</h3>
                <p className="text-sm text-muted mb-lg">Export or delete your data anytime. GDPR compliant.</p>

                <div style={{ display: 'flex', gap: 12 }}>
                    <button className="btn btn-secondary" onClick={onExport}>
                        📥 Export All Data
                    </button>
                    <button
                        className="btn btn-danger"
                        onClick={() => setShowDeleteConfirm(true)}
                    >
                        🗑️ Delete All Data
                    </button>
                </div>

                {showDeleteConfirm && (
                    <div style={{
                        marginTop: 16, padding: 16,
                        background: 'var(--risk-high-bg)', borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--risk-high)',
                    }}>
                        <p style={{ fontWeight: 600, color: 'var(--risk-high)', marginBottom: 8 }}>
                            ⚠️ This action is permanent and cannot be undone
                        </p>
                        <p className="text-sm" style={{ marginBottom: 12 }}>
                            This will delete all your health inputs, risk assessments, feedback, and account data.
                        </p>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button className="btn btn-danger btn-sm" onClick={() => { onDelete(); setShowDeleteConfirm(false); }}>
                                Yes, Delete Everything
                            </button>
                            <button className="btn btn-secondary btn-sm" onClick={() => setShowDeleteConfirm(false)}>
                                Cancel
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* About */}
            <div className="card">
                <h3 style={{ marginBottom: 12 }}>ℹ️ About Hea</h3>
                <p className="text-sm" style={{ lineHeight: 1.7 }}>
                    Hea is a wellness companion that helps you catch health signals early. We analyze patterns in your
                    self-reported data using AI, and present findings in simple, non-clinical language.
                </p>
                <p className="text-sm text-muted" style={{ marginTop: 12, fontStyle: 'italic' }}>
                    Hea is not a medical device and does not diagnose conditions or prescribe treatment.
                    Always consult a qualified healthcare professional for medical advice.
                </p>
                <p className="text-sm text-muted" style={{ marginTop: 8 }}>Version 0.1.0 (Prototype)</p>
            </div>
        </div>
    );
}
