import React, { useState } from 'react';

const SYMPTOM_OPTIONS = [
    { key: 'headache', label: '🤕 Headache', category: 'pain' },
    { key: 'fatigue', label: '😴 Fatigue', category: 'energy' },
    { key: 'chest_pain', label: '💔 Chest pain', category: 'cardiac' },
    { key: 'shortness_of_breath', label: '😮‍💨 Shortness of breath', category: 'cardiac' },
    { key: 'dizziness', label: '😵 Dizziness', category: 'neuro' },
    { key: 'nausea', label: '🤢 Nausea', category: 'digestive' },
    { key: 'back_pain', label: '🔙 Back pain', category: 'pain' },
    { key: 'joint_pain', label: '🦴 Joint pain', category: 'pain' },
    { key: 'muscle_weakness', label: '💪 Muscle weakness', category: 'musculo' },
    { key: 'vision_changes', label: '👁️ Vision changes', category: 'neuro' },
];

const DEPRESSION_ITEMS = [
    { key: 'felt_depressed', label: 'Felt depressed or down' },
    { key: 'everything_effort', label: 'Everything felt like an effort' },
    { key: 'sleep_restless', label: 'Sleep was restless' },
    { key: 'not_happy', label: 'Could not get going' },
    { key: 'felt_lonely', label: 'Felt lonely' },
    { key: 'people_unfriendly', label: 'People were unfriendly' },
    { key: 'felt_sad', label: 'Felt sad' },
    { key: 'felt_happy', label: 'Felt happy (positive indicator)' },
];

const ACTIVITY_LIMITATIONS = [
    { key: 'walking_difficulty', label: 'Difficulty walking several blocks' },
    { key: 'climbing_stairs', label: 'Trouble climbing stairs' },
    { key: 'dressing', label: 'Difficulty dressing' },
    { key: 'bathing', label: 'Difficulty bathing' },
    { key: 'getting_in_out_bed', label: 'Difficulty getting in/out of bed' },
    { key: 'heavy_housework', label: 'Unable to do heavy housework' },
    { key: 'shopping', label: 'Difficulty shopping for groceries' },
    { key: 'preparing_meals', label: 'Difficulty preparing meals' },
];

const HEALTH_RATINGS = [
    { value: 1, label: 'Excellent', emoji: '🌟', color: '#10b981' },
    { value: 2, label: 'Very Good', emoji: '😊', color: '#34d399' },
    { value: 3, label: 'Good', emoji: '🙂', color: '#6366f1' },
    { value: 4, label: 'Fair', emoji: '😐', color: '#f59e0b' },
    { value: 5, label: 'Poor', emoji: '😟', color: '#ef4444' },
];

export default function DailyInputForm({ onSubmit, isLoading }) {
    const [symptomText, setSymptomText] = useState('');
    const [selectedSymptoms, setSelectedSymptoms] = useState([]);
    const [selfRatedHealth, setSelfRatedHealth] = useState(null);
    const [depressionChecks, setDepressionChecks] = useState([]);
    const [activityLimitations, setActivityLimitations] = useState([]);
    const [painFrequency, setPainFrequency] = useState(0);
    const [metrics, setMetrics] = useState({
        sleep_hours: 7,
        mood_score: 5,
        energy_level: 5,
        stress_level: 5,
        steps_count: 5000,
        water_intake_ml: 1500,
    });

    const toggleSymptom = (key) => {
        setSelectedSymptoms(prev =>
            prev.includes(key) ? prev.filter(s => s !== key) : [...prev, key]
        );
    };

    const toggleDepression = (key) => {
        setDepressionChecks(prev =>
            prev.includes(key) ? prev.filter(s => s !== key) : [...prev, key]
        );
    };

    const toggleActivity = (key) => {
        setActivityLimitations(prev =>
            prev.includes(key) ? prev.filter(s => s !== key) : [...prev, key]
        );
    };

    const [metricsEdited, setMetricsEdited] = useState(false);
    const [coreMetricEdited, setCoreMetricEdited] = useState(false);

    const updateMetric = (key, value) => {
        setMetrics(prev => ({ ...prev, [key]: Number(value) }));
        setMetricsEdited(true);
        if (['sleep_hours', 'mood_score', 'energy_level', 'stress_level'].includes(key)) {
            setCoreMetricEdited(true);
        }
    };

    // 3-step validation — ALL required
    const step1Done = selfRatedHealth !== null;
    const step2Done = selectedSymptoms.length > 0 || depressionChecks.length > 0
        || activityLimitations.length > 0 || symptomText.trim().length > 5;
    const step3Done = coreMetricEdited;
    const isValid = step1Done && step2Done && step3Done;

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!isValid) return;

        // Build enriched text with structured data
        const enrichedParts = [];
        if (selfRatedHealth) {
            const label = HEALTH_RATINGS.find(h => h.value === selfRatedHealth)?.label || '';
            enrichedParts.push(`Self-rated health: ${label}`);
        }
        if (depressionChecks.length > 0) {
            enrichedParts.push(`Depression indicators: ${depressionChecks.join(', ')}`);
        }
        if (painFrequency > 0) {
            enrichedParts.push(`Pain frequency: ${painFrequency === 1 ? 'some days' : 'most/every day'}`);
        }
        if (activityLimitations.length > 0) {
            enrichedParts.push(`Activity limitations: ${activityLimitations.join(', ')}`);
        }
        if (symptomText.trim()) {
            enrichedParts.push(symptomText.trim());
        }

        onSubmit({
            symptom_text: enrichedParts.filter(Boolean).join(' ') || null,
            emoji_inputs: [],
            checkbox_selections: selectedSymptoms,
            daily_metrics: metrics,
            input_source: 'web',
        });
    };

    // Calculate depression score for visual feedback
    const cesdScore = depressionChecks.filter(k => k !== 'felt_happy').length +
        (depressionChecks.includes('felt_happy') ? 0 : 1);

    return (
        <form onSubmit={handleSubmit}>
            {/* SECTION 1: Self-Rated Health */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <h3>🩺 How would you rate your overall health today?</h3>
                    <span style={{
                        fontSize: '0.65rem', padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(99,102,241,0.1)', color: '#6366f1',
                        fontWeight: 600, whiteSpace: 'nowrap',
                    }}>KEY SIGNAL</span>
                </div>
                <p className="text-sm text-muted mb-md">This is the #1 predictor in our model — your own perception matters most.</p>
                <div style={{ display: 'flex', gap: 8 }}>
                    {HEALTH_RATINGS.map(rating => (
                        <button key={rating.value} type="button" onClick={() => setSelfRatedHealth(rating.value)}
                            style={{
                                flex: 1, padding: '12px 4px', borderRadius: 'var(--radius-md)', cursor: 'pointer',
                                border: selfRatedHealth === rating.value ? `2px solid ${rating.color}` : '2px solid var(--hea-border-light)',
                                background: selfRatedHealth === rating.value ? `${rating.color}15` : 'var(--hea-bg-light)',
                                transition: 'all 0.15s',
                            }}>
                            <div style={{ fontSize: '1.5rem' }}>{rating.emoji}</div>
                            <div style={{ fontSize: '0.7rem', fontWeight: 600, marginTop: 4, color: selfRatedHealth === rating.value ? rating.color : 'inherit' }}>
                                {rating.label}
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* SECTION 2: Depression Screening */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <h3>🧠 Mental Wellbeing Check</h3>
                    <span style={{
                        fontSize: '0.65rem', padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(139,92,246,0.1)', color: '#8b5cf6',
                        fontWeight: 600, whiteSpace: 'nowrap',
                    }}>KEY SIGNAL</span>
                </div>
                <p className="text-sm text-muted mb-md">Check any that applied in the past week. These map to validated depression screening scales.</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {DEPRESSION_ITEMS.map(item => (
                        <label key={item.key} style={{
                            display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px',
                            borderRadius: 'var(--radius-md)', cursor: 'pointer',
                            border: depressionChecks.includes(item.key) ? '2px solid var(--hea-primary)' : '2px solid var(--hea-border-light)',
                            background: depressionChecks.includes(item.key) ? 'rgba(99,102,241,0.05)' : 'transparent',
                            fontSize: '0.85rem', transition: 'all 0.15s',
                        }}>
                            <input type="checkbox" checked={depressionChecks.includes(item.key)} onChange={() => toggleDepression(item.key)}
                                style={{ accentColor: 'var(--hea-primary)' }} />
                            {item.label}
                        </label>
                    ))}
                </div>
                {depressionChecks.length > 0 && (
                    <div style={{
                        marginTop: 12, padding: '8px 12px', borderRadius: 'var(--radius-md)',
                        background: cesdScore >= 4 ? 'var(--risk-moderate-bg)' : 'var(--risk-low-bg)',
                        fontSize: '0.8rem', color: cesdScore >= 4 ? 'var(--risk-moderate)' : 'var(--risk-low)',
                    }}>
                        Depression screening score: {cesdScore}/8 — {cesdScore >= 4 ? 'Worth discussing with a professional' : 'Within normal range'}
                    </div>
                )}
            </div>

            {/* SECTION 3: Pain & Symptoms */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <h3 style={{ marginBottom: 4 }}>🩹 Pain & Symptoms</h3>
                <p className="text-sm text-muted mb-md">How often have you experienced pain recently?</p>
                <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                    {[{ v: 0, l: 'Never' }, { v: 1, l: 'Some days' }, { v: 2, l: 'Most/every day' }].map(opt => (
                        <button key={opt.v} type="button" onClick={() => setPainFrequency(opt.v)}
                            style={{
                                flex: 1, padding: '10px', borderRadius: 'var(--radius-md)', cursor: 'pointer',
                                border: painFrequency === opt.v ? '2px solid var(--hea-primary)' : '2px solid var(--hea-border-light)',
                                background: painFrequency === opt.v ? 'rgba(99,102,241,0.05)' : 'transparent',
                                fontWeight: painFrequency === opt.v ? 600 : 400, fontSize: '0.85rem',
                            }}>
                            {opt.l}
                        </button>
                    ))}
                </div>

                <p className="text-sm text-muted mb-sm">Select any symptoms you're experiencing:</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {SYMPTOM_OPTIONS.map(s => (
                        <button key={s.key} type="button" onClick={() => toggleSymptom(s.key)}
                            className={`btn btn-sm ${selectedSymptoms.includes(s.key) ? 'btn-primary' : 'btn-secondary'}`}
                            style={{ fontSize: '0.8rem' }}>
                            {s.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* SECTION 4: Activity Limitations */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <h3 style={{ marginBottom: 4 }}>🚶 Activity Limitations</h3>
                <p className="text-sm text-muted mb-md">Check any activities you have difficulty with due to health:</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {ACTIVITY_LIMITATIONS.map(item => (
                        <label key={item.key} style={{
                            display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px',
                            borderRadius: 'var(--radius-md)', cursor: 'pointer',
                            border: activityLimitations.includes(item.key) ? '2px solid #f59e0b' : '2px solid var(--hea-border-light)',
                            background: activityLimitations.includes(item.key) ? 'rgba(245,158,11,0.05)' : 'transparent',
                            fontSize: '0.85rem', transition: 'all 0.15s',
                        }}>
                            <input type="checkbox" checked={activityLimitations.includes(item.key)} onChange={() => toggleActivity(item.key)}
                                style={{ accentColor: '#f59e0b' }} />
                            {item.label}
                        </label>
                    ))}
                </div>
            </div>

            {/* SECTION 5: Daily Metrics */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <h3>📊 Daily Metrics</h3>
                    <span style={{
                        fontSize: '0.65rem', padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(16,185,129,0.1)', color: '#10b981',
                        fontWeight: 600, whiteSpace: 'nowrap',
                    }}>REQUIRED</span>
                </div>
                <p className="text-sm text-muted mb-md">Adjust sliders to reflect your day. At least one health slider (sleep, mood, energy, or stress) is required.</p>

                {[
                    { key: 'sleep_hours', label: '😴 Sleep', min: 0, max: 14, step: 0.5, unit: 'hours', icon: true },
                    { key: 'mood_score', label: '😊 Mood', min: 1, max: 10, step: 1, unit: '/10' },
                    { key: 'energy_level', label: '⚡ Energy', min: 1, max: 10, step: 1, unit: '/10' },
                    { key: 'stress_level', label: '😰 Stress', min: 1, max: 10, step: 1, unit: '/10' },
                ].map(m => (
                    <div key={m.key} style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>{m.label}</label>
                            <span className="text-sm text-muted">{metrics[m.key]}{m.unit}</span>
                        </div>
                        <input type="range" min={m.min} max={m.max} step={m.step} value={metrics[m.key]}
                            onChange={(e) => updateMetric(m.key, e.target.value)}
                            style={{ width: '100%', accentColor: 'var(--hea-primary)' }} />
                    </div>
                ))}

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>
                    <div>
                        <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: 4 }}>👟 Steps</label>
                        <input type="number" className="input-field" value={metrics.steps_count}
                            onChange={(e) => updateMetric('steps_count', e.target.value)} min="0" max="50000" />
                    </div>
                    <div>
                        <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: 4 }}>💧 Water (ml)</label>
                        <input type="number" className="input-field" value={metrics.water_intake_ml}
                            onChange={(e) => updateMetric('water_intake_ml', e.target.value)} min="0" max="5000" />
                    </div>
                </div>
            </div>

            {/* SECTION 6: Free Text */}
            <div className="card animate-in" style={{ marginBottom: 24 }}>
                <h3 style={{ marginBottom: 4 }}>💬 How are you feeling?</h3>
                <p className="text-sm text-muted mb-md">Describe anything else in your own words (optional, but helps our NLP analysis).</p>
                <textarea className="input-field" placeholder="e.g., I've had a persistent headache for 3 days and I'm feeling more tired than usual..."
                    value={symptomText} onChange={(e) => setSymptomText(e.target.value.slice(0, 5000))}
                    rows={4} style={{ resize: 'vertical', minHeight: 80 }} />
                <p className="text-sm text-muted" style={{ marginTop: 4, textAlign: 'right' }}>
                    {symptomText.length}/5000
                </p>
            </div>

            {/* Validation message */}
            {!isValid && (
                <div style={{
                    padding: '12px 16px', borderRadius: 'var(--radius-md)',
                    background: 'var(--risk-weak-bg)', color: 'var(--risk-weak)',
                    fontSize: '0.85rem', marginBottom: 12, textAlign: 'center',
                }}>
                    ⚠️ {!step1Done
                        ? 'Step 1: Please rate your overall health above — it\'s the #1 predictor in our model.'
                        : !step2Done
                            ? 'Step 2: Please check at least one symptom, mental wellbeing item, or describe how you feel.'
                            : 'Step 3: Please adjust at least one health slider (sleep, mood, energy, or stress).'
                    }
                </div>
            )}

            {/* Submit */}
            <button
                type="submit"
                className="btn btn-primary btn-lg"
                style={{ width: '100%', opacity: isValid ? 1 : 0.5 }}
                disabled={isLoading || !isValid}
            >
                {isLoading ? (
                    <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                        <div className="spinner" style={{ width: 18, height: 18 }} />
                        Analyzing with ML model...
                    </span>
                ) : (
                    '🔬 Submit & Get Risk Assessment'
                )}
            </button>
        </form>
    );
}
