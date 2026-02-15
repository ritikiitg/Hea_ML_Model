/**
 * Hea API Client — Axios wrapper for backend communication.
 */

import axios from 'axios';

const API_BASE = '/api/v1';

const client = axios.create({
    baseURL: API_BASE,
    headers: { 'Content-Type': 'application/json' },
    timeout: 30000,  // 30s — trained model needs time to load on first call
});

// ─── User Endpoints ──────────────────────────────────
export const createUser = (consent, notificationPreferences = {}) =>
    client.post('/users/', { consent, notification_preferences: notificationPreferences });

export const getUser = (userId) =>
    client.get(`/users/${userId}`);

export const completeOnboarding = (userId) =>
    client.put(`/users/${userId}/complete-onboarding`);

// ─── Health Input Endpoints ──────────────────────────
export const submitHealthInput = (userId, inputData) =>
    client.post(`/inputs/?user_id=${userId}`, inputData);

export const getUserInputs = (userId, limit = 30) =>
    client.get(`/inputs/?user_id=${userId}&limit=${limit}`);

// ─── Inference Endpoints ─────────────────────────────
export const runAssessment = (userId, historyDays = 7) =>
    client.post('/assess/', { user_id: userId, include_history_days: historyDays });

export const getAssessmentHistory = (userId, limit = 10) =>
    client.get(`/assess/history?user_id=${userId}&limit=${limit}`);

export const getAssessment = (assessmentId) =>
    client.get(`/assess/${assessmentId}`);

// ─── Feedback Endpoints ──────────────────────────────
export const submitFeedback = (userId, feedbackData) =>
    client.post(`/feedback/?user_id=${userId}`, feedbackData);

export const getFeedbackStats = (userId) =>
    client.get(`/feedback/stats?user_id=${userId}`);

// ─── Privacy Endpoints ───────────────────────────────
export const getPrivacySettings = (userId) =>
    client.get(`/privacy/${userId}`);

export const updatePrivacySettings = (userId, settings) =>
    client.put(`/privacy/${userId}`, settings);

export const exportUserData = (userId) =>
    client.get(`/privacy/${userId}/export`);

export const deleteUserData = (userId) =>
    client.delete(`/privacy/${userId}?confirm=true`);

// ─── AI Insights Endpoints ───────────────────────────
export const getAIStatus = () =>
    client.get('/insights/status');

export const analyzePatterns = (userId, days = 7, context = '') =>
    client.post('/insights/analyze', { user_id: userId, include_days: days, context });

export const getQuickTip = (dailyLog) =>
    client.post('/insights/quick-tip', dailyLog);

export default client;
