import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

export default function Navbar() {
    const location = useLocation();
    const navigate = useNavigate();
    const path = location.pathname;
    const [menuOpen, setMenuOpen] = useState(false);

    return (
        <nav className="navbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 8,
                    background: 'linear-gradient(135deg, var(--hea-primary), var(--hea-secondary))',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#fff', fontWeight: 800, fontSize: '1rem',
                }}>H</div>
                <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>Hea</span>
            </div>

            {/* Desktop nav */}
            <div className="navbar-links-desktop">
                <Link to="/" className={`nav-link ${path === '/' ? 'active' : ''}`}>Home</Link>
                <Link to="/daily" className={`nav-link ${path === '/daily' ? 'active' : ''}`}>Daily Log</Link>
                <Link to="/insights" className={`nav-link ${path === '/insights' ? 'active' : ''}`}>Insights</Link>
                <Link to="/settings" className={`nav-link ${path === '/settings' ? 'active' : ''}`}>Settings</Link>
            </div>

            <button className="btn btn-primary btn-sm navbar-cta-desktop" onClick={() => navigate('/daily')}>
                Log Today
            </button>

            {/* Mobile hamburger */}
            <button
                className="navbar-hamburger"
                onClick={() => setMenuOpen(!menuOpen)}
                aria-label="Toggle menu"
            >
                <span style={{
                    display: 'block', width: 20, height: 2,
                    background: 'var(--hea-text-dark)',
                    marginBottom: 4, transition: 'all 0.2s',
                    transform: menuOpen ? 'rotate(45deg) translateY(6px)' : 'none',
                }} />
                <span style={{
                    display: 'block', width: 20, height: 2,
                    background: 'var(--hea-text-dark)',
                    marginBottom: 4, opacity: menuOpen ? 0 : 1,
                    transition: 'opacity 0.2s',
                }} />
                <span style={{
                    display: 'block', width: 20, height: 2,
                    background: 'var(--hea-text-dark)',
                    transition: 'all 0.2s',
                    transform: menuOpen ? 'rotate(-45deg) translateY(-6px)' : 'none',
                }} />
            </button>

            {/* Mobile menu */}
            {menuOpen && (
                <div className="navbar-mobile-menu">
                    <Link to="/" className={`nav-link ${path === '/' ? 'active' : ''}`} onClick={() => setMenuOpen(false)}>Home</Link>
                    <Link to="/daily" className={`nav-link ${path === '/daily' ? 'active' : ''}`} onClick={() => setMenuOpen(false)}>Daily Log</Link>
                    <Link to="/insights" className={`nav-link ${path === '/insights' ? 'active' : ''}`} onClick={() => setMenuOpen(false)}>Insights</Link>
                    <Link to="/settings" className={`nav-link ${path === '/settings' ? 'active' : ''}`} onClick={() => setMenuOpen(false)}>Settings</Link>
                    <button className="btn btn-primary btn-sm" style={{ marginTop: 8, width: '100%' }}
                        onClick={() => { setMenuOpen(false); navigate('/daily'); }}>
                        Log Today
                    </button>
                </div>
            )}
        </nav>
    );
}
