import { Component } from 'react';

/**
 * React Error Boundary — catches render-phase errors and shows
 * a recovery UI instead of a blank white screen.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info?.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100dvh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          fontFamily: 'system-ui, sans-serif',
          background: '#0a0f1a',
          color: '#e0e6f0',
          textAlign: 'center',
          gap: '1rem',
        }}>
          <div style={{ fontSize: '3rem' }}>⚠️</div>
          <h2 style={{ margin: 0, color: '#ff6b6b' }}>Something went wrong</h2>
          <p style={{ margin: 0, opacity: 0.7, maxWidth: '400px', fontSize: '0.9rem' }}>
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={() => window.location.href = '/'}
            style={{
              marginTop: '0.5rem',
              padding: '0.75rem 2rem',
              background: '#0054A6',
              color: '#fff',
              border: 'none',
              borderRadius: '12px',
              fontSize: '1rem',
              cursor: 'pointer',
            }}
          >
            Go Home
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
