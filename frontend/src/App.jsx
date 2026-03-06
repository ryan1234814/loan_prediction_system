import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import {
  CheckCircle, XCircle, Info, Loader2, CreditCard, User, Landmark, HelpCircle
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:5001';

function App() {
  const [metadata, setMetadata] = useState(null);
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetadata();
  }, []);

  const fetchMetadata = async () => {
    try {
      const res = await axios.get(`${API_BASE}/metadata`);
      setMetadata(res.data);
      // Initialize form data with defaults
      const initialData = {};
      res.data.num_cols.forEach(col => initialData[col] = '');
      res.data.cat_cols.forEach(col => initialData[col] = res.data.cat_options[col][0]);
      setFormData(initialData);
    } catch (err) {
      setError("Failed to connect to backend server. Make sure the Flask app is running on port 5001.");
      console.error(err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Process numeric values
      const processedData = { ...formData };
      metadata.num_cols.forEach(col => {
        processedData[col] = parseFloat(processedData[col]) || 0;
      });

      const res = await axios.post(`${API_BASE}/predict`, processedData);
      setResult(res.data[0]);
    } catch (err) {
      setError("Prediction failed. Please check your inputs.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatExplanationData = () => {
    if (!result || !result.explanation) return [];
    return Object.entries(result.explanation)
      .map(([name, value]) => ({ name, value: parseFloat(value.toFixed(4)) }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  };

  if (!metadata && !error) {
    return (
      <div className="container loader">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="container">
      <h1>AI Loan Prediction System</h1>

      {error && (
        <div className="glass-card" style={{ borderColor: 'var(--error)', marginBottom: '2rem' }}>
          <p style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <XCircle size={20} /> {error}
          </p>
        </div>
      )}

      <div className="grid">
        {/* Form Section */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <User size={24} color="var(--primary)" />
            <h2 style={{ margin: 0 }}>Applicant Details</h2>
          </div>

          <form onSubmit={handleSubmit}>
            {metadata && (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {metadata.cat_cols.map(col => (
                    <div key={col} className="form-group">
                      <label>{col.replace('_', ' ')}</label>
                      <select name={col} value={formData[col]} onChange={handleInputChange}>
                        {metadata.cat_options[col].map(opt => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {metadata.num_cols.map(col => (
                    <div key={col} className="form-group">
                      <label>{col.replace('_', ' ')}</label>
                      <input
                        type="number"
                        name={col}
                        value={formData[col]}
                        onChange={handleInputChange}
                        placeholder={`Enter ${col.replace('_', ' ')}`}
                        required
                      />
                    </div>
                  ))}
                </div>
              </>
            )}

            <button type="submit" disabled={loading}>
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                  <Loader2 className="animate-spin" /> Analyzing...
                </span>
              ) : 'Submit for Prediction'}
            </button>
          </form>
        </div>

        {/* Result Section */}
        <div className="glass-card result-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', justifyContent: 'center' }}>
            <Landmark size={24} color="white" />
            <h2 style={{ margin: 0 }}>Prediction Result</h2>
          </div>

          {!result && !loading && (
            <div style={{ padding: '3rem 0', opacity: 0.5 }}>
              <HelpCircle size={64} style={{ marginBottom: '1rem' }} />
              <p>Enter details and submit to see the prediction</p>
            </div>
          )}

          {loading && (
            <div className="loader">
              <div className="spinner"></div>
            </div>
          )}

          {result && (
            <div className="fade-in">
              <p style={{ color: 'var(--text-secondary)' }}>Status</p>
              <div className={`status-badge status-${result.prediction}`}>
                {result.prediction === 'Approved' ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <CheckCircle size={28} /> {result.prediction.toUpperCase()}
                  </span>
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <XCircle size={28} /> {result.prediction.toUpperCase()}
                  </span>
                )}
              </div>

              <div style={{ marginTop: '1.5rem' }}>
                <p style={{ color: 'var(--text-secondary)' }}>Approval Probability</p>
                <p style={{ fontSize: '2rem', fontWeight: 800, margin: '0.5rem 0' }}>
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>

              <div className="feature-impact">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                  <Info size={18} color="var(--primary)" />
                  <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Explainability AI (SHAP Analysis)</h3>
                </div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem', textAlign: 'left' }}>
                  The chart below shows how each factor contributed to the prediction.
                  Positive values (green) increased the chance of approval, while negative values (red) decreased it.
                </p>

                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={formatExplanationData()}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
                      <XAxis type="number" stroke="var(--text-secondary)" />
                      <YAxis type="category" dataKey="name" stroke="var(--text-secondary)" fontSize={12} width={100} />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                        itemStyle={{ color: 'white' }}
                      />
                      <ReferenceLine x={0} stroke="white" strokeWidth={1} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {formatExplanationData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.value > 0 ? 'var(--success)' : 'var(--error)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <footer style={{ textAlign: 'center', marginTop: '4rem', padding: '2rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        <p>© 2026 AI Loan Intelligence | Responsible AI & Transparency</p>
      </footer>
    </div>
  );
}

export default App;
