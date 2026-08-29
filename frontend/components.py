"""frontend/components.py — Simple, clean, professional UI components for ControlPlane.ai."""

from __future__ import annotations
import plotly.graph_objects as go
import streamlit as st


def render_header(env: str = "Production") -> None:
    """Minimalist, clean header bar."""
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-header-left">
            <div class="app-logo">🛡️ ControlPlane.ai</div>
            <span class="app-tag">Enterprise Gateway</span>
          </div>
          <div class="app-header-right">
            <div class="status-pill">
              <span class="status-dot"></span> {env} Gateway Active
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(decision: str, risk: float, policy: str, latency_ms: float, reason: str = "") -> None:
    """Clean, high-visibility decision banner."""
    d_upper = decision.upper()
    css_class = "allow" if d_upper == "ALLOW" else ("block" if d_upper == "BLOCK" else ("review" if d_upper == "HUMAN_REVIEW" else "modify"))

    reason_str = f" · {reason}" if reason else ""
    st.markdown(
        f"""
        <div class="decision-banner {css_class}">
          <div style="display: flex; align-items: center; gap: 0.8rem;">
            <span class="decision-badge {css_class}">{d_upper}</span>
            <span><strong>Rule:</strong> <span class="cp-mono">{policy}</span>{reason_str}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 1rem; color: var(--text-muted); font-size: 0.82rem;">
            <span>Risk: <strong style="color: var(--text-main); font-family: 'JetBrains Mono';">{risk:.2f}</strong></span>
            <span>Latency: <strong style="color: var(--text-main); font-family: 'JetBrains Mono';">{latency_ms:.1f}ms</strong></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_strip(detect: str, risk: float, policy: str, decision: str) -> None:
    """Compact 4-stage pipeline summary."""
    d_upper = decision.upper()
    st.markdown(
        f"""
        <div class="pipeline-strip">
          <div class="pipeline-box">
            <div class="pipeline-box-label">Detect</div>
            <div class="pipeline-box-value">{detect}</div>
          </div>
          <div class="pipeline-box">
            <div class="pipeline-box-label">Risk</div>
            <div class="pipeline-box-value">{risk:.2f}</div>
          </div>
          <div class="pipeline-box">
            <div class="pipeline-box-label">Policy</div>
            <div class="pipeline-box-value">{policy}</div>
          </div>
          <div class="pipeline-box">
            <div class="pipeline-box-label">Decide</div>
            <div class="pipeline-box-value" style="color: {'#10B981' if d_upper == 'ALLOW' else ('#EF4444' if d_upper == 'BLOCK' else '#F59E0B')};">{d_upper}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_distribution_chart(allowed: int, blocked: int, modified: int, review: int) -> None:
    """Clean Plotly bar chart for request distribution."""
    categories = ["Allow", "Block", "Modify", "Human Review"]
    counts = [allowed, blocked, modified, review]
    colors = ["#10B981", "#EF4444", "#6366F1", "#F59E0B"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=counts,
            marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
            text=counts,
            textposition="auto",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(color="#9CA3AF", size=11),
            showgrid=False,
            showline=True,
            linecolor="#1F2937",
        ),
        yaxis=dict(
            tickfont=dict(color="#9CA3AF", size=10),
            showgrid=True,
            gridcolor="#1F2937",
            showline=False,
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_latency_sparkline(latencies: list[float]) -> None:
    """Clean latency trend line."""
    y_vals = latencies if latencies else [1.5, 2.0, 1.8, 2.3, 1.9, 2.1]
    fig = go.Figure(
        go.Scatter(
            y=y_vals,
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=4, color="#3B82F6"),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(
            tickfont=dict(color="#9CA3AF", size=10),
            ticksuffix="ms",
            showgrid=True,
            gridcolor="#1F2937",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
