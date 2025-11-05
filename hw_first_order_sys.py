"""Streamlit port of the 20251103 Mustafa homework interactive demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st


@dataclass(frozen=True)
class SimulationConstants:
    """Container for the plant, controller, and integration settings."""

    A: float = -1.2
    B1: float = 1.0
    B2: float = 0.35
    C: float = 1.0
    kp: float = 1.8
    t0: float = 0.0
    tf: float = 8.0
    dt: float = 0.01


DEFAULT_CONSTANTS = SimulationConstants()


def compute_response(
    constants: SimulationConstants,
    *,
    magnitude: float,
    omega: float,
    disturbance: float,
    x0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-Euler integration matching the original HTML app."""
    time = np.arange(constants.t0, constants.tf + constants.dt, constants.dt)
    x_traj = np.empty_like(time)
    y_traj = np.empty_like(time)

    x_state = x0
    for idx, t in enumerate(time):
        reference = magnitude * np.cos(omega * t)
        forcing = constants.B1 * constants.kp * reference + constants.B2 * disturbance
        dx = constants.A * x_state + forcing
        x_state = x_state + dx * constants.dt
        x_traj[idx] = x_state
        y_traj[idx] = constants.C * x_state

    return time, x_traj, y_traj


def render_plot(time: np.ndarray, y: np.ndarray) -> go.Figure:
    """Create a Plotly figure mirroring the styling of the HTML original."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=y,
            mode="lines",
            line=dict(width=3, color="#3730a3", shape="spline"),
            hovertemplate="<b>t:</b> %{x:.2f} s<br><b>y(t):</b> %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=60, r=18, t=28, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.92)",
        font=dict(family="Inter, sans-serif", color="#111827"),
        xaxis=dict(title="Time (s)", gridcolor="rgba(99,102,241,0.15)", zeroline=False),
        yaxis=dict(title="Output y(t)", gridcolor="rgba(99,102,241,0.15)", zeroline=False),
        height=420,
        showlegend=False,
    )

    return fig


def configure_page() -> None:
    """Page config plus light-touch CSS to stay close to the HTML look."""
    st.set_page_config(
        page_title="First-Order Scalar System Demo",
        page_icon="📈",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: #f5f7fb;
                color: #1f2937;
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            .exercise-card {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(45, 212, 191, 0.14));
                border: 1px solid rgba(16, 185, 129, 0.25);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                color: #065f46;
            }
            .demo-card {
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.1);
                padding: 2rem;
            }
            .slider-card {
                background: linear-gradient(135deg, #f9fafb 0%, #eef2ff 100%);
                border-radius: 12px;
                border: 1px solid rgba(99, 102, 241, 0.15);
                padding: 1rem 1.25rem 1.25rem;
                margin-bottom: 1rem;
            }
            .slider-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
            }
            .slider-label {
                font-size: 0.95rem;
                font-weight: 600;
                color: #1e1b4b;
            }
            .slider-value {
                font-weight: 600;
                font-variant-numeric: tabular-nums;
                color: #4c1d95;
            }
            .slider-body {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            .slider-body > div:first-child {
                flex: 1 1 auto;
            }
            .slider-body > div:last-child {
                width: 140px;
            }
            .footnote {
                margin-top: 1.75rem;
                color: #6b7280;
                font-size: 0.85rem;
            }
            .bordered-block {
                border: 1px solid #d1d5db;          /* light gray border */
                border-radius: 12px;
                padding: 1.5rem;
                background-color: #ffffff;          /* white card look */
                margin: 1.5rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def slider_with_number(
    label: str,
    *,
    min_value: float,
    max_value: float,
    step: float,
    default: float,
    format_str: str = "%.2f",
    key: str,
) -> float:
    """Simplified slider + number input combo block, minimal styling."""
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    def _sync_from_slider() -> None:
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_from_number() -> None:
        st.session_state[slider_key] = st.session_state[number_key]

    slider_col, number_col = st.columns([4, 1], gap="small")
    with slider_col:
        slider_value = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=default,
            key=slider_key,
            on_change=_sync_from_slider,
        )

    with number_col:
        number_value = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=default,
            format=format_str,
            key=number_key,
            on_change=_sync_from_number,
            label_visibility="hidden",
        )

    return float(st.session_state[number_key])


def main() -> None:
    configure_page()

    st.title("Response of first-order system under step inputs")
    st.markdown(
        """
        Interactively tune initial conditions and exogenous inputs for a closed-loop first-order plant.
        You can hover on the plot to get measurements.
        """
    )

    with st.container():
        st.markdown("#### Question 1")
        st.markdown(
            r"""
            Consider the feedback interconnection above where $A$, $B_{1}$, $B_{2}$, and $C$ are scalars and the initial condition of the system satisfies $x(0) = x_{0}$.

            **By hand, derive the steady-state gain from the step reference input $r(t) = \bar{r}$ to the output $y$ in terms of:**
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            &\text{The state matrix (scalar) } A \\
            &\text{The input matrix (scalar) } B_{1} \text{ for the input } u \\
            &\text{The proportional gain (scalar) } k_{p} \text{ for the input } u \\
            &\text{The input matrix (scalar) } B_{2} \text{ for the input } d \\
            &\text{The output matrix (scalar) } C \\
            &\text{The input magnitude } \bar{r} \text{ for the input } r \\
            &\text{The input magnitude } \bar{d} \text{ for the input } d
            \end{aligned}
            """
        )
        st.markdown(
            """
            Explain your reasoning and include any drawings that support your answers.
            """
        )

        st.markdown("---")

        st.markdown("#### Question 2")
        st.markdown(
            r"""
            Identify the parameters by setting the values of the initial condition $x(0)=x_{0}$, sinusoidal reference input $r(t) = \bar{r} \cos(\omega t)$, and step disturbance input $d(t) = \bar{d}$.
            """
        )
        st.markdown(
            r"""
            If any of the parameters is not uniquely identifiable, state the most precise condition for the parameter.  
            Explain your reasoning and include any plots that support your answers.
            """
        )

    st.markdown("---")

    with st.container():
        st.markdown("#### Interactive Plot")
        col1, col2 = st.columns(2)
        with col1:
            magnitude = slider_with_number(
                r"Reference input magnitude $\bar{r}$",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                default=1.0,
                key="magnitude",
            )
            disturbance = slider_with_number(
                r"Disturbance input magnitude $\bar{d}$",
                min_value=-1.5,
                max_value=1.5,
                step=0.01,
                default=0.5,
                key="disturbance",
            )

        with col2:
            omega = slider_with_number(
                r"Reference input frequency $\omega$",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                default=2.0,
                key="omega",
                format_str="%.1f",
            )
            x0 = slider_with_number(
                r"Initial state $x_{0}$",
                min_value=-2.0,
                max_value=2.0,
                step=0.01,
                default=-0.2,
                key="x0",
            )

        time, _x, y = compute_response(
            DEFAULT_CONSTANTS,
            magnitude=magnitude,
            omega=omega,
            disturbance=disturbance,
            x0=x0,
        )

        fig = render_plot(time, y)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
