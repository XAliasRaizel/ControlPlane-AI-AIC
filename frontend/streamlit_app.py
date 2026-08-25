import os
import time
import requests
import streamlit as st

API = os.getenv("CONTROLPLANE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ControlPlane.ai — AI Governance & Chatbot",
    page_icon="🛡️",
    layout="wide",
)

# Custom header
st.title("🛡️ ControlPlane.ai")
st.caption("Enterprise AI Governance Control Plane — Parallel Hot Path · Policy Engine · Async Deep Analysis · LLM Chat")

# Session state initialization
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI assistant protected by ControlPlane.ai governance. How can I help you today?",
            "governance": None,
            "async_data": None,
        }
    ]
if "last_gov_result" not in st.session_state:
    st.session_state.last_gov_result = None
if "last_async_result" not in st.session_state:
    st.session_state.last_async_result = None
if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = {}

# Sidebar — Request & Security Context
with st.sidebar:
    st.header("👤 Caller & Security Context")
    user_id = st.text_input("User ID", "employee-101")
    application_id = st.selectbox(
        "Application",
        ["support-bot", "hr-copilot", "loan-decision", "hiring-decision", "medical-decision"],
    )
    department = st.text_input("Department", "HR")
    user_role = st.selectbox("User Role", ["employee", "hr-manager", "finance-manager", "doctor", "admin"])
    data_classification = st.selectbox("Data Classification", ["PUBLIC", "INTERNAL", "HIGH", "RESTRICTED"])

    st.divider()
    st.subheader("⚙️ Gateway Config")
    api_key = st.text_input("API Key", "demo-key-001", type="password")

    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you?",
                "governance": None,
                "async_data": None,
            }
        ]
        st.session_state.last_gov_result = None
        st.session_state.last_async_result = None
        st.rerun()


def fetch_async_analysis(job_id: str, timeout: float = 3.0):
    """Fetch async deep analysis result from the gateway."""
    if not job_id:
        return None
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API}/v1/jobs/{job_id}", timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "COMPLETED" and data.get("result"):
                    return data["result"]
        except Exception:
            pass
        time.sleep(0.3)
    return None


tab_chat, tab_manual, tab_metrics, tab_policies, tab_reviews = st.tabs([
    "💬 Governance Chatbot",
    "🔬 Advanced Inspector",
    "📊 Platform Metrics",
    "📜 Policy Rules",
    "🗂️ Review Queue",
])


# ==============================================================================
# TAB 1: GOVERNANCE CHATBOT
# ==============================================================================
with tab_chat:
    col_chat, col_telemetry = st.columns([1.2, 0.8])

    with col_chat:
        st.subheader("Interactive AI Chatbot")
        st.caption("Every message is evaluated in real-time by hot-path detectors and policy rules.")

        # Render chat messages
        for msg_idx, msg in enumerate(st.session_state.chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # If assistant message has governance metadata, show summary pills
                gov = msg.get("governance")
                if gov:
                    action = gov["decision"]["action"]
                    risk = gov["risk"]["overall_risk"]
                    latency = gov["latency_ms"]

                    if action == "BLOCK":
                        st.error(f"🛑 Decision: **BLOCK** | Policy: `{gov['policy']['policy_id']}` | Latency: `{latency}ms`")
                    elif action == "MODIFY":
                        st.warning(f"⚠️ Decision: **MODIFY (Sanitized)** | Risk: `{risk:.2f}` | Latency: `{latency}ms`")
                    elif action == "HUMAN_REVIEW":
                        st.info(f"⏳ Decision: **HUMAN REVIEW** | Risk: `{risk:.2f}`")
                    else:
                        st.success(f"✅ Decision: **ALLOW** | Risk: `{risk:.2f}` | Latency: `{latency}ms`")

        # Chat Input
        if user_prompt := st.chat_input("Ask a question (e.g. 'How do I reset my password?', 'Show Rahul's salary', 'hack HR data')..."):
            # Add user message to UI
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt, "governance": None, "async_data": None})

            # Call /v1/chat API
            payload = {
                "user_id": user_id,
                "user_role": user_role,
                "department": department,
                "application_id": application_id,
                "prompt": user_prompt,
                "data_classification": data_classification,
            }

            try:
                with st.spinner("Evaluating governance & generating response..."):
                    res = requests.post(f"{API}/v1/chat", json=payload, headers={"x-api-key": api_key}, timeout=10)
                    res.raise_for_status()
                    chat_data = res.json()

                    gov_info = chat_data["governance"]
                    st.session_state.last_gov_result = gov_info

                    # Fetch async results
                    async_data = None
                    if gov_info.get("async_job_id"):
                        async_data = fetch_async_analysis(gov_info["async_job_id"], timeout=1.5)
                        st.session_state.last_async_result = async_data

                    # Append assistant response
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": chat_data["message"],
                        "governance": gov_info,
                        "async_data": async_data,
                    })

                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Gateway Error: {exc}")

    # Right Column: Real-time Telemetry for the latest turn
    with col_telemetry:
        st.subheader("🛡️ Real-Time Governance Telemetry")
        last_gov = st.session_state.last_gov_result

        if last_gov:
            action = last_gov["decision"]["action"]

            # Top Decision Card
            if action == "BLOCK":
                st.error(f"### 🛑 Decision: BLOCK\n**Rule**: `{last_gov['policy']['policy_id']}`\n\n**Reason**: {last_gov['decision']['reason']}")
            elif action == "MODIFY":
                st.warning(f"### ⚠️ Decision: MODIFY\n**Rule**: `{last_gov['policy']['policy_id']}`\n\n**Reason**: {last_gov['decision']['reason']}")
            elif action == "HUMAN_REVIEW":
                st.info(f"### ⏳ Decision: HUMAN REVIEW\n**Rule**: `{last_gov['policy']['policy_id']}`\n\n**Reason**: {last_gov['decision']['reason']}")
            else:
                st.success(f"### ✅ Decision: ALLOW\n**Rule**: `{last_gov['policy']['policy_id']}`\n\nRequest conforms to all security policies.")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Risk", f"{last_gov['risk']['overall_risk']:.3f}")
            m2.metric("Confidence", f"{last_gov['risk']['confidence']:.2f}")
            m3.metric("Hot Path Latency", f"{last_gov['latency_ms']} ms")

            # Detectors expander
            with st.expander("🔍 Hot-Path Detectors (Parallel Execution)", expanded=True):
                for det in last_gov["detectors"]:
                    d_col1, d_col2 = st.columns([1, 1])
                    with d_col1:
                        st.markdown(f"**{det['detector_name'].upper()}**")
                        st.caption(f"Latency: {det.get('latency_ms', 0):.2f}ms")
                    with d_col2:
                        label_color = "red" if det["score"] >= 0.7 else ("orange" if det["score"] > 0 else "green")
                        st.markdown(f":{label_color}[{det['label']} ({det['score']:.2f})]")
                    if det.get("evidence"):
                        st.caption(f"Evidence: `{det['evidence']}`")
                    st.divider()

            # Async Deep Analysis Section
            with st.expander("⚡ Async Deep Analysis (Non-blocking)", expanded=True):
                last_async = st.session_state.last_async_result
                if not last_async and last_gov.get("async_job_id"):
                    # Retry fetching
                    last_async = fetch_async_analysis(last_gov["async_job_id"], timeout=1.0)
                    st.session_state.last_async_result = last_async

                if last_async and "analytics" in last_async:
                    analytics = last_async["analytics"]
                    st.success("✅ Deep analysis completed asynchronously")
                    for eng_name, eng_val in analytics.items():
                        status = eng_val.get("status", "OK")
                        score = eng_val.get("score", 0.0)
                        evidence = eng_val.get("evidence", [])

                        # Color-coded status badge
                        status_color = "red" if status in ["HIGH", "CRITICAL"] else ("orange" if status in ["MEDIUM", "UNKNOWN"] else "green")
                        st.markdown(f"• **{eng_name.replace('_', ' ').title()}**: :{status_color}[**{status}**] `(Score: {score:.2f})`")
                        if evidence:
                            for ev in evidence:
                                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {ev}")
                else:
                    st.info(f"Job ID: `{last_gov.get('async_job_id')}`\n\nProcessing deep analysis in background...")

            # Human Feedback
            with st.expander("✍️ Reviewer Feedback Loop"):
                req_id = last_gov["request_id"]
                fb_action = st.selectbox("Correct Action", ["BLOCK", "ALLOW", "MODIFY", "HUMAN_REVIEW"], key="fb_chat_act")
                fb_comment = st.text_input("Reviewer Notes", key="fb_chat_notes")
                if st.button("Submit Feedback", key="fb_chat_btn"):
                    try:
                        f_res = requests.post(
                            f"{API}/v1/feedback",
                            json={
                                "request_id": req_id,
                                "original_action": last_gov["decision"]["action"],
                                "final_action": fb_action,
                                "notes": fb_comment,
                            },
                            headers={"x-api-key": api_key},
                            timeout=5,
                        )
                        if f_res.status_code == 200:
                            st.success("Feedback recorded for threshold optimization!")
                    except Exception as e:
                        st.error(f"Error submitting feedback: {e}")
        else:
            st.info("💡 Send a message in the chat to see real-time hot-path detector scores, policy decisions, and async deep analysis.")


# ==============================================================================
# TAB 2: ADVANCED INSPECTOR (Manual Test Form)
# ==============================================================================
with tab_manual:
    st.subheader("Manual AI Interaction Inspector")
    st.caption("Inspect candidate responses, retrieved RAG documents, and custom payloads.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        m_prompt = st.text_area("Prompt", "Give me Rahul's salary and personal phone number.", height=100, key="m_prompt")
        m_response = st.text_area("Candidate AI Response (optional)", "Rahul's salary is $85,000. Contact: rahul@company.com or +91 9876543210.", height=100, key="m_resp")
        m_retrieved = st.text_area("Retrieved Context / RAG Docs (optional)", "HR Policy: Salary information is restricted to authorized HR managers.", height=80, key="m_ret")

        if st.button("🚀 Run Governance Inspection", type="primary", key="m_btn"):
            m_payload = {
                "user_id": user_id,
                "application_id": application_id,
                "department": department,
                "user_role": user_role,
                "prompt": m_prompt,
                "response": m_response if m_response else None,
                "retrieved_context": [m_retrieved] if m_retrieved else [],
                "data_classification": data_classification,
            }
            try:
                with st.spinner("Evaluating..."):
                    m_res = requests.post(f"{API}/v1/govern", json=m_payload, headers={"x-api-key": api_key}, timeout=10)
                    m_res.raise_for_status()
                    st.session_state.m_data = m_res.json()
                    if st.session_state.m_data.get("async_job_id"):
                        st.session_state.m_async = fetch_async_analysis(st.session_state.m_data["async_job_id"], timeout=1.5)
            except Exception as e:
                st.error(f"Error: {e}")

    with col_out:
        if "m_data" in st.session_state:
            d = st.session_state.m_data
            act = d["decision"]["action"]

            if act == "BLOCK":
                st.error(f"### Decision: BLOCK\n**Rule**: `{d['policy']['policy_id']}`\n\n{d['decision']['reason']}")
            elif act == "MODIFY":
                st.warning(f"### Decision: MODIFY (Sanitized)\n**Rule**: `{d['policy']['policy_id']}`\n\n{d['decision']['reason']}")
            else:
                st.success(f"### Decision: ALLOW\n**Rule**: `{d['policy']['policy_id']}`")

            st.metric("Overall Risk", f"{d['risk']['overall_risk']:.3f}")
            st.metric("Latency", f"{d['latency_ms']} ms")

            if d.get("sanitized_response"):
                st.subheader("Controlled / Sanitized Output")
                st.code(d["sanitized_response"])

            st.subheader("Raw Governance Output")
            st.json(d)

            if "m_async" in st.session_state and st.session_state.m_async:
                st.subheader("⚡ Async Deep Analysis Results")
                st.json(st.session_state.m_async)


# ==============================================================================
# TAB 3: PLATFORM METRICS
# ==============================================================================
with tab_metrics:
    st.subheader("📊 ControlPlane.ai Telemetry & Metrics")
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    try:
        metrics = requests.get(f"{API}/v1/metrics", timeout=5).json()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Requests", metrics.get("total_requests", 0))
        c2.metric("Blocked", metrics.get("blocked", 0))
        c3.metric("Modified", metrics.get("modified", 0))
        c4.metric("Human Review", metrics.get("human_review", 0))
        c5.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.2f} ms")
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")

    st.divider()
    st.subheader("Recent Audit Records")
    try:
        audits = requests.get(f"{API}/v1/audits?limit=10", timeout=5).json()
        if audits:
            for item in audits:
                with st.expander(f"Request `{item['request_id'][:8]}...` — Decision: {item.get('decision_details', {}).get('action', 'N/A')}"):
                    st.json(item)
        else:
            st.info("No audit records found yet.")
    except Exception as e:
        st.caption(f"Audits unavailable: {e}")


# ==============================================================================
# TAB 4: POLICY RULES
# ==============================================================================
with tab_policies:
    st.subheader("📜 Active Policy Rules & Hierarchy")
    st.caption("Precedence: Application Scope > Department Scope > Global Policy")

    try:
        pol = requests.get(f"{API}/v1/policies", timeout=5).json()
        st.write(f"**Policy Set:** `{pol.get('policy_name')}` (v{pol.get('policy_version')})")
        st.dataframe(pol.get("rules", []), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not fetch policies: {e}")


# ==============================================================================
# TAB 5: REVIEW QUEUE
# ==============================================================================
# NEW: HUMAN_REVIEW decisions used to be auto-downgraded to BLOCK before this
# fix (see backend/review/queue.py) because there was nowhere for them to
# land. Now they're held as genuinely pending, and this tab is where a human
# actually resolves them -- the missing last step in the decision loop.
with tab_reviews:
    st.subheader("🗂️ Human Review Queue")
    st.caption("Requests routed to HUMAN_REVIEW are held here — nothing is silently auto-blocked.")

    if st.button("🔄 Refresh Queue"):
        st.rerun()

    try:
        pending = requests.get(f"{API}/v1/reviews", timeout=5).json()
    except Exception as e:
        pending = []
        st.warning(f"Could not load review queue: {e}")

    if not pending:
        st.info("No pending reviews. 🎉")
    else:
        for item in pending:
            with st.container(border=True):
                st.markdown(
                    f"**Request** `{item['request_id'][:8]}...`  ·  "
                    f"**Risk:** `{item['risk']:.2f}`  ·  **Policy:** `{item['policy_id']}`"
                )
                st.caption(item["reason"])
                c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                reviewer = c1.text_input("Reviewer", "reviewer", key=f"rv_{item['request_id']}")
                action = c2.selectbox(
                    "Resolve as", ["BLOCK", "ALLOW", "MODIFY", "REROUTE"], key=f"act_{item['request_id']}"
                )
                notes = c4.text_input("Notes", key=f"notes_{item['request_id']}")
                if c3.button("Resolve", key=f"resolve_{item['request_id']}"):
                    try:
                        res = requests.post(
                            f"{API}/v1/reviews/{item['request_id']}/resolve",
                            json={"reviewer_id": reviewer, "final_action": action, "notes": notes},
                            timeout=5,
                        )
                        if res.status_code == 200:
                            st.success("Resolved.")
                            st.rerun()
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Error: {e}")

