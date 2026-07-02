import streamlit as st
import cv2
import numpy as np
import json
import os
from PIL import Image

from core.comparator import compare_drawings

st.set_page_config(page_title="Drawing QA Toolkit", layout="wide")

st.title("Drawing QA Toolkit")
st.caption("Three prototype checks for the drawing-review workflow — demo build")


def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_display(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def api_key_input(default_help=""):
    return st.text_input("Anthropic API key", type="password",
                          value=os.environ.get("ANTHROPIC_API_KEY", ""),
                          help=default_help or "Get one at console.anthropic.com. Required for all AI-based checks below.")


mode = st.sidebar.radio(
    "What do you want to check?",
    [
        "1. CET Revision Comparison",
        "2. Prototype Completeness Check",
        "3. Typo & Cross-View Consistency Check",
    ],
)

# =============================================================================
# MODE 1 — CET Revision Comparison (CV diff + AI change-request reconciliation)
# =============================================================================
if mode.startswith("1"):
    st.header("CET Revision Comparison")
    st.caption("Verify requested changes were applied, and flag anything that wasn't requested.")

    with st.sidebar:
        st.subheader("Inputs")
        use_sample = st.checkbox("Use sample drawings (no upload needed)", value=True)
        if not use_sample:
            master_file = st.file_uploader("Master Drawing", type=["png", "jpg", "jpeg"], key="m1_master")
            revision_file = st.file_uploader("New Revision", type=["png", "jpg", "jpeg"], key="m1_rev")
        else:
            master_file, revision_file = None, None

        st.subheader("Sensitivity")
        threshold = st.slider("Pixel difference threshold", 5, 100, 30)
        min_area = st.slider("Minimum region size (px²)", 50, 2000, 250)

        st.subheader("Change Request Verification (AI)")
        ai_enabled = st.checkbox("Reconcile diffs against requested changes", value=False)
        api_key = None
        requested_changes_text = ""
        if ai_enabled:
            api_key = api_key_input()
            requested_changes_text = st.text_area(
                "Requested changes (one per line)",
                value="Move the top-right bolt hole further right and down\n"
                      "Remove the keyway slot above the center bore\n"
                      "Add a new small hole to the left of center bore\n"
                      "Update the R25.4 dimension callout",
                height=140,
            )
        run_btn = st.button("Run Comparison", type="primary", use_container_width=True)

    if run_btn:
        if use_sample:
            master = cv2.imread("samples/master_drawing.png")
            revision = cv2.imread("samples/revised_drawing.png")
        else:
            if not master_file or not revision_file:
                st.error("Please upload both a master drawing and a revision, or check 'Use sample drawings'.")
                st.stop()
            master = load_image(master_file)
            revision = load_image(revision_file)

        with st.spinner("Aligning drawings and computing discrepancy map..."):
            result = compare_drawings(master, revision, threshold=threshold, min_area=min_area)

        reconciliation = None
        requested_changes = [c.strip() for c in requested_changes_text.splitlines() if c.strip()]
        if ai_enabled:
            if not api_key:
                st.error("Enter your Anthropic API key in the sidebar to run change-request verification.")
            elif not requested_changes:
                st.error("Enter at least one requested change in the sidebar.")
            elif not result.discrepancies:
                st.info("No pixel-level discrepancies were found, so there's nothing to reconcile.")
            else:
                from core.change_verifier import reconcile_changes
                with st.spinner(f"Reconciling {len(result.discrepancies)} region(s) against "
                                 f"{len(requested_changes)} requested change(s)..."):
                    try:
                        reconciliation = reconcile_changes(
                            master, result.aligned_revision, result.discrepancies,
                            requested_changes, api_key=api_key,
                        )
                    except Exception as e:
                        st.error(f"AI reconciliation failed: {e}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alignment", "OK" if result.aligned_ok else "FAILED")
        c2.metric("Match confidence", f"{result.match_confidence:.0%}")
        c3.metric("Discrepancies found", len(result.discrepancies))
        c4.metric("Processing time", f"{result.processing_time_s:.2f}s")

        if not result.aligned_ok:
            st.warning("Automatic alignment had low confidence — results may include false positives from "
                       "scan skew/offset rather than real design changes.")

        tab_names = ["Side by Side", "Discrepancy Map", "Annotated Revision", "Report (JSON)"]
        if reconciliation:
            tab_names.insert(0, "Change Verification")
        tabs = st.tabs(tab_names)
        tab_map = dict(zip(tab_names, tabs))

        if reconciliation:
            with tab_map["Change Verification"]:
                r = reconciliation.to_dict()
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Applied as requested", len(r["applied"]))
                cc2.metric("Unintended changes", len(r["unintended"]))
                cc3.metric("Requested but missing", len(r["missing"]))
                if r["applied"]:
                    st.success("**Applied as requested**")
                    st.table(r["applied"])
                if r["unintended"]:
                    st.error("**Unintended changes** — flagged but not part of the request")
                    st.table(r["unintended"])
                if r["missing"]:
                    st.warning("**Requested but not found** — CET may not have applied these")
                    for m in r["missing"]:
                        st.write(f"- {m}")
                if not r["unintended"] and not r["missing"]:
                    st.success("All detected changes match the request list, and nothing requested is missing.")

        with tab_map["Side by Side"]:
            st.image(cv2_to_display(result.side_by_side), caption="Master (left) vs. Revision (right)", use_container_width=True)
        with tab_map["Discrepancy Map"]:
            st.image(result.discrepancy_map, caption="Binary discrepancy mask", use_container_width=True)
        with tab_map["Annotated Revision"]:
            st.image(cv2_to_display(result.annotated_revision),
                     caption="Yellow=minor, orange=moderate, red=major", use_container_width=True)
        with tab_map["Report (JSON)"]:
            report = result.to_report_dict()
            if reconciliation:
                report["change_reconciliation"] = reconciliation.to_dict()
            st.json(report)
            st.download_button("Download report.json", data=json.dumps(report, indent=2),
                                file_name="drawing_comparison_report.json", mime="application/json")

        st.subheader("Flagged Changes")
        if result.discrepancies:
            st.table([{"ID": d.id, "Severity": d.severity, "Area (px²)": d.area_px, "Location (x,y,w,h)": d.bbox}
                       for d in result.discrepancies])
        else:
            st.success("No discrepancies detected above the current sensitivity threshold.")
    else:
        st.info("Set your options in the sidebar and click **Run Comparison** to get started.")


# =============================================================================
# MODE 2 — Prototype Completeness Check
# =============================================================================
elif mode.startswith("2"):
    st.header("Prototype Completeness Check")
    st.caption("For prototype requests: confirm no missing or ambiguous instructions "
               "(base fabric type, sewing method, panel positioning method, etc.)")

    with st.sidebar:
        st.subheader("Inputs")
        use_sample = st.checkbox("Use sample prototype drawing", value=True)
        if not use_sample:
            drawing_file = st.file_uploader("Prototype Drawing", type=["png", "jpg", "jpeg"], key="m2_drawing")
        else:
            drawing_file = None

        st.subheader("Required Instructions Checklist")
        from core.completeness_checker import DEFAULT_CHECKLIST
        checklist_text = st.text_area(
            "One item per line — edit to match your drawing standard",
            value="\n".join(DEFAULT_CHECKLIST),
            height=140,
        )
        api_key = api_key_input()
        run_btn2 = st.button("Run Completeness Check", type="primary", use_container_width=True)

    if run_btn2:
        if use_sample:
            drawing = cv2.imread("samples/prototype_drawing_sample.png")
        else:
            if not drawing_file:
                st.error("Upload a drawing or check 'Use sample prototype drawing'.")
                st.stop()
            drawing = load_image(drawing_file)

        checklist = [c.strip() for c in checklist_text.splitlines() if c.strip()]
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar.")
        elif not checklist:
            st.error("Add at least one checklist item.")
        else:
            from core.completeness_checker import check_completeness
            with st.spinner(f"Checking drawing against {len(checklist)} required instruction(s)..."):
                try:
                    result = check_completeness(drawing, checklist, api_key=api_key)
                except Exception as e:
                    st.error(f"Check failed: {e}")
                    result = None

            if result:
                r = result.to_dict()
                c1, c2, c3 = st.columns(3)
                c1.metric("Checklist items", len(r["items"]))
                c2.metric("Missing", r["missing_count"])
                c3.metric("Ambiguous", r["ambiguous_count"])

                st.image(cv2_to_display(drawing), caption="Drawing under review", use_container_width=True)

                for item in r["items"]:
                    if item["status"] == "present":
                        st.success(f"**{item['item']}** — present\n\n> {item['evidence']}")
                    elif item["status"] == "missing":
                        st.error(f"**{item['item']}** — MISSING\n\n{item['note']}")
                    else:
                        st.warning(f"**{item['item']}** — AMBIGUOUS\n\n{item['note']}")

                st.download_button("Download report.json", data=json.dumps(r, indent=2),
                                    file_name="completeness_report.json", mime="application/json")
    else:
        st.info("Set your checklist in the sidebar and click **Run Completeness Check** to get started. "
                "The sample drawing is missing a panel positioning instruction on purpose.")


# =============================================================================
# MODE 3 — Typo & Cross-View Consistency Check
# =============================================================================
else:
    st.header("Typo & Cross-View Consistency Check")
    st.caption("Find spelling mistakes and mismatches between the main drawing and its views/sections.")

    with st.sidebar:
        st.subheader("Inputs")
        use_sample = st.checkbox("Use sample drawing sheet", value=True)
        if not use_sample:
            drawing_file = st.file_uploader("Drawing (with views/sections)", type=["png", "jpg", "jpeg"], key="m3_drawing")
        else:
            drawing_file = None

        st.subheader("Known Domain Terms (won't be flagged as typos)")
        from core.consistency_checker import DEFAULT_DOMAIN_TERMS
        domain_terms_text = st.text_area(
            "One term per line",
            value="\n".join(DEFAULT_DOMAIN_TERMS),
            height=120,
        )
        api_key = api_key_input()
        run_btn3 = st.button("Run Consistency Check", type="primary", use_container_width=True)

    if run_btn3:
        if use_sample:
            drawing = cv2.imread("samples/prototype_drawing_sample.png")
        else:
            if not drawing_file:
                st.error("Upload a drawing or check 'Use sample drawing sheet'.")
                st.stop()
            drawing = load_image(drawing_file)

        domain_terms = [t.strip() for t in domain_terms_text.splitlines() if t.strip()]
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar.")
        else:
            from core.consistency_checker import check_consistency
            with st.spinner("Reading all views and checking for typos and mismatches..."):
                try:
                    result = check_consistency(drawing, domain_terms, api_key=api_key)
                except Exception as e:
                    st.error(f"Check failed: {e}")
                    result = None

            if result:
                r = result.to_dict()
                c1, c2 = st.columns(2)
                c1.metric("Typos found", len(r["typos"]))
                c2.metric("Cross-view mismatches", len(r["mismatches"]))

                st.image(cv2_to_display(drawing), caption="Drawing under review", use_container_width=True)

                if r["typos"]:
                    st.subheader("Typos")
                    st.table(r["typos"])
                else:
                    st.success("No typos found.")

                if r["mismatches"]:
                    st.subheader("Cross-View Mismatches")
                    st.table(r["mismatches"])
                else:
                    st.success("No cross-view mismatches found.")

                st.download_button("Download report.json", data=json.dumps(r, indent=2),
                                    file_name="consistency_report.json", mime="application/json")
    else:
        st.info("Click **Run Consistency Check** to get started. The sample sheet has a planted typo "
                "(\"seem allowance\") and a planted mismatch (panel width 120mm vs 125mm).")
