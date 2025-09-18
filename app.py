import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import io
from fpdf import FPDF
import os
from datetime import datetime

# ------------------------------
# Job and Grid Simulation Classes
# ------------------------------
class Job:
    def __init__(self, jid, arrival, duration, priority=1):
        self.jid = jid
        self.arrival = arrival
        self.duration = duration
        self.priority = priority
        self.start = None
        self.finish = None
        self.orig_duration = duration  # keep original for reports

    def copy(self):
        return Job(self.jid, self.arrival, self.orig_duration, self.priority)

class GridNode:
    def __init__(self, nid):
        self.nid = nid
        self.time = 0
        self.jobs = []

    def assign_job(self, job, current_time):
        job.start = max(current_time, self.time)
        job.finish = job.start + job.duration
        self.jobs.append(job)
        self.time = job.finish

    def reset(self):
        self.time = 0
        self.jobs = []

# ------------------------------
# Utility helpers
# ------------------------------
def reset_job_copies(base_jobs):
    """Return fresh copies of Job objects so runs are independent."""
    return [j.copy() for j in base_jobs]

def reset_nodes(num_nodes):
    return [GridNode(i) for i in range(num_nodes)]

# ------------------------------
# Scheduling Algorithms
# ------------------------------
def fcfs_scheduler(jobs, nodes):
    jobs.sort(key=lambda j: j.arrival)
    node_idx = 0
    for job in jobs:
        nodes[node_idx].assign_job(job, job.arrival)
        node_idx = (node_idx + 1) % len(nodes)
    return nodes

def round_robin_scheduler(jobs, nodes, quantum=2):
    # We operate on a shallow copy of durations so original durations preserved externally
    for j in jobs:
        j.remaining = j.duration
        j.start = None
        j.finish = None

    time_now = 0
    queue = sorted(jobs, key=lambda x: x.arrival)
    ready = []
    idx = 0
    while queue or ready:
        # move arrived jobs to ready
        while queue and queue[0].arrival <= time_now:
            ready.append(queue.pop(0))
        if not ready:
            # fast-forward if no ready jobs
            if queue:
                time_now = queue[0].arrival
                continue
            else:
                break
        job = ready.pop(0)
        exec_time = min(job.remaining, quantum)
        if job.start is None:
            job.start = time_now
        time_now += exec_time
        job.remaining -= exec_time
        if job.remaining <= 0:
            job.finish = time_now
            # assign to least-loaded node (by current time)
            node = min(nodes, key=lambda n: n.time)
            # but we need consistent start/finish on node timeline
            # We'll set job.start/finish relative to node
            node.assign_job(job, job.start)
        else:
            # push newly arrived jobs first before re-queue
            while queue and queue[0].arrival <= time_now:
                ready.append(queue.pop(0))
            ready.append(job)
    return nodes

def space_sharing_scheduler(jobs, nodes):
    jobs.sort(key=lambda j: j.duration)  # shortest job first across nodes
    for idx, job in enumerate(jobs):
        nodes[idx % len(nodes)].assign_job(job, job.arrival)
    return nodes

def time_sharing_scheduler(jobs, nodes):
    # assign each job to node with minimum current time -> simulates time-sharing load balancing
    for job in jobs:
        node = min(nodes, key=lambda n: n.time)
        node.assign_job(job, job.arrival)
    return nodes

# ------------------------------
# Simulated Annealing Scheduler (improved)
# ------------------------------
def cost_function(nodes):
    return max((n.time for n in nodes), default=0)

def random_schedule_assignments(jobs, nodes):
    for n in nodes:
        n.reset()
    for job in jobs:
        # create a fresh job copy for assignment to avoid cross-run mutations
        node = random.choice(nodes)
        node.assign_job(job, job.arrival)
    return nodes

def simulated_annealing(jobs, nodes, T=200.0, cooling=0.92, max_iter=500):
    # initial random solution
    current_nodes = reset_nodes(len(nodes))
    current_nodes = random_schedule_assignments(jobs, current_nodes)
    best_nodes = [GridNode(n.nid) for n in current_nodes]
    # deep copy jobs in nodes to best_nodes
    for i, n in enumerate(current_nodes):
        best_nodes[i].jobs = list(n.jobs)
        best_nodes[i].time = n.time
    best_cost = cost_function(best_nodes)
    cur_cost = best_cost
    cur_nodes = current_nodes

    for it in range(max_iter):
        # propose neighbor: swap two jobs between nodes or move one job
        new_nodes = [GridNode(n.nid) for n in nodes]
        # flatten jobs, shuffle, distribute randomly
        all_jobs = []
        for n in cur_nodes:
            for j in n.jobs:
                # create shallow copy to avoid reference carryover
                job_copy = Job(j.jid, j.arrival, j.orig_duration, j.priority)
                all_jobs.append(job_copy)
        random.shuffle(all_jobs)
        # random redistribution with some bias to keep structure
        for job in all_jobs:
            new_nodes[random.randrange(len(new_nodes))].assign_job(job, job.arrival)

        new_cost = cost_function(new_nodes)
        delta = new_cost - cur_cost
        accept = False
        if delta < 0 or random.random() < math.exp(-delta / max(1e-9, T)):
            accept = True

        if accept:
            cur_nodes = new_nodes
            cur_cost = new_cost
            if new_cost < best_cost:
                best_nodes = new_nodes
                best_cost = new_cost

        T *= cooling
        # early exit
        if T < 1e-3:
            break

    # final best_nodes may contain Job.duration modified? They contain start/finish set earlier.
    return best_nodes

# ------------------------------
# Metrics & Reporting
# ------------------------------
def compute_metrics(jobs, nodes):
    waiting_times = []
    turnaround_times = []
    finish_times = []
    total_busy_time = 0

    for n in nodes:
        for j in n.jobs:
            waiting_times.append(j.start - j.arrival)
            turnaround_times.append(j.finish - j.arrival)
            finish_times.append(j.finish)
            total_busy_time += (j.finish - j.start)

    makespan = max(finish_times) if finish_times else 0
    avg_waiting = sum(waiting_times) / len(waiting_times) if waiting_times else 0
    avg_turnaround = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0
    throughput = len(jobs) / makespan if makespan > 0 else 0
    utilization = total_busy_time / (makespan * len(nodes)) if makespan > 0 and len(nodes) > 0 else 0

    return {
        "Makespan": makespan,
        "Average Waiting Time": avg_waiting,
        "Average Turnaround Time": avg_turnaround,
        "Throughput": throughput,
        "CPU Utilization": utilization
    }

# ------------------------------
# Visualization helpers
# ------------------------------
def plot_schedule(nodes, title="Schedule"):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(nodes))))
    cmap = plt.get_cmap("tab20")
    color_idx = 0
    for i, node in enumerate(nodes):
        for job in node.jobs:
            ax.barh(i, job.finish - job.start, left=job.start,
                    height=0.6, color=cmap(color_idx % 20), edgecolor="black")
            ax.text(job.start + (job.finish - job.start)/2, i, f"J{job.jid}", ha="center", va="center", color="black", fontsize=8)
            color_idx += 1
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels([f"Node {n.nid}" for n in nodes])
    ax.set_xlabel("Time")
    ax.set_ylabel("Grid Nodes")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_bar(values_dict, metric_name):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    algos = list(values_dict.keys())
    vals = [values_dict[a] for a in algos]
    ax.bar(algos, vals, edgecolor="black")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name + " Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig

# ------------------------------
# PDF builder
# ------------------------------
def build_pdf_report(job_table_df, summary_metrics, chart_images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Grid Scheduling Simulator Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Summary Metrics (per algorithm)", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    for algo, mets in summary_metrics.items():
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, algo, ln=True)
        pdf.set_font("Arial", size=10)
        for k, v in mets.items():
            pdf.cell(0, 6, f"  {k}: {round(v, 4)}", ln=True)
        pdf.ln(2)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Job Details (sample)", ln=True)
    pdf.ln(4)
    # include first 40 rows of job_table_df
    pdf.set_font("Arial", size=8)
    # create a simple table header
    cols = ["jid", "arrival", "duration", "start", "finish", "node"]
    header = " | ".join(cols)
    pdf.multi_cell(0, 5, header)
    pdf.ln(1)
    for _, row in job_table_df.head(40).iterrows():
        line = " | ".join(str(row[c]) for c in cols)
        pdf.multi_cell(0, 5, line)
    pdf.add_page()

    for img in chart_images:
        pdf.image(img, x=10, w=190)
        pdf.ln(4)

    # output bytes
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Grid Scheduling Simulator", layout="wide")
st.title("⚡ Grid Scheduling Simulator — Full Project (SA + Metrics + Comparison + Export)")
st.markdown("**Covering FI1966 (Grid Scheduling)** — FCFS, Round Robin, Space Sharing, Time Sharing, Simulated Annealing, and Queuing Metrics (Makespan, Waiting Time, Turnaround, Throughput, Utilization).")

# Controls
with st.sidebar:
    st.header("Simulation Controls")
    num_jobs = st.slider("Number of Jobs", 5, 40, 12)
    num_nodes = st.slider("Number of Grid Nodes", 1, 8, 3)
    seed = st.number_input("Random Seed (0 = random each run)", min_value=0, value=0)
    include_plots_in_pdf = st.checkbox("Include plots in PDF", value=True)
    st.markdown("---")
    st.write("Simulated Annealing parameters")
    sa_T = st.number_input("SA Initial Temperature", value=200.0)
    sa_cooling = st.slider("SA Cooling Factor", 0.80, 0.99, 0.92)
    sa_iters = st.number_input("SA Max Iterations", min_value=50, max_value=2000, value=400, step=50)

mode = st.radio("Mode", ["Run Single Algorithm", "Compare All Algorithms"], horizontal=True)

if seed != 0:
    random.seed(seed)

# Generate base job set (kept immutable by copying for each algorithm)
base_jobs = []
for i in range(num_jobs):
    arrival = random.randint(0, max(0, num_jobs // 3))
    duration = random.randint(1, max(2, num_jobs // 3 + 2))
    base_jobs.append(Job(i, arrival, duration))

algorithms = {
    "FCFS": lambda jobs, nodes: fcfs_scheduler(jobs, nodes),
    "Round Robin": lambda jobs, nodes: round_robin_scheduler(jobs, nodes, quantum=2),
    "Space Sharing": lambda jobs, nodes: space_sharing_scheduler(jobs, nodes),
    "Time Sharing": lambda jobs, nodes: time_sharing_scheduler(jobs, nodes),
    "Simulated Annealing": lambda jobs, nodes: simulated_annealing(jobs, nodes, T=sa_T, cooling=sa_cooling, max_iter=sa_iters)
}

# Execution & Presentation
if mode == "Run Single Algorithm":
    algo = st.selectbox("Choose Scheduling Algorithm", list(algorithms.keys()))
    if st.button("Run Simulation"):
        jobs = reset_job_copies(base_jobs)
        nodes = reset_nodes(num_nodes)
        scheduled = algorithms[algo](jobs, nodes)

        # collect job table
        job_rows = []
        for n in scheduled:
            for j in n.jobs:
                job_rows.append({
                    "jid": j.jid,
                    "arrival": j.arrival,
                    "duration": j.orig_duration,
                    "start": j.start,
                    "finish": j.finish,
                    "node": n.nid
                })
        job_df = pd.DataFrame(job_rows).sort_values(by=["node", "start"])

        metrics = compute_metrics(jobs, scheduled)
        st.subheader(f"{algo} — Schedule Visualization")
        fig = plot_schedule(scheduled, title=f"{algo} Scheduling")
        st.pyplot(fig)

        st.subheader("📊 Queue Performance Metrics")
        for k, v in metrics.items():
            st.metric(k, f"{round(v,4)}")

        st.subheader("Job Table")
        st.dataframe(job_df)

        # CSV download
        csv_buf = job_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Job Table (CSV)", csv_buf, file_name=f"jobs_{algo}.csv", mime="text/csv")

        # PDF report with single algorithm
        if st.button("Export PDF Report"):
            # Save schedule image
            tmp_imgs = []
            with tempfile.TemporaryDirectory() as td:
                img_path = os.path.join(td, f"{algo}_schedule.png")
                fig.savefig(img_path, bbox_inches="tight")
                tmp_imgs.append(img_path)

                # build summary_metrics in dict shape
                summary_metrics = {algo: metrics}
                pdf_buf = build_pdf_report(job_df, summary_metrics, tmp_imgs if include_plots_in_pdf else [])
                st.download_button("Download PDF Report", pdf_buf.read(), file_name=f"report_{algo}.pdf", mime="application/pdf")

elif mode == "Compare All Algorithms":
    if st.button("Run Comparison"):
        comparison_results = {}
        job_tables = {}
        chart_images = []
        with st.spinner("Running algorithms..."):
            for algo_name, func in algorithms.items():
                jobs = reset_job_copies(base_jobs)
                nodes = reset_nodes(num_nodes)
                scheduled = func(jobs, nodes)
                metrics = compute_metrics(jobs, scheduled)
                comparison_results[algo_name] = metrics

                # job table for the algorithm
                rows = []
                for n in scheduled:
                    for j in n.jobs:
                        rows.append({
                            "jid": j.jid,
                            "arrival": j.arrival,
                            "duration": j.orig_duration,
                            "start": j.start,
                            "finish": j.finish,
                            "node": n.nid
                        })
                job_tables[algo_name] = pd.DataFrame(rows).sort_values(by=["node", "start"])

                # save visual schedule image
                fig_alg = plot_schedule(scheduled, title=f"{algo_name} Scheduling")
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig_alg.savefig(tmpf.name, bbox_inches="tight")
                chart_images.append(tmpf.name)
                plt.close(fig_alg)

        st.subheader("📊 Algorithm Comparison — Metrics")
        # show metrics table
        metrics_df = pd.DataFrame(comparison_results).T
        st.dataframe(metrics_df.style.format("{:.4f}"))

        # show bar charts for each metric
        metric_names = ["Makespan", "Average Waiting Time", "Average Turnaround Time", "Throughput", "CPU Utilization"]
        saved_chart_paths = []
        for metric in metric_names:
            vals = {algo: comparison_results[algo][metric] for algo in algorithms.keys()}
            figm = plot_bar(vals, metric)
            st.pyplot(figm)
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            figm.savefig(tmpf.name, bbox_inches="tight")
            saved_chart_paths.append(tmpf.name)
            plt.close(figm)

        # allow per-algo job table download as zipped CSVs? (simple single CSV per algo)
        st.subheader("Download Results")
        for algo_name, df in job_tables.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(f"Download {algo_name} job CSV", csv_bytes, file_name=f"jobs_{algo_name}.csv", mime="text/csv")

        # Generate combined PDF report
        if st.button("Export Combined PDF Report"):
            pdf_buf = build_pdf_report(
                job_table_df=pd.concat(list(job_tables.values())).reset_index(drop=True),
                summary_metrics=comparison_results,
                chart_images=(chart_images + saved_chart_paths) if include_plots_in_pdf else []
            )
            st.download_button("Download Combined PDF Report", pdf_buf.read(), file_name="grid_scheduling_report.pdf", mime="application/pdf")

        st.success("✅ Comparison Completed!")

st.markdown("---")
st.markdown("**Notes & Tips:**")
st.markdown("- Use the sidebar to tune SA hyperparameters (Initial Temp, Cooling, Iterations).")
st.markdown("- Use the seed input to reproduce runs when preparing your demo.")
st.markdown("- Deploy on Streamlit Cloud or run locally with `streamlit run app.py`.")
