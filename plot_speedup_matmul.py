from struphy.gpu.test_pyccel_timings import compare_pyccel_cpu_gpu
import numpy as np
import plotly.graph_objects as go
import struphy.post_processing.likwid.maxplotlylib as mply

if __name__ == '__main__':
    run_tests = False
    if run_tests:
        # Warm‑up
        _ , _ = compare_pyccel_cpu_gpu(Nvec=[2,4,8], max_time=1.0)

        # Actual timings
        iterations = 10
        Nvec = np.unique(np.logspace(2, 12, num=100, base=2).astype(int))
        elapsed_cpu_vec, elapsed_gpu_vec = compare_pyccel_cpu_gpu(Nvec=Nvec, max_time=1.0)

        # Convert to numpy arrays for easy slicing
        elapsed_cpu_vec = np.array(elapsed_cpu_vec)
        elapsed_gpu_vec = np.array(elapsed_gpu_vec)

        np.save("/u/maxlin/git_repos/struphy/elapsed_cpu_vec.npy", elapsed_cpu_vec)
        np.save("/u/maxlin/git_repos/struphy/elapsed_gpu_vec.npy", elapsed_gpu_vec)

    elapsed_cpu_vec = np.load("/u/maxlin/git_repos/struphy/elapsed_cpu_vec.npy")
    elapsed_gpu_vec = np.load("/u/maxlin/git_repos/struphy/elapsed_gpu_vec.npy")

    speedup = elapsed_cpu_vec[:, 1] / elapsed_gpu_vec[:, 1]
    print(f"{speedup = }")
    # Build Plotly figure
    fig = go.Figure()

    # CPU trace
    fig.add_trace(go.Scatter(
        x=elapsed_cpu_vec[:, 0],
        y=elapsed_cpu_vec[:, 1],
        mode='lines+markers',
        line=dict(color='red', dash='dash'),
        name='CPU'
    ))

    # GPU trace
    fig.add_trace(go.Scatter(
        x=elapsed_gpu_vec[:, 0],
        y=elapsed_gpu_vec[:, 1],
        mode='lines+markers',
        line=dict(color='green'),
        name='GPU'
    ))

    fig.add_trace(go.Scatter(
        x=elapsed_gpu_vec[:, 0], y=speedup,
        mode='lines+markers',
        name='Speedup (CPU / GPU)',
        line=dict(color='blue'),
        yaxis='y2'
    ))

    # Layout
    fig.update_layout(
        # title="Pyccel CPU vs GPU matmul timings",
        xaxis_title="Matrix size N",
        yaxis_title="Elapsed time (s)",
        # legend_title="Backend",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=0, b=0, l=0, r=0),
        yaxis2=dict(
        title="Speedup (CPU / GPU)",
            # type="log",
            overlaying='y',
            side='right',
            tickformat="none",
            showgrid=False,
            color="red",
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            linecolor='red',
            tickcolor='red',
            # range=[0,50],
        ),
    )

    mply.format_size(fig,width=800, height=500)
    mply.format_axes(fig)
    mply.format_font(fig)
    mply.format_grid(fig)

    # Show interactively
    # fig.show()

    # Save to file if you like:
    fig.write_html("timings.html")
    fig.write_image("timings.pdf")


    import matplotlib.pyplot as plt
    plt.plot(elapsed_cpu_vec[:,0], elapsed_cpu_vec[:,1],'.-', label = 'CPU')
    plt.plot(elapsed_gpu_vec[:,0], elapsed_gpu_vec[:,1],'.-', label = 'GPU')
    plt.show()
    plt.savefig('test.pdf')
